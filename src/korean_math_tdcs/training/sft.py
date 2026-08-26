from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from tqdm.auto import tqdm

from korean_math_tdcs.data.difficulty import validate_primary_counts
from korean_math_tdcs.data.formatting import AssistantOnlyCollator, verify_serialization
from korean_math_tdcs.data.loading import load_reasoning_sft
from korean_math_tdcs.data.sampling import (
    LevelPoolSampler,
    optimizer_batches,
    random_epoch_indices,
    resolved_training_budget,
    stage_boundaries,
)
from korean_math_tdcs.training.callbacks import RunTimer
from korean_math_tdcs.training.tdcs_trainer import TDCSState
from korean_math_tdcs.utils.config import save_config
from korean_math_tdcs.utils.io import (
    append_jsonl,
    ensure_dir,
    git_commit,
    utc_timestamp,
    write_json,
)
from korean_math_tdcs.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


def _torch_dtype(name: str | None) -> Any:
    import torch

    if name in (None, "auto"):
        supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if supports_bf16 else torch.float32
    return getattr(torch, name)


def load_trainable_model(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=config["model"].get("revision"),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=config["model"].get("revision"),
        trust_remote_code=config["model"].get("trust_remote_code", False),
        torch_dtype=_torch_dtype(config["model"].get("dtype", "auto")),
        attn_implementation=config["model"].get("attn_implementation"),
    ).to(device)
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora.get("alpha", int(lora["rank"]) * 2)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=lora.get("target_modules", "all-linear"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer, device


def _batch_chunks(indices: list[int], micro_batch_size: int) -> list[list[int]]:
    return [
        indices[start : start + micro_batch_size]
        for start in range(0, len(indices), micro_batch_size)
    ]


def _optimizer_step(
    model: Any,
    optimizer: Any,
    dataset: Any,
    collator: Any,
    indices: list[int],
    micro_batch_size: int,
    device: Any,
    max_grad_norm: float,
) -> tuple[float, int]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    training_tokens = 0
    total_examples = len(indices)
    for chunk in _batch_chunks(indices, micro_batch_size):
        batch = collator([dataset[index] for index in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        weight = len(chunk) / total_examples
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            output = model(**batch, use_cache=False)
            loss = output.loss * weight
        loss.backward()
        total_loss += float(output.loss.detach()) * weight
        training_tokens += int((batch["labels"] != -100).sum().item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return total_loss, training_tokens


def train(config: dict[str, Any], method: str) -> dict[str, Any]:
    import torch

    if method not in {"random", "tdcs"}:
        raise ValueError("method must be random or tdcs")
    seed = int(config.get("seed", 42))
    seed_everything(seed, bool(config.get("deterministic", False)))
    run_dir = ensure_dir(config["output"]["run_dir"])
    dataset_dict = load_reasoning_sft(config, with_difficulty=True)
    train_dataset = dataset_dict["train"]
    if config.get("difficulty", {}).get("validate_primary_counts", True):
        validate_primary_counts(train_dataset)

    model, tokenizer, device = load_trainable_model(config)
    max_length = int(config["training"]["max_seq_length"])
    collator = AssistantOnlyCollator(tokenizer, max_length=max_length)
    rendered = verify_serialization(tokenizer, train_dataset, max_length, count=3)
    for index, text in enumerate(rendered, start=1):
        LOGGER.info("Serialized training sample %d:\n%s", index, text)

    epochs = int(config["training"]["epochs"])
    effective_batch_size = int(config["training"]["effective_batch_size"])
    micro_batch_size = int(config["training"].get("micro_batch_size") or effective_batch_size)
    if micro_batch_size > effective_batch_size:
        raise ValueError("micro_batch_size cannot exceed effective_batch_size")
    total_examples, total_steps = resolved_training_budget(
        len(train_dataset), epochs, effective_batch_size
    )
    expected_examples = config["training"].get("expected_example_draws")
    if expected_examples is not None and total_examples != int(expected_examples):
        raise ValueError(f"Expected {expected_examples} draws but resolved {total_examples}")
    expected_steps = config["training"].get("expected_optimizer_steps")
    if expected_steps is not None and total_steps != int(expected_steps):
        raise ValueError(f"Expected {expected_steps} optimizer steps but resolved {total_steps}")
    config["training"]["resolved_micro_batch_size"] = micro_batch_size
    config["training"]["resolved_example_draws"] = total_examples
    config["training"]["resolved_optimizer_steps"] = total_steps
    if method == "tdcs":
        config["curriculum"]["resolved_stage_boundaries"] = [
            {"level": level, "first_step": start, "last_step": end}
            for level, (start, end) in enumerate(stage_boundaries(total_steps), start=1)
        ]
    save_config(config, run_dir / "config.yaml")

    from torch.optim import AdamW
    from transformers import get_scheduler

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = get_scheduler(
        config["training"].get("lr_scheduler_type", "constant"),
        optimizer=optimizer,
        num_warmup_steps=int(config["training"].get("warmup_steps", 0)),
        num_training_steps=total_steps,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timer = RunTimer()
    total_training_tokens = 0
    total_draws = 0
    cumulative_counts = Counter()
    random_batches = None
    tdcs_state = None
    if method == "random":
        indices = random_epoch_indices(len(train_dataset), epochs, seed)
        random_batches = iter(optimizer_batches(indices, effective_batch_size))
    else:
        tdcs_state = TDCSState(
            sampler=LevelPoolSampler(train_dataset["tdcs_level"], seed),
            dataset=train_dataset,
            collator=collator,
            model=model,
            device=device,
            config=config,
            total_steps=total_steps,
            rng=__import__("numpy").random.default_rng(seed + 1),
        )

    model.train()
    progress = tqdm(range(1, total_steps + 1), desc=f"{method.upper()} SFT")
    for step in progress:
        remaining = total_examples - total_draws
        optimizer_batch_size = min(effective_batch_size, remaining)
        refresh_record = None
        if method == "tdcs":
            changed, probe_stats = tdcs_state.refresh_if_needed(step)
            if changed:
                start, end = stage_boundaries(total_steps)[tdcs_state.current_level - 1]
                refresh_record = {
                    "timestamp": utc_timestamp(),
                    "global_step": step - 1,
                    "next_optimizer_step": step,
                    "current_stage": tdcs_state.current_level,
                    "current_stage_progress": (step - start) / max(1, end - start + 1),
                    "stage_step_range": [start, end],
                    "transfer_matrix": tdcs_state.transfer_matrix.tolist(),
                    "sampling_probabilities": tdcs_state.probabilities.tolist(),
                    "probe_stats": probe_stats,
                }
            batch_indices, level_counts = tdcs_state.draw_optimizer_batch(optimizer_batch_size)
        else:
            batch_indices = next(random_batches)
            level_counts = Counter(
                int(train_dataset[index]["tdcs_level"]) for index in batch_indices
            )
            level_counts = {level: level_counts.get(level, 0) for level in range(1, 6)}
        cumulative_counts.update({level: count for level, count in level_counts.items()})
        if refresh_record is not None:
            refresh_record["sample_counts_next_batch"] = level_counts
            append_jsonl(refresh_record, run_dir / "transfer_history.jsonl")

        loss, tokens = _optimizer_step(
            model,
            optimizer,
            train_dataset,
            collator,
            batch_indices,
            micro_batch_size,
            device,
            float(config["training"].get("max_grad_norm", 1.0)),
        )
        scheduler.step()
        total_training_tokens += tokens
        total_draws += len(batch_indices)
        record = {
            "timestamp": utc_timestamp(),
            "global_step": step,
            "loss": loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "ordinary_example_draws": total_draws,
            "ordinary_training_tokens": total_training_tokens,
            "sample_counts": level_counts,
        }
        append_jsonl(record, run_dir / "training_log.jsonl")
        progress.set_postfix(loss=f"{loss:.4f}")

    adapter_dir = ensure_dir(run_dir / "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        "method": method,
        "model": config["model"]["name"],
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(),
        "ordinary_example_draws": total_draws,
        "optimizer_steps": total_steps,
        "ordinary_training_tokens": total_training_tokens,
        "sample_counts_per_difficulty": {str(k): cumulative_counts[k] for k in range(1, 6)},
        "wall_clock_seconds": timer.elapsed_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "adapter_path": str(adapter_dir),
    }
    if tdcs_state is not None:
        metrics.update(
            {
                "probe_examples": tdcs_state.probe_examples,
                "probe_tokens": tdcs_state.probe_tokens,
                "probe_backward_passes": tdcs_state.probe_backward_passes,
            }
        )
    write_json(metrics, run_dir / "metrics.json")
    return metrics

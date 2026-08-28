from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from korean_math_tdcs.evaluation.benchmarks import (
    answers_equal,
    gold_answer,
    load_benchmark,
    parse_answer,
)
from korean_math_tdcs.evaluation.generation import generate_batch
from korean_math_tdcs.utils.io import append_jsonl, git_commit, utc_timestamp, write_json
from korean_math_tdcs.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


def _load_existing_predictions(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed prediction line %d in %s", line_number, path)
                continue
            records[str(record["uid"])] = record
    return records


def load_inference_model(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    base_name = model_config["name"]
    adapter = model_config.get("adapter")
    tokenizer_name = adapter or base_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = model_config.get("dtype", "auto")
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        attn_implementation=model_config.get("attn_implementation"),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def evaluate_loaded(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    *,
    benchmark_names: list[str] | None = None,
    recirculation: dict[str, Any] | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    generation = config["evaluation"]["generation"]
    benchmarks = benchmark_names or config["evaluation"]["benchmarks"]
    summaries = {}
    total_tokens = 0
    total_seconds = 0.0
    peak_memory = 0
    existing_predictions = _load_existing_predictions(predictions_path)
    for name in benchmarks:
        examples = load_benchmark(name, config["evaluation"].get(name, {}))
        benchmark_uids = {example.uid for example in examples}
        resumed = {
            uid: record
            for uid, record in existing_predictions.items()
            if uid in benchmark_uids and record.get("benchmark") == name
        }
        correct = sum(int(record["correct"]) for record in resumed.values())
        subset_counts = defaultdict(lambda: [0, 0])
        for record in resumed.values():
            subset_counts[record["subset"]][0] += int(record["correct"])
            subset_counts[record["subset"]][1] += 1
            total_tokens += int(record.get("generated_tokens", 0))
            total_seconds += float(record.get("latency_seconds", 0.0))
            peak_memory = max(peak_memory, int(record.get("peak_memory_bytes", 0)))
        pending = [example for example in examples if example.uid not in resumed]
        batch_size = 1 if recirculation else int(config["evaluation"].get("batch_size", 64))
        progress = tqdm(
            total=len(examples),
            initial=len(resumed),
            desc=f"Evaluating {name}",
        )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = generate_batch(
                model,
                tokenizer,
                [example.prompt for example in batch],
                generation,
                recirculation,
            )
            for example, result in zip(batch, results, strict=True):
                predicted = parse_answer(result.text, example.parser)
                expected = gold_answer(example.answer, example.parser)
                is_correct = answers_equal(predicted, expected, example.parser)
                correct += int(is_correct)
                subset_counts[example.subset][0] += int(is_correct)
                subset_counts[example.subset][1] += 1
                total_tokens += result.generated_tokens
                total_seconds += result.elapsed_seconds
                peak_memory = max(peak_memory, result.peak_memory_bytes)
                if predictions_path is not None:
                    append_jsonl(
                        {
                            "benchmark": name,
                            "uid": example.uid,
                            "subset": example.subset,
                            "prediction": predicted,
                            "gold": expected,
                            "correct": is_correct,
                            "output": result.text,
                            "generated_tokens": result.generated_tokens,
                            "latency_seconds": result.elapsed_seconds,
                            "peak_memory_bytes": result.peak_memory_bytes,
                        },
                        predictions_path,
                    )
            progress.update(len(batch))
        progress.close()
        metric = (
            "symbolic_equivalence"
            if examples and all(example.parser == "olympiad_math" for example in examples)
            else "exact_match"
        )
        summaries[name] = {
            "metric": metric,
            "score": correct / len(examples) if examples else 0.0,
            "correct": correct,
            "total": len(examples),
            "subsets": {
                subset: {"score": values[0] / values[1], "correct": values[0], "total": values[1]}
                for subset, values in sorted(subset_counts.items())
            },
        }
    return {
        "benchmarks": summaries,
        "generation": generation,
        "generated_tokens": total_tokens,
        "latency_seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
        "peak_gpu_memory_bytes": peak_memory,
    }


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    model, tokenizer = load_inference_model(config)
    output = Path(config["output"]["results_path"])
    predictions = output.with_name(output.stem + "_predictions.jsonl")
    if predictions.exists() and not config["evaluation"].get("resume", True):
        predictions.unlink()
    recirculation = config.get("recirculation")
    if (
        recirculation
        and recirculation.get("mode") == "fixed"
        and recirculation.get("source_layer") is not None
    ):
        recirculation = {**recirculation, "enabled": True}
    else:
        recirculation = None
    results = evaluate_loaded(
        model,
        tokenizer,
        config,
        recirculation=recirculation,
        predictions_path=predictions,
    )
    results.update(
        {
            "model": config["model"],
            "timestamp": utc_timestamp(),
            "git_commit": git_commit(),
            "recirculation": recirculation,
        }
    )
    write_json(results, output)
    LOGGER.info("Saved evaluation to %s", output)
    return results

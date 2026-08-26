from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from korean_math_tdcs.data.formatting import format_eval_prompt
from korean_math_tdcs.recirculation.fixed import FixedRecirculation


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int
    elapsed_seconds: float
    peak_memory_bytes: int


def _render_eval_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _generated_length(token_ids: Any, eos_token_ids: set[int], pad_token_id: int) -> int:
    values = token_ids.tolist()
    for index, token_id in enumerate(values):
        if token_id in eos_token_ids:
            return index + 1
    while values and values[-1] == pad_token_id:
        values.pop()
    return len(values)


def _sample(logits: Any, config: dict[str, Any]) -> Any:
    import torch

    if not config.get("do_sample", True):
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = float(config.get("temperature", 0.6))
    top_p = float(config.get("top_p", 0.95))
    probabilities = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: dict[str, Any],
    recirculation: dict[str, Any] | None = None,
) -> GenerationResult:
    return generate_batch(model, tokenizer, [prompt], config, recirculation)[0]


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, Any],
    recirculation: dict[str, Any] | None = None,
) -> list[GenerationResult]:
    import torch

    if not prompts:
        return []
    if recirculation and recirculation.get("enabled", True):
        if len(prompts) != 1:
            raise ValueError("Fixed Recirculation currently requires evaluation batch_size=1")
        input_ids = torch.tensor(
            [format_eval_prompt(tokenizer, prompts[0])],
            dtype=torch.long,
            device=model.device,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids = _generate_recirculation(model, input_ids, config, recirculation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return [
            GenerationResult(
                text=tokenizer.decode(output_ids[0], skip_special_tokens=True),
                generated_tokens=output_ids.shape[1],
                elapsed_seconds=elapsed,
                peak_memory_bytes=(
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
            )
        ]

    rendered = [_render_eval_prompt(tokenizer, prompt) for prompt in prompts]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        tokenizer.padding_side = previous_padding_side
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs = {
        "max_new_tokens": int(config.get("max_new_tokens", 512)),
        "do_sample": bool(config.get("do_sample", True)),
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs.update(
            temperature=float(config.get("temperature", 0.6)),
            top_p=float(config.get("top_p", 0.95)),
        )
    generated = model.generate(**inputs, **kwargs)
    output_ids = generated[:, inputs["input_ids"].shape[1] :]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    eos = tokenizer.eos_token_id
    eos_token_ids = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
    per_item_elapsed = elapsed / len(prompts)
    return [
        GenerationResult(
            text=tokenizer.decode(token_ids, skip_special_tokens=True),
            generated_tokens=_generated_length(token_ids, eos_token_ids, pad_token_id),
            elapsed_seconds=per_item_elapsed,
            peak_memory_bytes=peak_memory,
        )
        for token_ids in output_ids
    ]


def _generate_recirculation(
    model: Any,
    input_ids: Any,
    generation: dict[str, Any],
    recirculation: dict[str, Any],
) -> Any:
    import torch

    cache = None
    logits = None
    attention_mask = torch.ones((1, 0), dtype=torch.long, device=input_ids.device)
    generated = []
    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(eos_token_id, list):
        eos_token_ids = set(eos_token_id)
    elif eos_token_id is None:
        eos_token_ids = set()
    else:
        eos_token_ids = {eos_token_id}
    with torch.inference_mode(), FixedRecirculation(
        model,
        source_layer=int(recirculation["source_layer"]),
        destination_layer=int(recirculation["destination_layer"]),
        alpha=float(recirculation["alpha"]),
        beta=recirculation.get("beta"),
    ):
        for position in range(input_ids.shape[1]):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=input_ids.device),
                ],
                dim=1,
            )
            output = model(
                input_ids=input_ids[:, position : position + 1],
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits[:, -1, :]
        for _ in range(int(generation.get("max_new_tokens", 512))):
            next_token = _sample(logits, generation)
            generated.append(next_token)
            if next_token.item() in eos_token_ids:
                break
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=input_ids.device),
                ],
                dim=1,
            )
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits[:, -1, :]
    if not generated:
        return torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
    return torch.cat(generated, dim=1)

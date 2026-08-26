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
    import torch

    input_ids = torch.tensor(
        [format_eval_prompt(tokenizer, prompt)], dtype=torch.long, device=model.device
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    if recirculation and recirculation.get("enabled", True):
        output_ids = _generate_recirculation(model, input_ids, config, recirculation)
    else:
        kwargs = {
            "max_new_tokens": int(config.get("max_new_tokens", 512)),
            "do_sample": bool(config.get("do_sample", True)),
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs.update(
                temperature=float(config.get("temperature", 0.6)),
                top_p=float(config.get("top_p", 0.95)),
            )
        generated = model.generate(input_ids=input_ids, **kwargs)
        output_ids = generated[:, input_ids.shape[1] :]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return GenerationResult(
        text=text,
        generated_tokens=output_ids.shape[1],
        elapsed_seconds=elapsed,
        peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    )


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def assistant_content(reasoning: str, response: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{response.strip()}"


def format_sft_text(tokenizer: Any, example: dict[str, Any]) -> str:
    messages = [
        {"role": "user", "content": str(example["instruction"]).strip()},
        {
            "role": "assistant",
            # EXAONE's template natively supports a separate reasoning field.
            # Supplying literal <think> tags makes that template reparse and
            # alter the reasoning text before rendering it.
            "reasoning_content": str(example["reasoning"]).strip(),
            "content": str(example["response"]).strip(),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        skip_think=False,
    )


def format_eval_prompt(tokenizer: Any, prompt: str) -> list[int]:
    # Transformers 4 returned a plain list here by default, while Transformers
    # 5 returns a BatchEncoding unless return_dict=False. Rendering first keeps
    # this stable across both APIs and prevents torch.tensor from seeing dict keys.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    max_length: int = 512

    def encode(self, example: dict[str, Any]) -> dict[str, list[int]]:
        full_text = format_sft_text(self.tokenizer, example)
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(example["instruction"]).strip()}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            raise ValueError("Serialized example has no assistant target after truncation")
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self.encode(example) for example in examples]
        max_len = max(len(item["input_ids"]) for item in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in encoded:
            padding = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def verify_serialization(
    tokenizer: Any, dataset: Any, max_length: int, count: int = 3
) -> list[str]:
    collator = AssistantOnlyCollator(tokenizer, max_length=max_length)
    rendered: list[str] = []
    for index in range(min(count, len(dataset))):
        text = format_sft_text(tokenizer, dataset[index])
        reasoning = str(dataset[index]["reasoning"]).strip()
        if "<think>" not in text or "</think>" not in text or reasoning not in text:
            raise ValueError(f"Gold reasoning is missing from serialized sample {index}")
        encoded = collator.encode(dataset[index])
        if len(encoded["input_ids"]) > max_length:
            raise ValueError(f"Serialized sample {index} exceeds max_length")
        rendered.append(text)
    return rendered

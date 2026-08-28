#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
from pathlib import Path

from korean_math_tdcs.data.formatting import AssistantOnlyCollator, format_sft_text
from korean_math_tdcs.data.loading import load_reasoning_sft
from korean_math_tdcs.utils.config import apply_overrides, config_argument_parser, load_config
from korean_math_tdcs.utils.io import write_json
from korean_math_tdcs.utils.logging import configure_logging


def main() -> None:
    parser = config_argument_parser("Audit training difficulty and EXAONE serialization")
    args = parser.parse_args()
    configure_logging()
    config = apply_overrides(load_config(args.config), args.set)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = load_reasoning_sft(config, with_difficulty=True)
    train = datasets["train"]
    max_length = int(config["training"]["max_seq_length"])
    collator = AssistantOnlyCollator(tokenizer, max_length=max_length)
    lengths = []
    truncated = []
    for index, row in enumerate(train):
        text = format_sft_text(tokenizer, row)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if len(ids) > max_length:
            truncated.append({"index": index, "tokens": len(ids)})
        collator.encode(row)
    counts = Counter(int(level) for level in train["tdcs_level"])
    artifact = {
        "dataset": config["data"]["dataset"],
        "config": config["data"].get("config"),
        "dataset_revision": config["data"].get("revision"),
        "tokenizer": config["model"]["name"],
        "tokenizer_revision": config["model"].get("revision"),
        "train_rows": len(train),
        "validation_rows": len(datasets["validation"]),
        "operator_count_distribution": (
            dict(sorted(Counter(train["operator_count"]).items()))
            if "operator_count" in train.column_names
            else None
        ),
        "tdcs_level_distribution": {f"D{level}": counts[level] for level in range(1, 6)},
        "serialized_token_length": {"min": min(lengths), "max": max(lengths)},
        "max_seq_length": max_length,
        "truncated_examples": truncated,
        "samples": [format_sft_text(tokenizer, train[index]) for index in range(3)],
    }
    output = Path(
        config.get("audit", {}).get(
            "output", "results/dataset_audit/training_dataset_audit.json"
        )
    )
    write_json(artifact, output)
    print(f"Wrote {output}")
    for index, sample in enumerate(artifact["samples"], start=1):
        print(f"\n===== SERIALIZED SAMPLE {index} =====\n{sample}")
    if truncated and config.get("audit", {}).get("fail_on_truncation", False):
        raise SystemExit(
            f"Audit failed: {len(truncated)} examples exceed max_seq_length={max_length}"
        )


if __name__ == "__main__":
    main()

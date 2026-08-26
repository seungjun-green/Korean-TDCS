from __future__ import annotations

from typing import Any

from .difficulty import add_difficulty_columns

REQUIRED_SFT_FIELDS = {
    "instruction",
    "reasoning",
    "response",
    "gold_trace",
    "gold_trace_exact",
    "gold_answer",
}


def load_reasoning_sft(config: dict[str, Any], *, with_difficulty: bool = True) -> dict[str, Any]:
    from datasets import load_dataset

    data = config["data"]
    try:
        dataset = load_dataset(
            data["dataset"],
            data.get("config", "reasoning-sft"),
            revision=data.get("revision"),
        )
    except ValueError as error:
        # The Hub metadata currently advertises a `Json` feature that some
        # released `datasets` clients cannot deserialize. Loading the public
        # JSONL files directly preserves their inferred nested structure.
        if "Feature type 'Json' not found" not in str(error):
            raise
        from huggingface_hub import hf_hub_url

        if data["dataset"] != "keunhyeung/dmath-ko-reasoning-dpo":
            raise
        revision = data.get("revision") or "main"
        files = data.get(
            "files",
            {
                "train": "data/reasoning/train_sft_eligible.jsonl",
                "validation": "data/reasoning/validation_sft_eligible.jsonl",
            },
        )
        urls = {
            split: hf_hub_url(
                data["dataset"], filename, repo_type="dataset", revision=revision
            )
            for split, filename in files.items()
        }
        dataset = load_dataset("json", data_files=urls)
    result = {
        "train": dataset[data.get("train_split", "train")],
        "validation": dataset[data.get("validation_split", "validation")],
    }
    missing = REQUIRED_SFT_FIELDS - set(result["train"].column_names)
    if missing:
        raise ValueError(f"Training dataset is missing required fields: {sorted(missing)}")

    if with_difficulty:
        boundaries = config["difficulty"]["boundaries"]
        for split, values in result.items():
            if "operator_count" in values.column_names or "tdcs_level" in values.column_names:
                values = values.remove_columns(
                    [
                        name
                        for name in ("operator_count", "tdcs_level")
                        if name in values.column_names
                    ]
                )
            result[split] = values.map(
                add_difficulty_columns,
                fn_kwargs={"boundaries": boundaries},
                desc=f"Deriving difficulty for {split}",
            )
    return result

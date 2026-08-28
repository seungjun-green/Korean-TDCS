from __future__ import annotations

import re
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


def _last_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def _solution_final_answer(solution: str) -> str:
    boxed = _last_boxed_content(solution)
    if boxed:
        return boxed
    markers = re.findall(r"(?:최종\s*답|정답|답)\s*[:：은는]?\s*([^\n]+)", solution)
    if markers:
        return markers[-1].strip().rstrip(".。")
    lines = [line.strip() for line in solution.splitlines() if line.strip()]
    if not lines:
        raise ValueError("solution_ko must not be empty")
    return lines[-1].strip("$ ").rstrip(".。")


def _normalize_olympiad_sft(example: dict[str, Any]) -> dict[str, Any]:
    instruction = str(example["problem_ko"] or "").strip()
    reasoning = str(example["solution_ko"] or "").strip()
    if not instruction or not reasoning:
        raise ValueError("problem_ko and solution_ko must both be non-empty")
    level = int(example["difficulty_level"])
    if level not in range(1, 6):
        raise ValueError(f"difficulty_level must be in 1..5, found {level}")
    answer = _solution_final_answer(reasoning)
    return {
        "instruction": instruction,
        "reasoning": reasoning,
        "response": f"\\boxed{{{answer}}}",
        "gold_trace": [reasoning],
        "gold_trace_exact": [reasoning],
        "gold_answer": answer,
        "difficulty_level": level,
        "tdcs_level": level,
    }


def load_reasoning_sft(config: dict[str, Any], *, with_difficulty: bool = True) -> dict[str, Any]:
    from datasets import load_dataset

    data = config["data"]
    dataset_config = data.get("config")
    load_args = [data["dataset"]]
    if dataset_config is not None:
        load_args.append(dataset_config)
    try:
        dataset = load_dataset(
            *load_args,
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
    result: dict[str, Any] = {
        "train": dataset[data.get("train_split", "train")],
        "validation": dataset[data.get("validation_split", "validation")],
    }
    data_format = data.get("format", "reasoning_sft")
    if data_format == "olympiad_tdcs":
        required = {"problem_ko", "solution_ko", "difficulty_level"}
        for split, values in result.items():
            missing = required - set(values.column_names)
            if missing:
                raise ValueError(
                    f"Olympiad TDCS {split} split is missing fields: {sorted(missing)}"
                )
            result[split] = values.map(
                _normalize_olympiad_sft,
                remove_columns=values.column_names,
                desc=f"Normalizing Olympiad TDCS {split}",
            )
    elif data_format != "reasoning_sft":
        raise ValueError(f"Unknown training data format: {data_format}")

    missing = REQUIRED_SFT_FIELDS - set(result["train"].column_names)
    if missing:
        raise ValueError(f"Training dataset is missing required fields: {sorted(missing)}")

    if with_difficulty:
        if data_format == "olympiad_tdcs":
            for split, values in result.items():
                observed = {int(level) for level in values["tdcs_level"]}
                if observed != set(range(1, 6)):
                    raise ValueError(
                        f"Olympiad TDCS {split} split must contain D1-D5; found {sorted(observed)}"
                    )
        else:
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

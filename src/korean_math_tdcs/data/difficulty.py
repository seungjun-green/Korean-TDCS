from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

EXPECTED_LEVEL_COUNTS = {1: 575, 2: 984, 3: 510, 4: 325, 5: 314}


def operator_count(example: Mapping[str, Any]) -> int:
    trace = example.get("gold_trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise ValueError("gold_trace must be a non-string sequence")
    if not trace:
        raise ValueError("gold_trace must be non-empty")
    return len(trace)


def tdcs_level(count: int, boundaries: Mapping[str, int] | None = None) -> int:
    bounds = boundaries or {"d1_max": 1, "d2_max": 2, "d3_max": 3, "d4_max": 4}
    ordered = [bounds[f"d{i}_max"] for i in range(1, 5)]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError(f"Difficulty boundaries must be strictly increasing: {ordered}")
    for level, maximum in enumerate(ordered, start=1):
        if count <= maximum:
            return level
    return 5


def add_difficulty_columns(
    example: Mapping[str, Any], boundaries: Mapping[str, int]
) -> dict[str, int]:
    count = operator_count(example)
    return {"operator_count": count, "tdcs_level": tdcs_level(count, boundaries)}


def level_counts(dataset: Any) -> dict[int, int]:
    counts = Counter(int(level) for level in dataset["tdcs_level"])
    return {level: counts.get(level, 0) for level in range(1, 6)}


def validate_primary_counts(dataset: Any) -> None:
    observed = level_counts(dataset)
    if observed != EXPECTED_LEVEL_COUNTS:
        raise ValueError(
            "Primary reasoning-sft difficulty counts changed: "
            f"expected {EXPECTED_LEVEL_COUNTS}, observed {observed}. "
            "Check dataset revision or label derivation before training."
        )

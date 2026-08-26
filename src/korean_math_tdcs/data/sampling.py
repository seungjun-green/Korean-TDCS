from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np


def random_epoch_indices(size: int, epochs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    draws: list[int] = []
    for _ in range(epochs):
        draws.extend(rng.permutation(size).tolist())
    return draws


class LevelPoolSampler:
    """Samples levels by probability, reshuffling examples within each level on exhaustion."""

    def __init__(self, levels: Sequence[int], seed: int):
        self.rng = np.random.default_rng(seed)
        self.pools = {
            level: np.flatnonzero(np.asarray(levels) == level).astype(int).tolist()
            for level in range(1, 6)
        }
        if any(not pool for pool in self.pools.values()):
            raise ValueError("All five TDCS difficulty levels must contain at least one example")
        self.queues: dict[int, list[int]] = {level: [] for level in range(1, 6)}

    def _draw_from_level(self, level: int) -> int:
        if not self.queues[level]:
            queue = self.pools[level].copy()
            self.rng.shuffle(queue)
            self.queues[level] = queue
        return self.queues[level].pop()

    def draw(self, probabilities: Sequence[float], count: int) -> tuple[list[int], list[int]]:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (5,) or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
            raise ValueError(f"Invalid TDCS probabilities: {probabilities}")
        sampled_levels = self.rng.choice(np.arange(1, 6), size=count, p=probs).tolist()
        return [self._draw_from_level(level) for level in sampled_levels], sampled_levels


def optimizer_batches(indices: Sequence[int], effective_batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(indices), effective_batch_size):
        yield list(indices[start : start + effective_batch_size])


def stage_boundaries(total_steps: int, num_levels: int = 5) -> list[tuple[int, int]]:
    """Return inclusive 1-based step boundaries using an equal-step partition."""
    if total_steps < num_levels:
        raise ValueError("total_steps must be at least num_levels")
    result: list[tuple[int, int]] = []
    start = 1
    # Nearest integer cumulative boundaries reproduce the resolved 339-step
    # schedule: 68 / 68 / 67 / 68 / 68.
    for level in range(1, num_levels + 1):
        end = round(total_steps * level / num_levels)
        result.append((start, end))
        start = end + 1
    return result


def stage_for_step(step: int, total_steps: int, num_levels: int = 5) -> int:
    for level, (start, end) in enumerate(stage_boundaries(total_steps, num_levels), start=1):
        if start <= step <= end:
            return level
    raise ValueError(f"Step {step} is outside 1..{total_steps}")


def resolved_training_budget(
    dataset_size: int, epochs: int, effective_batch_size: int
) -> tuple[int, int]:
    examples = dataset_size * epochs
    return examples, math.ceil(examples / effective_batch_size)

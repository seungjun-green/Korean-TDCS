from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from korean_math_tdcs.data.sampling import LevelPoolSampler, stage_for_step
from korean_math_tdcs.training.transfer import estimate_transfer_matrix


def tdcs_probabilities(
    transfer: np.ndarray,
    current_level: int,
    *,
    tau_e: float = 0.5,
    sigmoid_alpha: float = 10.0,
    sigmoid_beta: float = 0.5,
    replay_lambda: float = 1.0,
    tau_h: float = 0.85,
    replay_scope: str = "previous_only",
    harder_fraction: float = 0.10,
) -> np.ndarray:
    """Apply the paper's TDCS rules with one explicit harder-mass implementation choice.

    The paper says to move "a portion" of current mass to qualifying harder levels but
    does not define that portion. ``harder_fraction`` makes that missing scale explicit.
    """
    matrix = np.asarray(transfer, dtype=np.float64)
    if matrix.shape != (5, 5) or not np.isfinite(matrix).all():
        raise ValueError("transfer must be a finite 5x5 matrix")
    if current_level not in range(1, 6):
        raise ValueError("current_level must be in 1..5")
    if replay_scope not in {"previous_only", "all_except_current"}:
        raise ValueError("replay_scope must be previous_only or all_except_current")
    if not 0.0 <= harder_fraction <= 1.0:
        raise ValueError("harder_fraction must be in [0, 1]")

    k = current_level - 1
    candidates = range(k) if replay_scope == "previous_only" else (i for i in range(5) if i != k)
    replay = [i for i in candidates if matrix[i, k] < tau_e]
    probabilities = np.zeros(5, dtype=np.float64)
    if replay:
        mean_transfer = float(np.mean([matrix[i, k] for i in replay]))
        rho = 1.0 / (1.0 + math.exp(-sigmoid_alpha * (mean_transfer - sigmoid_beta)))
        weights = np.exp(np.clip(-replay_lambda * matrix[replay, k], -50.0, 50.0))
        probabilities[replay] = (1.0 - rho) * weights / weights.sum()
        probabilities[k] = rho
    else:
        probabilities[k] = 1.0

    harder = [j for j in range(k + 1, 5) if matrix[k, j] > tau_h]
    if harder and probabilities[k] > 0:
        transfer_mass = probabilities[k] * harder_fraction
        harder_weights = np.maximum(matrix[k, harder], 0.0)
        if harder_weights.sum() > 0:
            probabilities[k] -= transfer_mass
            probabilities[harder] += transfer_mass * harder_weights / harder_weights.sum()

    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0:
        raise FloatingPointError("TDCS produced zero total probability")
    probabilities /= total
    return probabilities


@dataclass
class TDCSState:
    sampler: LevelPoolSampler
    dataset: Any
    collator: Any
    model: Any
    device: Any
    config: dict[str, Any]
    total_steps: int
    rng: np.random.Generator
    transfer_matrix: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    current_level: int | None = None
    last_refresh_step: int | None = None
    probe_examples: int = 0
    probe_tokens: int = 0
    probe_backward_passes: int = 0

    def _probe_indices(self) -> dict[int, list[int]]:
        batch_size = int(self.config["transfer"]["probe_batch_size"])
        pools = {
            level: np.flatnonzero(np.asarray(self.dataset["tdcs_level"]) == level)
            for level in range(1, 6)
        }
        return {
            level: self.rng.choice(pool, size=batch_size, replace=len(pool) < batch_size).tolist()
            for level, pool in pools.items()
        }

    def refresh_if_needed(self, next_step: int) -> tuple[bool, dict[str, Any] | None]:
        level = stage_for_step(next_step, self.total_steps)
        interval = int(self.config["transfer"]["update_every_steps"])
        periodic = next_step == 1 or (next_step - 1) % interval == 0
        stage_changed = self.current_level is not None and level != self.current_level
        needs_matrix = self.transfer_matrix is None or periodic

        probe_stats = None
        if needs_matrix:
            self.transfer_matrix, probe_stats = estimate_transfer_matrix(
                self.model,
                self.dataset,
                self.collator,
                self._probe_indices(),
                self.device,
            )
            self.probe_examples += probe_stats["probe_examples"]
            self.probe_tokens += probe_stats["probe_tokens"]
            self.probe_backward_passes += probe_stats["probe_backward_passes"]
            self.last_refresh_step = next_step - 1

        if needs_matrix or stage_changed or self.probabilities is None:
            parameters = self.config["tdcs"]
            self.probabilities = tdcs_probabilities(
                self.transfer_matrix,
                level,
                tau_e=float(parameters["tau_e"]),
                sigmoid_alpha=float(parameters["sigmoid_alpha"]),
                sigmoid_beta=float(parameters["sigmoid_beta"]),
                replay_lambda=float(parameters["replay_lambda"]),
                tau_h=float(parameters["tau_h"]),
                replay_scope=str(parameters["replay_scope"]),
                harder_fraction=float(parameters.get("harder_fraction", 0.10)),
            )
            changed = True
        else:
            changed = False
        self.current_level = level
        return changed, probe_stats

    def draw_optimizer_batch(self, size: int) -> tuple[list[int], dict[int, int]]:
        if self.probabilities is None:
            raise RuntimeError("TDCS state must be refreshed before drawing")
        indices, levels = self.sampler.draw(self.probabilities, size)
        counts = Counter(levels)
        return indices, {level: counts.get(level, 0) for level in range(1, 6)}


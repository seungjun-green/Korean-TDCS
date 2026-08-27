from __future__ import annotations

import math
from dataclasses import dataclass, field


def validation_check_steps(
    total_steps: int,
    epochs: int,
    checks_per_epoch: int,
) -> list[int]:
    """Return evenly spaced validation steps within each training epoch."""
    if total_steps < 1 or epochs < 1 or checks_per_epoch < 1:
        raise ValueError("total_steps, epochs, and checks_per_epoch must be positive")
    if total_steps < epochs * checks_per_epoch:
        raise ValueError("Not enough optimizer steps for the requested validation frequency")

    steps: list[int] = []
    for epoch_index in range(epochs):
        first_step = (epoch_index * total_steps) // epochs + 1
        last_step = ((epoch_index + 1) * total_steps) // epochs
        epoch_steps = last_step - first_step + 1
        for check_index in range(1, checks_per_epoch + 1):
            offset = math.ceil(check_index * epoch_steps / checks_per_epoch) - 1
            steps.append(first_step + offset)
    return steps


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_loss: float = field(default=math.inf, init=False)
    best_step: int | None = field(default=None, init=False)
    bad_checks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.min_delta < 0:
            raise ValueError("early-stopping min_delta cannot be negative")

    def update(self, loss: float, step: int) -> tuple[bool, bool]:
        improved = math.isfinite(loss) and loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.best_step = step
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return improved, self.bad_checks >= self.patience

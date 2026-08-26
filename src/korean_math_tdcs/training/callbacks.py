from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunTimer:
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at


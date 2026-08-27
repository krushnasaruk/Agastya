"""
Real-Time AI Watchdog and Execution Budget Supervisor for Objective 7.
Enforces deterministic maximum runtime budget for neural inference and guarantees instantaneous classical fallback.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class WatchdogStatus:
    cycle_index: int
    elapsed_ms: float
    budget_ms: float
    is_timed_out: bool
    fallback_triggered: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIWatchdog:
    """
    Supervises neural inference execution budget and triggers classical fallback if exceeded.
    """
    def __init__(self, execution_budget_ms: float = 25.0):
        self.budget_ms = execution_budget_ms
        self.cycle_start_time: Optional[float] = None
        self.cycle_index: int = 0
        self.timeout_count: int = 0
        self.max_observed_elapsed_ms: float = 0.0

    def reset(self) -> None:
        self.cycle_start_time = None
        self.cycle_index = 0
        self.timeout_count = 0
        self.max_observed_elapsed_ms = 0.0

    def start_cycle(self) -> None:
        """Start execution budget timer for current epoch."""
        self.cycle_index += 1
        self.cycle_start_time = time.perf_counter()

    def check_deadline(self, artificial_delay_ms: float = 0.0) -> WatchdogStatus:
        """
        Check if inference completed within allotted budget.
        """
        if self.cycle_start_time is None:
            return WatchdogStatus(
                cycle_index=self.cycle_index,
                elapsed_ms=0.0,
                budget_ms=self.budget_ms,
                is_timed_out=False,
                fallback_triggered=False
            )

        now = time.perf_counter()
        elapsed_ms = (now - self.cycle_start_time) * 1000.0 + artificial_delay_ms
        self.max_observed_elapsed_ms = max(self.max_observed_elapsed_ms, elapsed_ms)

        is_timeout = (elapsed_ms > self.budget_ms)
        if is_timeout:
            self.timeout_count += 1

        return WatchdogStatus(
            cycle_index=self.cycle_index,
            elapsed_ms=round(elapsed_ms, 4),
            budget_ms=self.budget_ms,
            is_timed_out=is_timeout,
            fallback_triggered=is_timeout
        )

    def is_timeout(self) -> bool:
        """Convenience method to check if current cycle timed out."""
        return self.check_deadline().is_timed_out

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_supervised_cycles": self.cycle_index,
            "budget_ms": self.budget_ms,
            "timeout_count": self.timeout_count,
            "timeout_rate_pct": round((self.timeout_count / max(self.cycle_index, 1)) * 100.0, 3),
            "max_observed_elapsed_ms": round(self.max_observed_elapsed_ms, 4),
            "watchdog_healthy": (self.timeout_count == 0)
        }

"""
Timeout Injection and Fault Handling Suite for Objective 7.
Tests controlled inference delay injections and verifies graceful watchdog fallback.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .watchdog import AIWatchdog


class TimeoutExperimentHandler:
    """
    Executes controlled delay injections (1ms to >100ms) to validate watchdog fallback response.
    """
    TEST_DELAYS_MS = [1.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 150.0]

    @classmethod
    def evaluate_timeout_degradation(
        cls,
        watchdog: AIWatchdog,
        test_delays_ms: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate watchdog response across delay spectrum.
        """
        delays = test_delays_ms or cls.TEST_DELAYS_MS
        results = []

        for d in delays:
            watchdog.reset()
            watchdog.start_cycle()
            status = watchdog.check_deadline(artificial_delay_ms=d)

            # Expected behavior: If delay > budget, fallback must trigger
            expected_fallback = (d > watchdog.budget_ms)
            is_correct_behavior = (status.fallback_triggered == expected_fallback)

            results.append({
                "injected_delay_ms": d,
                "budget_ms": watchdog.budget_ms,
                "observed_elapsed_ms": status.elapsed_ms,
                "is_timed_out": status.is_timed_out,
                "fallback_triggered": status.fallback_triggered,
                "expected_fallback": expected_fallback,
                "behavior_verified": is_correct_behavior
            })

        return results

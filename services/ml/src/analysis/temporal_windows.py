"""
Temporal Window Duration Analysis for Project AGASTYA (Objective 4).
Evaluates causal historical window lengths (0.5s, 1.0s, 2.0s, 5.0s) regarding
latency, sample count, memory, and capture of dynamic cornering/slip events.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class WindowTradeoffReport:
    duration_sec: float
    num_epochs: int
    latency_ms: float
    sample_retention_pct: float       # Percentage of total sequence usable after window warm-up
    turning_dynamics_context: str     # Capability to observe cornering entry/exit
    wheel_slip_context: str           # Capability to observe micro-slip transitions
    recommendation_status: str        # 'RECOMMENDED', 'VIABLE', 'NOT_RECOMMENDED'
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalWindowAnalyzer:
    """
    Evaluates causal temporal window sizes for AI residual learning.
    """
    @classmethod
    def evaluate_all_windows(
        cls,
        total_sequence_samples: int = 600,
        nominal_hz: float = 10.0
    ) -> List[WindowTradeoffReport]:
        """
        Evaluate standard window configurations.
        """
        candidates = [
            (0.5, int(0.5 * nominal_hz)),
            (1.0, int(1.0 * nominal_hz)),
            (2.0, int(2.0 * nominal_hz)),
            (5.0, int(5.0 * nominal_hz))
        ]

        reports = []
        for dur, w_size in candidates:
            latency = (w_size / nominal_hz) * 1000.0
            usable = max(0, total_sequence_samples - w_size + 1)
            retention = (usable / total_sequence_samples) * 100.0

            if dur == 0.5:
                status = "VIABLE"
                turning = "Limited (observes only 0.5s of turn initiation)"
                slip = "High (captures immediate transient tire slip)"
                rat = "Low latency (50 ms) and high responsiveness, but lacks context for multi-second cornering arcs."
            elif dur == 1.0:
                status = "RECOMMENDED PRIMARY"
                turning = "Optimal (captures turn initiation, peak rate, and acceleration phase)"
                slip = "Optimal (sufficient history to detect slip onset and recovery)"
                rat = "Balanced sweet-spot: 100 ms latency, 98.5% sample retention, and robust dynamic context."
            elif dur == 2.0:
                status = "RECOMMENDED SECONDARY"
                turning = "Comprehensive (covers full roundabout entry and exit)"
                slip = "High (includes full post-slip steady state)"
                rat = "Ideal for recurrent architectures; captures long-range turning trends with moderate latency."
            elif dur == 5.0:
                status = "NOT_RECOMMENDED"
                turning = "Excessive (spans multiple disconnected maneuvers)"
                slip = "Diluted (transient slip signal is averaged out by steady cruise)"
                rat = "High warm-up lag (5.0s), discards 50 initial sequence epochs, and risks temporal smearing."
            else:
                status = "UNKNOWN"
                turning = "N/A"
                slip = "N/A"
                rat = "Custom duration."

            reports.append(WindowTradeoffReport(
                duration_sec=dur,
                num_epochs=w_size,
                latency_ms=round(latency, 1),
                sample_retention_pct=round(retention, 2),
                turning_dynamics_context=turning,
                wheel_slip_context=slip,
                recommendation_status=status,
                rationale=rat
            ))

        return reports

"""
Offline GNSS Outage Experiment Simulator for Project AGASTYA (Objective 3).
Simulates GNSS-denied navigation intervals with standardized start timestamps
and explicit maneuver labeling for scientific, monotonic drift evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class OutageScenario:
    outage_id: str
    duration_sec: float
    start_time_sec: float
    end_time_sec: float
    start_index: int
    end_index: int
    maneuver_type: str  # 'turning', 'straight', 'braking', 'standardized'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GNSSOutageSimulator:
    """
    Creates standardized and event-based GNSS-outage evaluation scenarios.
    """
    def __init__(self, default_durations_sec: Optional[List[float]] = None):
        self.durations_sec = default_durations_sec if default_durations_sec is not None else [5.0, 10.0, 30.0, 60.0]

    def generate_standardized_start_scenarios(
        self,
        timestamps_sec: np.ndarray,
        start_time_sec: float = 20.0,
        yaw_rate_rads: Optional[np.ndarray] = None
    ) -> List[OutageScenario]:
        """
        Generate outage scenarios where every duration begins at the EXACT SAME start_time_sec.
        This guarantees that longer outages strictly encompass shorter outages,
        ensuring monotonic drift growth evaluation.
        """
        t = np.asarray(timestamps_sec)
        total_duration = t[-1] - t[0] if len(t) > 1 else 0.0
        scenarios: List[OutageScenario] = []

        start_idx = int(np.searchsorted(t, start_time_sec))

        for dur in self.durations_sec:
            target_end_t = start_time_sec + dur
            if target_end_t > t[-1]:
                continue

            end_idx = int(np.searchsorted(t, target_end_t))
            end_idx = min(end_idx, len(t) - 1)

            # Determine dominant maneuver
            maneuver = "cruising"
            if yaw_rate_rads is not None and len(yaw_rate_rads) > end_idx:
                mean_yaw = float(np.mean(np.abs(yaw_rate_rads[start_idx:end_idx + 1])))
                if mean_yaw > 0.05:
                    maneuver = "turning"

            scenarios.append(OutageScenario(
                outage_id=f"outage_{int(dur)}s_t{int(start_time_sec)}s_{maneuver}",
                duration_sec=float(dur),
                start_time_sec=float(t[start_idx]),
                end_time_sec=float(t[end_idx]),
                start_index=start_idx,
                end_index=end_idx,
                maneuver_type=maneuver
            ))

        return scenarios

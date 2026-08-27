"""
Standardized GNSS Outage Simulation and Evaluation Suite for Objective 6.
Evaluates drift metrics for outages of 5s, 10s, 15s, 20s, 30s, and 45s starting at t = 20.0s.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from navigation_engine.state import DeadReckoningTrajectory
from navigation_engine.evaluation import DeadReckoningEvaluator


class StandardizedOutageSimulator:
    """
    Executes standardized GNSS denial benchmarking across multiple durations.
    """
    DEFAULT_DURATIONS = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    ENTRY_TIME_SEC = 20.0

    @classmethod
    def evaluate_multi_duration_outages(
        cls,
        classical_traj: DeadReckoningTrajectory,
        obj5_traj: DeadReckoningTrajectory,
        obj6_traj: DeadReckoningTrajectory,
        ref_east_m: np.ndarray,
        ref_north_m: np.ndarray,
        entry_time_sec: float = 20.0,
        durations: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multi-duration outage comparison on held-out trajectory.
        """
        durs = durations or cls.DEFAULT_DURATIONS
        timestamps = classical_traj.timestamps_sec
        total_time = float(timestamps[-1] - timestamps[0])

        outage_records = []

        for d in durs:
            t_start = entry_time_sec
            t_end = t_start + d

            if t_end > timestamps[-1]:
                # Trajectory cannot support this outage length
                continue

            idx_start = int(np.searchsorted(timestamps, t_start))
            idx_end = int(np.searchsorted(timestamps, t_end))

            # Segment distance
            d_e = ref_east_m[idx_start:idx_end+1]
            d_n = ref_north_m[idx_start:idx_end+1]
            dist = float(np.sum(np.sqrt(np.diff(d_e)**2 + np.diff(d_n)**2)))

            # 1. Classical Baseline
            c_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(classical_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            c_acc = DeadReckoningEvaluator.compute_outage_accumulated_drift(classical_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            c_max = DeadReckoningEvaluator.compute_outage_max_drift(classical_traj, ref_east_m, ref_north_m, idx_start, idx_end)

            # 2. Objective 5 Velocity-Only
            o5_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(obj5_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            o5_acc = DeadReckoningEvaluator.compute_outage_accumulated_drift(obj5_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            o5_max = DeadReckoningEvaluator.compute_outage_max_drift(obj5_traj, ref_east_m, ref_north_m, idx_start, idx_end)

            # 3. Objective 6 Selective Velocity
            o6_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(obj6_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            o6_acc = DeadReckoningEvaluator.compute_outage_accumulated_drift(obj6_traj, ref_east_m, ref_north_m, idx_start, idx_end)
            o6_max = DeadReckoningEvaluator.compute_outage_max_drift(obj6_traj, ref_east_m, ref_north_m, idx_start, idx_end)

            imp_vs_classical = ((c_ate - o6_ate) / max(c_ate, 1e-6)) * 100.0
            imp_vs_obj5 = ((o5_ate - o6_ate) / max(o5_ate, 1e-6)) * 100.0

            outage_records.append({
                "duration_sec": d,
                "start_time_sec": t_start,
                "end_time_sec": t_end,
                "distance_m": round(dist, 2),
                "classical": {
                    "ate_rmse_m": round(c_ate, 4),
                    "accumulated_drift_m": round(c_acc, 4),
                    "max_drift_m": round(c_max, 4),
                    "drift_rate_pct": round((c_acc / max(dist, 1.0)) * 100.0, 3)
                },
                "objective5_velocity": {
                    "ate_rmse_m": round(o5_ate, 4),
                    "accumulated_drift_m": round(o5_acc, 4),
                    "max_drift_m": round(o5_max, 4),
                    "drift_rate_pct": round((o5_acc / max(dist, 1.0)) * 100.0, 3)
                },
                "objective6_selective": {
                    "ate_rmse_m": round(o6_ate, 4),
                    "accumulated_drift_m": round(o6_acc, 4),
                    "max_drift_m": round(o6_max, 4),
                    "drift_rate_pct": round((o6_acc / max(dist, 1.0)) * 100.0, 3)
                },
                "improvement_vs_classical_pct": round(imp_vs_classical, 2),
                "improvement_vs_obj5_pct": round(imp_vs_obj5, 2)
            })

        return outage_records

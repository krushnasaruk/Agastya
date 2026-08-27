"""
Standardized GNSS Outage Evaluation for AI-Corrected Dead Reckoning (Objective 5).
Evaluates 5s, 10s, and 30s outages starting at t = 20.0s comparing Classical vs AI.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from navigation_engine.evaluation import DeadReckoningEvaluator, OutageEvaluationMetrics
from navigation_engine.outage import GNSSOutageSimulator, OutageScenario
from navigation_engine.state import DeadReckoningTrajectory


class OutageComparator:
    """
    Compares Classical vs AI dead-reckoning trajectories under standardized GNSS outages.
    """
    @classmethod
    def evaluate_outages(
        cls,
        classical_traj: DeadReckoningTrajectory,
        ai_traj: DeadReckoningTrajectory,
        ref_east_m: np.ndarray,
        ref_north_m: np.ndarray,
        start_time_sec: float = 20.0,
        durations: List[float] = [5.0, 10.0, 30.0]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and return comparative outage metrics.
        """
        sim = GNSSOutageSimulator(default_durations_sec=durations)
        scenarios = sim.generate_standardized_start_scenarios(
            classical_traj.timestamps_sec,
            start_time_sec=start_time_sec,
            yaw_rate_rads=classical_traj.yaw_rate_rads
        )

        comparison_results = []

        for sc in scenarios:
            # 1. Classical metrics
            c_acc_drift = DeadReckoningEvaluator.compute_outage_accumulated_drift(
                classical_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )
            c_max_drift = DeadReckoningEvaluator.compute_outage_max_drift(
                classical_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )
            c_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(
                classical_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )

            # 2. AI metrics
            ai_acc_drift = DeadReckoningEvaluator.compute_outage_accumulated_drift(
                ai_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )
            ai_max_drift = DeadReckoningEvaluator.compute_outage_max_drift(
                ai_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )
            ai_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(
                ai_traj, ref_east_m, ref_north_m, sc.start_index, sc.end_index
            )

            # Distance
            d_e = ref_east_m
            d_n = ref_north_m
            dist = float(np.sum(np.sqrt(np.diff(d_e[sc.start_index:sc.end_index+1])**2 + np.diff(d_n[sc.start_index:sc.end_index+1])**2)))

            c_drift_rate = (c_acc_drift / max(dist, 1.0)) * 100.0
            ai_drift_rate = (ai_acc_drift / max(dist, 1.0)) * 100.0
            ate_imp = ((c_ate - ai_ate) / max(c_ate, 1e-6)) * 100.0

            record = {
                "outage_id": sc.outage_id,
                "duration_sec": sc.duration_sec,
                "start_time_sec": sc.start_time_sec,
                "end_time_sec": sc.end_time_sec,
                "distance_traveled_m": round(dist, 2),
                "maneuver_type": sc.maneuver_type,
                "classical": {
                    "accumulated_drift_m": round(c_acc_drift, 4),
                    "max_drift_m": round(c_max_drift, 4),
                    "outage_ate_rmse_m": round(c_ate, 4),
                    "drift_rate_pct": round(c_drift_rate, 3)
                },
                "ai_corrected": {
                    "accumulated_drift_m": round(ai_acc_drift, 4),
                    "max_drift_m": round(ai_max_drift, 4),
                    "outage_ate_rmse_m": round(ai_ate, 4),
                    "drift_rate_pct": round(ai_drift_rate, 3)
                },
                "ate_improvement_pct": round(ate_imp, 2)
            }
            comparison_results.append(record)

        return comparison_results

"""
Deterministic Error Evaluation & Drift Benchmarking Engine for Project AGASTYA (Objective 3).
Provides distinct, mathematically explicit metric APIs:
  - compute_global_ate
  - compute_outage_accumulated_drift
  - compute_outage_max_drift
  - compute_outage_ate_rmse
  - compute_drift_rate
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .state import wrap_to_pi, DeadReckoningTrajectory


@dataclass
class OutageEvaluationMetrics:
    outage_id: str
    duration_sec: float
    start_time_sec: float
    end_time_sec: float
    accumulated_drift_m: float  # Final position error at outage end relative to outage start
    max_drift_m: float          # Peak position error accumulated during outage
    outage_ate_rmse_m: float    # Root-mean-square error during outage window
    drift_rate_pct: float       # (accumulated_drift / distance_during_outage) * 100
    distance_traveled_m: float
    maneuver_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NavigationMetrics:
    ate_rmse_m: float
    final_position_error_m: float
    max_position_error_m: float
    median_position_error_m: float
    p90_position_error_m: float
    p95_position_error_m: float
    drift_rate_pct: float
    heading_rmse_deg: float
    final_heading_error_deg: float
    max_heading_error_deg: float
    velocity_rmse_ms: float
    total_trajectory_distance_m: float
    evaluation_duration_sec: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeadReckoningEvaluator:
    """
    Evaluates estimated dead-reckoning trajectories against offline reference ground truth.
    All metric functions are mathematically explicit and distinguish global vs local outage errors.
    """
    @classmethod
    def compute_global_ate(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray
    ) -> float:
        """
        Compute Global Absolute Trajectory Error (ATE RMSE) over the full trajectory.
        Formula: sqrt( 1/N * sum( (p_E_est - p_E_ref)^2 + (p_N_est - p_N_ref)^2 ) )
        """
        e_est = estimated_traj.p_east_m
        n_est = estimated_traj.p_north_m
        e_ref = np.asarray(reference_p_east_m)
        n_ref = np.asarray(reference_p_north_m)
        pos_errors = np.sqrt((e_est - e_ref) ** 2 + (n_est - n_ref) ** 2)
        return float(np.sqrt(np.mean(pos_errors ** 2)))

    @classmethod
    def compute_outage_accumulated_drift(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """
        Compute the net position error accumulated strictly DURING the outage window,
        relative to the position at outage onset (start_idx).
        
        Formula:
          d_E_est = p_E_est[end] - p_E_est[start]
          d_N_est = p_N_est[end] - p_N_est[start]
          d_E_ref = p_E_ref[end] - p_E_ref[start]
          d_N_ref = p_N_ref[end] - p_N_ref[start]
          drift = sqrt( (d_E_est - d_E_ref)^2 + (d_N_est - d_N_ref)^2 )
        """
        d_e_est = estimated_traj.p_east_m[end_idx] - estimated_traj.p_east_m[start_idx]
        d_n_est = estimated_traj.p_north_m[end_idx] - estimated_traj.p_north_m[start_idx]
        d_e_ref = reference_p_east_m[end_idx] - reference_p_east_m[start_idx]
        d_n_ref = reference_p_north_m[end_idx] - reference_p_north_m[start_idx]
        return float(np.sqrt((d_e_est - d_e_ref) ** 2 + (d_n_est - d_n_ref) ** 2))

    @classmethod
    def compute_outage_max_drift(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """
        Compute maximum position drift accumulated at any intermediate step inside the outage window,
        relative to the start position.
        """
        e_est_rel = estimated_traj.p_east_m[start_idx:end_idx + 1] - estimated_traj.p_east_m[start_idx]
        n_est_rel = estimated_traj.p_north_m[start_idx:end_idx + 1] - estimated_traj.p_north_m[start_idx]
        e_ref_rel = reference_p_east_m[start_idx:end_idx + 1] - reference_p_east_m[start_idx]
        n_ref_rel = reference_p_north_m[start_idx:end_idx + 1] - reference_p_north_m[start_idx]
        errors = np.sqrt((e_est_rel - e_ref_rel) ** 2 + (n_est_rel - n_ref_rel) ** 2)
        return float(np.max(errors))

    @classmethod
    def compute_outage_ate_rmse(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """
        Compute root-mean-square error over the outage window relative to outage start pose.
        """
        e_est_rel = estimated_traj.p_east_m[start_idx:end_idx + 1] - estimated_traj.p_east_m[start_idx]
        n_est_rel = estimated_traj.p_north_m[start_idx:end_idx + 1] - estimated_traj.p_north_m[start_idx]
        e_ref_rel = reference_p_east_m[start_idx:end_idx + 1] - reference_p_east_m[start_idx]
        n_ref_rel = reference_p_north_m[start_idx:end_idx + 1] - reference_p_north_m[start_idx]
        errors = np.sqrt((e_est_rel - e_ref_rel) ** 2 + (n_est_rel - n_ref_rel) ** 2)
        return float(np.sqrt(np.mean(errors ** 2)))

    @classmethod
    def evaluate(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray,
        reference_heading_rad: Optional[np.ndarray] = None,
        reference_speed_ms: Optional[np.ndarray] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        align_initial_pose: bool = False
    ) -> Tuple[NavigationMetrics, np.ndarray, np.ndarray]:
        """
        Compute comprehensive error metrics between estimate and reference over [start_idx, end_idx].
        """
        e_est = estimated_traj.p_east_m[start_idx:end_idx]
        n_est = estimated_traj.p_north_m[start_idx:end_idx]
        h_est = estimated_traj.heading_rad[start_idx:end_idx]
        v_est = estimated_traj.forward_speed_ms[start_idx:end_idx]
        t = estimated_traj.timestamps_sec[start_idx:end_idx]

        e_ref = np.asarray(reference_p_east_m)[start_idx:end_idx]
        n_ref = np.asarray(reference_p_north_m)[start_idx:end_idx]

        num_samples = len(e_est)
        if num_samples == 0:
            raise ValueError("Evaluation slice contains 0 samples.")

        if align_initial_pose:
            d_e_init = e_est[0] - e_ref[0]
            d_n_init = n_est[0] - n_ref[0]
            e_calc = e_est - d_e_init
            n_calc = n_est - d_n_init
        else:
            e_calc = e_est
            n_calc = n_est

        # 1. 2D Position Errors
        pos_errors = np.sqrt((e_calc - e_ref) ** 2 + (n_calc - n_ref) ** 2)
        ate_rmse = float(np.sqrt(np.mean(pos_errors ** 2)))
        final_err = float(pos_errors[-1])
        max_err = float(np.max(pos_errors))
        med_err = float(np.median(pos_errors))
        p90_err = float(np.percentile(pos_errors, 90))
        p95_err = float(np.percentile(pos_errors, 95))

        # Total distance
        if len(e_ref) >= 2:
            step_dists = np.sqrt(np.diff(e_ref) ** 2 + np.diff(n_ref) ** 2)
            total_dist = float(np.sum(step_dists))
        else:
            total_dist = 0.0

        drift_pct = float((final_err / max(total_dist, 1.0)) * 100.0)

        # 2. Heading Errors
        if reference_heading_rad is not None:
            h_ref = np.asarray(reference_heading_rad)[start_idx:end_idx]
            head_diffs = np.array([wrap_to_pi(h_est[i] - h_ref[i]) for i in range(num_samples)])
            head_errors_deg = np.degrees(np.abs(head_diffs))
            head_rmse = float(np.sqrt(np.mean(head_errors_deg ** 2)))
            final_head_err = float(head_errors_deg[-1])
            max_head_err = float(np.max(head_errors_deg))
        else:
            head_errors_deg = np.zeros(num_samples)
            head_rmse = 0.0
            final_head_err = 0.0
            max_head_err = 0.0

        # 3. Velocity Errors (Unrounded Calculation)
        if reference_speed_ms is not None:
            v_ref = np.asarray(reference_speed_ms)[start_idx:end_idx]
            v_diffs = v_est - v_ref
            vel_rmse = float(np.sqrt(np.mean(v_diffs ** 2)))
        else:
            vel_rmse = 0.0

        duration = float(t[-1] - t[0]) if num_samples > 1 else 0.0

        metrics = NavigationMetrics(
            ate_rmse_m=round(ate_rmse, 4),
            final_position_error_m=round(final_err, 4),
            max_position_error_m=round(max_err, 4),
            median_position_error_m=round(med_err, 4),
            p90_position_error_m=round(p90_err, 4),
            p95_position_error_m=round(p95_err, 4),
            drift_rate_pct=round(drift_pct, 3),
            heading_rmse_deg=round(head_rmse, 3),
            final_heading_error_deg=round(final_head_err, 3),
            max_heading_error_deg=round(max_head_err, 3),
            velocity_rmse_ms=round(vel_rmse, 5),
            total_trajectory_distance_m=round(total_dist, 2),
            evaluation_duration_sec=round(duration, 2),
            num_samples=num_samples
        )

        return metrics, pos_errors, head_errors_deg

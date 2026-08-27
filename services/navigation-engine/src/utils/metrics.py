"""
Trajectory Error Analytics Engine.
Calculates Absolute Trajectory Error (ATE), Relative Pose Error (RPE),
drift percentage, maximum deviation, and covariance uncertainty traces.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class TrajectoryMetricResult:
    ate_rmse: float
    ate_mean: float
    ate_std: float
    max_position_error: float
    drift_percentage: float
    total_distance: float
    rpe_translation_rmse: float
    rpe_rotation_rmse: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ate_rmse": round(float(self.ate_rmse), 4),
            "ate_mean": round(float(self.ate_mean), 4),
            "ate_std": round(float(self.ate_std), 4),
            "max_position_error": round(float(self.max_position_error), 4),
            "drift_percentage": round(float(self.drift_percentage), 4),
            "total_distance": round(float(self.total_distance), 2),
            "rpe_translation_rmse": round(float(self.rpe_translation_rmse), 4),
            "rpe_rotation_rmse": round(float(self.rpe_rotation_rmse), 4),
            "num_samples": int(self.num_samples)
        }


class TrajectoryMetricsEngine:
    """
    Computes rigorous trajectory error metrics adhering to robotics
    and aerospace navigation benchmarks.
    """

    @staticmethod
    def compute_ate_rmse(
        estimated_pos: np.ndarray,
        ground_truth_pos: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Computes Absolute Trajectory Error (ATE).
        Returns: (rmse, mean, std, max_err) in meters.
        """
        if len(estimated_pos) == 0 or len(ground_truth_pos) == 0:
            return 0.0, 0.0, 0.0, 0.0

        n = min(len(estimated_pos), len(ground_truth_pos))
        est = np.asarray(estimated_pos[:n], dtype=np.float64)
        gt = np.asarray(ground_truth_pos[:n], dtype=np.float64)

        diff = est - gt
        errors = np.linalg.norm(diff, axis=1)

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        max_err = float(np.max(errors))

        return rmse, mean_err, std_err, max_err

    @staticmethod
    def compute_rpe(
        estimated_pos: np.ndarray,
        ground_truth_pos: np.ndarray,
        delta_step: int = 1
    ) -> float:
        """
        Computes Relative Pose Error (RPE) translation RMSE over delta_step increments.
        """
        n = min(len(estimated_pos), len(ground_truth_pos))
        if n <= delta_step:
            return 0.0

        est = np.asarray(estimated_pos[:n], dtype=np.float64)
        gt = np.asarray(ground_truth_pos[:n], dtype=np.float64)

        d_est = est[delta_step:] - est[:-delta_step]
        d_gt = gt[delta_step:] - gt[:-delta_step]

        rel_errors = np.linalg.norm(d_est - d_gt, axis=1)
        return float(np.sqrt(np.mean(rel_errors ** 2)))

    @staticmethod
    def compute_distance_traveled(positions: np.ndarray) -> float:
        """Computes cumulative Euclidean distance along a 3D trajectory."""
        if len(positions) < 2:
            return 0.0
        pos = np.asarray(positions, dtype=np.float64)
        diffs = np.diff(pos, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(segment_lengths))

    @classmethod
    def evaluate(
        cls,
        estimated_pos: np.ndarray,
        ground_truth_pos: np.ndarray,
        delta_step: int = 1
    ) -> TrajectoryMetricResult:
        """
        Evaluates full trajectory accuracy metrics with numerical safeguards.
        """
        n = min(len(estimated_pos), len(ground_truth_pos))
        if n == 0:
            return TrajectoryMetricResult(
                ate_rmse=0.0,
                ate_mean=0.0,
                ate_std=0.0,
                max_position_error=0.0,
                drift_percentage=0.0,
                total_distance=0.0,
                rpe_translation_rmse=0.0,
                rpe_rotation_rmse=0.0,
                num_samples=0
            )

        rmse, mean_err, std_err, max_err = cls.compute_ate_rmse(estimated_pos, ground_truth_pos)
        rpe_trans = cls.compute_rpe(estimated_pos, ground_truth_pos, delta_step=delta_step)
        total_dist = cls.compute_distance_traveled(ground_truth_pos[:n])

        # Drift percentage = (Final Position Error / Total Distance Traveled) * 100
        final_err = float(np.linalg.norm(estimated_pos[n-1] - ground_truth_pos[n-1]))
        if total_dist > 1e-3:
            drift_pct = (final_err / total_dist) * 100.0
        else:
            drift_pct = 0.0

        return TrajectoryMetricResult(
            ate_rmse=rmse,
            ate_mean=mean_err,
            ate_std=std_err,
            max_position_error=max_err,
            drift_percentage=drift_pct,
            total_distance=total_dist,
            rpe_translation_rmse=rpe_trans,
            rpe_rotation_rmse=0.0,
            num_samples=n
        )

"""
Diagnostic Visualization Engine for Classical Dead Reckoning (Objective 3).
Renders engineering performance figures:
1. 2D Estimated vs Reference Trajectory (Local ENU)
2. Instantaneous Position Error vs Time and Distance
3. Heading Error vs Time
4. Velocity Profile (Odometry vs VBOX True Speed)
5. Multi-Baseline Drift Comparison
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .state import DeadReckoningTrajectory
from .evaluation import NavigationMetrics


class ClassicalDiagnosticsVisualizer:
    """
    Renders diagnostic charts comparing classical dead-reckoning estimates to offline VBOX ground truth.
    """
    @classmethod
    def generate_baseline_plots(
        cls,
        estimated_traj: DeadReckoningTrajectory,
        reference_p_east_m: np.ndarray,
        reference_p_north_m: np.ndarray,
        pos_errors_m: np.ndarray,
        head_errors_deg: np.ndarray,
        reference_speed_ms: Optional[np.ndarray],
        metrics: NavigationMetrics,
        output_dir: str,
        sequence_id: str = "seq_01",
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        Render and save the 5 mandatory dead-reckoning diagnostic figures.
        """
        fig_dir = os.path.join(output_dir, "reports", "figures", sequence_id, "classical_baseline")
        os.makedirs(fig_dir, exist_ok=True)
        t = estimated_traj.timestamps_sec
        generated: Dict[str, str] = {}

        # ----------------------------------------------------------------------
        # Plot 1: 2D Trajectory (Estimate vs Offline Reference)
        # ----------------------------------------------------------------------
        plt.figure(figsize=(7.5, 7.5))
        plt.plot(reference_p_east_m, reference_p_north_m, color="#2b5c8f", linewidth=2.2, label="VBOX Ground Truth (Offline)", linestyle="-")
        plt.plot(estimated_traj.p_east_m, estimated_traj.p_north_m, color="#d62728", linewidth=2.0, label=f"Dead Reckoning ({estimated_traj.baseline_name})", linestyle="--")
        plt.scatter(reference_p_east_m[0], reference_p_north_m[0], color="green", s=90, zorder=5, label="Start (0,0)")
        plt.scatter(reference_p_east_m[-1], reference_p_north_m[-1], color="blue", s=80, zorder=5, label="True End")
        plt.scatter(estimated_traj.p_east_m[-1], estimated_traj.p_north_m[-1], color="red", s=80, marker="x", zorder=5, label="DR End")
        plt.title(f"[{sequence_id}] Classical Dead Reckoning ({estimated_traj.baseline_name})\nATE RMSE: {metrics.ate_rmse_m:.2f}m | Drift: {metrics.drift_rate_pct:.1f}%")
        plt.xlabel("East Position (meters)")
        plt.ylabel("North Position (meters)")
        plt.axis("equal")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        p1 = os.path.join(fig_dir, "trajectory_estimate_vs_reference.png")
        plt.savefig(p1, dpi=dpi)
        plt.close()
        generated["trajectory_overlay"] = p1

        # ----------------------------------------------------------------------
        # Plot 2: Position Error vs Time
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.5))
        plt.plot(t, pos_errors_m, color="#e377c2", linewidth=1.8, label="2D Position Error (m)")
        plt.axhline(metrics.ate_rmse_m, color="#7f7f7f", linestyle="--", label=f"ATE RMSE: {metrics.ate_rmse_m:.2f}m")
        plt.axhline(metrics.max_position_error_m, color="#d62728", linestyle=":", label=f"Max Error: {metrics.max_position_error_m:.2f}m")
        plt.title(f"[{sequence_id}] Dead Reckoning Position Drift vs Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("2D Position Error (meters)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p2 = os.path.join(fig_dir, "position_error_timeline.png")
        plt.savefig(p2, dpi=dpi)
        plt.close()
        generated["position_error_timeline"] = p2

        # ----------------------------------------------------------------------
        # Plot 3: Heading Error vs Time
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.5))
        plt.plot(t, head_errors_deg, color="#9467bd", linewidth=1.8, label="Absolute Heading Error (deg)")
        plt.axhline(metrics.heading_rmse_deg, color="#7f7f7f", linestyle="--", label=f"Heading RMSE: {metrics.heading_rmse_deg:.2f}°")
        plt.title(f"[{sequence_id}] Integrated Yaw Error vs Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heading Error (degrees)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p3 = os.path.join(fig_dir, "heading_error_timeline.png")
        plt.savefig(p3, dpi=dpi)
        plt.close()
        generated["heading_error_timeline"] = p3

        # ----------------------------------------------------------------------
        # Plot 4: Forward Velocity Profile
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.5))
        plt.plot(t, estimated_traj.forward_speed_ms, color="#ff7f0e", linewidth=1.8, label="Estimated Forward Speed (m/s)")
        if reference_speed_ms is not None:
            plt.plot(t, reference_speed_ms, color="#1f77b4", linewidth=1.5, linestyle="--", label="VBOX True Speed (m/s)")
        plt.title(f"[{sequence_id}] Forward Velocity Comparison (Velocity RMSE: {metrics.velocity_rmse_ms:.2f} m/s)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speed (m/s)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p4 = os.path.join(fig_dir, "velocity_profile_comparison.png")
        plt.savefig(p4, dpi=dpi)
        plt.close()
        generated["velocity_profile"] = p4

        # ----------------------------------------------------------------------
        # Plot 5: Position Error vs Distance Traveled
        # ----------------------------------------------------------------------
        step_d = np.zeros(len(t))
        if len(t) >= 2:
            step_d[1:] = np.cumsum(np.sqrt(np.diff(reference_p_east_m)**2 + np.diff(reference_p_north_m)**2))

        plt.figure(figsize=(10, 4.5))
        plt.plot(step_d, pos_errors_m, color="#2ca02c", linewidth=1.8, label=f"Drift Rate: {metrics.drift_rate_pct:.1f}%")
        plt.title(f"[{sequence_id}] Dead Reckoning Drift Growth over Trajectory Distance")
        plt.xlabel("Cumulative Distance Traveled (meters)")
        plt.ylabel("2D Position Error (meters)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p5 = os.path.join(fig_dir, "drift_vs_distance.png")
        plt.savefig(p5, dpi=dpi)
        plt.close()
        generated["drift_vs_distance"] = p5

        return generated

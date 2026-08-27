"""
Comprehensive Diagnostic Visualizer for Project AGASTYA (Objective 5).
Generates all 12 required engineering figures for model training, residual prediction,
trajectory rollouts, GNSS outages, and ablation studies.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navigation_engine.state import DeadReckoningTrajectory


class Objective5Visualizer:
    """
    Renders all 12 engineering diagnostic figures for Objective 5.
    """
    @classmethod
    def generate_all_plots(
        cls,
        training_history: Dict[str, Any],
        y_true_phys: np.ndarray,
        y_pred_phys: np.ndarray,
        timestamps_sec: np.ndarray,
        classical_traj: DeadReckoningTrajectory,
        ai_traj: DeadReckoningTrajectory,
        ref_east_m: np.ndarray,
        ref_north_m: np.ndarray,
        ref_speed_ms: Optional[np.ndarray],
        ref_heading_rad: Optional[np.ndarray],
        outage_records: List[Dict[str, Any]],
        ablation_records: Dict[str, Dict[str, Any]],
        output_dir: str = "artifacts/objective5/figures",
        sequence_id: str = "sync_02",
        dpi: int = 150
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # ----------------------------------------------------------------------
        # 1. training_validation_loss.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8, 4.5))
        hist = training_history.get("history", {})
        epochs = hist.get("epochs", [])
        plt.plot(epochs, hist.get("train_loss", []), label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(epochs, hist.get("val_loss", []), label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
        plt.title("Multi-Task Training & Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Standardized MSE Loss")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p1 = os.path.join(output_dir, "training_validation_loss.png")
        plt.savefig(p1, dpi=dpi)
        plt.close()
        generated["training_validation_loss"] = p1

        # ----------------------------------------------------------------------
        # 2. velocity_residual_true_vs_pred.png
        # ----------------------------------------------------------------------
        v_true = y_true_phys[:, 0]
        v_pred = y_pred_phys[:, 0]
        plt.figure(figsize=(9, 4.5))
        plt.plot(timestamps_sec, v_true, label="True δv (m/s)", color="black", alpha=0.7, lw=1.5)
        plt.plot(timestamps_sec, v_pred, label="Predicted δv (m/s)", color="#2ca02c", alpha=0.85, lw=1.5)
        plt.title(f"[{sequence_id}] Velocity Residual: Ground Truth vs Model Prediction")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Residual δv (m/s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p2 = os.path.join(output_dir, "velocity_residual_true_vs_pred.png")
        plt.savefig(p2, dpi=dpi)
        plt.close()
        generated["velocity_residual_true_vs_pred"] = p2

        # ----------------------------------------------------------------------
        # 3. yaw_residual_true_vs_pred.png
        # ----------------------------------------------------------------------
        w_true = y_true_phys[:, 1]
        w_pred = y_pred_phys[:, 1]
        plt.figure(figsize=(9, 4.5))
        plt.plot(timestamps_sec, w_true, label="True δω (rad/s)", color="black", alpha=0.7, lw=1.5)
        plt.plot(timestamps_sec, w_pred, label="Predicted δω (rad/s)", color="#9467bd", alpha=0.85, lw=1.5)
        plt.title(f"[{sequence_id}] Yaw Rate Residual: Ground Truth vs Model Prediction")
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw Rate Residual δω (rad/s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p3 = os.path.join(output_dir, "yaw_residual_true_vs_pred.png")
        plt.savefig(p3, dpi=dpi)
        plt.close()
        generated["yaw_residual_true_vs_pred"] = p3

        # ----------------------------------------------------------------------
        # 4. velocity_prediction_error_distribution.png
        # ----------------------------------------------------------------------
        v_err = v_pred - v_true
        plt.figure(figsize=(8, 4.5))
        plt.hist(v_err, bins=35, color="#17becf", edgecolor="black", alpha=0.75)
        plt.axvline(np.mean(v_err), color="red", linestyle="--", label=f"Mean Error: {np.mean(v_err):+.4f} m/s")
        plt.axvline(0, color="gray", linestyle=":")
        plt.title(f"[{sequence_id}] Velocity Residual Prediction Error Distribution")
        plt.xlabel("Error (Pred - True) (m/s)")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p4 = os.path.join(output_dir, "velocity_prediction_error_distribution.png")
        plt.savefig(p4, dpi=dpi)
        plt.close()
        generated["velocity_prediction_error_distribution"] = p4

        # ----------------------------------------------------------------------
        # 5. yaw_prediction_error_distribution.png
        # ----------------------------------------------------------------------
        w_err = w_pred - w_true
        plt.figure(figsize=(8, 4.5))
        plt.hist(w_err, bins=35, color="#bcbd22", edgecolor="black", alpha=0.75)
        plt.axvline(np.mean(w_err), color="red", linestyle="--", label=f"Mean Error: {np.mean(w_err):+.5f} rad/s")
        plt.axvline(0, color="gray", linestyle=":")
        plt.title(f"[{sequence_id}] Yaw Rate Prediction Error Distribution")
        plt.xlabel("Error (Pred - True) (rad/s)")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p5 = os.path.join(output_dir, "yaw_prediction_error_distribution.png")
        plt.savefig(p5, dpi=dpi)
        plt.close()
        generated["yaw_prediction_error_distribution"] = p5

        # ----------------------------------------------------------------------
        # 6. classical_vs_ai_trajectory.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8, 7))
        plt.plot(ref_east_m, ref_north_m, label="Reference Ground Truth (VBOX)", color="black", lw=2)
        plt.plot(classical_traj.p_east_m, classical_traj.p_north_m, label="Classical Baseline A", color="#1f77b4", linestyle="--", lw=1.8)
        plt.plot(ai_traj.p_east_m, ai_traj.p_north_m, label="AI-Corrected Dead Reckoning", color="#2ca02c", linestyle="-.", lw=2)
        plt.scatter(ref_east_m[0], ref_north_m[0], color="green", s=60, marker="o", label="Start", zorder=5)
        plt.scatter(ref_east_m[-1], ref_north_m[-1], color="red", s=60, marker="x", label="End", zorder=5)
        plt.title(f"[{sequence_id}] Trajectory Comparison: Classical vs AI vs Reference")
        plt.xlabel("East Position (m)")
        plt.ylabel("North Position (m)")
        plt.legend()
        plt.axis("equal")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p6 = os.path.join(output_dir, "classical_vs_ai_trajectory.png")
        plt.savefig(p6, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_trajectory"] = p6

        # ----------------------------------------------------------------------
        # 7. classical_vs_ai_position_error.png
        # ----------------------------------------------------------------------
        c_pos_err = np.sqrt((classical_traj.p_east_m - ref_east_m)**2 + (classical_traj.p_north_m - ref_north_m)**2)
        ai_pos_err = np.sqrt((ai_traj.p_east_m - ref_east_m)**2 + (ai_traj.p_north_m - ref_north_m)**2)
        plt.figure(figsize=(9, 4.5))
        plt.plot(classical_traj.timestamps_sec, c_pos_err, label=f"Classical Baseline A (Final: {c_pos_err[-1]:.2f}m)", color="#1f77b4", lw=1.8)
        plt.plot(ai_traj.timestamps_sec, ai_pos_err, label=f"AI-Corrected (Final: {ai_pos_err[-1]:.2f}m)", color="#2ca02c", lw=2)
        plt.title(f"[{sequence_id}] Position Error Timeline Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Euclidean Position Error (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p7 = os.path.join(output_dir, "classical_vs_ai_position_error.png")
        plt.savefig(p7, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_position_error"] = p7

        # ----------------------------------------------------------------------
        # 8. classical_vs_ai_drift_vs_distance.png
        # ----------------------------------------------------------------------
        c_dist = np.cumsum(np.insert(np.sqrt(np.diff(classical_traj.p_east_m)**2 + np.diff(classical_traj.p_north_m)**2), 0, 0))
        plt.figure(figsize=(9, 4.5))
        plt.plot(c_dist, c_pos_err, label="Classical Baseline A", color="#1f77b4", lw=1.8)
        plt.plot(c_dist, ai_pos_err, label="AI-Corrected", color="#2ca02c", lw=2)
        plt.title(f"[{sequence_id}] Drift vs Traveled Distance")
        plt.xlabel("Cumulative Distance (m)")
        plt.ylabel("Position Error (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p8 = os.path.join(output_dir, "classical_vs_ai_drift_vs_distance.png")
        plt.savefig(p8, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_drift_vs_distance"] = p8

        # ----------------------------------------------------------------------
        # 9. classical_vs_ai_velocity_profile.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9, 4.5))
        if ref_speed_ms is not None:
            plt.plot(classical_traj.timestamps_sec, ref_speed_ms, label="VBOX Speed", color="black", lw=1.5, alpha=0.7)
        plt.plot(classical_traj.timestamps_sec, classical_traj.forward_speed_ms, label="Classical Speed", color="#1f77b4", linestyle="--", lw=1.5)
        plt.plot(ai_traj.timestamps_sec, ai_traj.forward_speed_ms, label="AI-Corrected Speed", color="#2ca02c", lw=1.8)
        plt.title(f"[{sequence_id}] Forward Velocity Profile Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Forward Speed (m/s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p9 = os.path.join(output_dir, "classical_vs_ai_velocity_profile.png")
        plt.savefig(p9, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_velocity_profile"] = p9

        # ----------------------------------------------------------------------
        # 10. classical_vs_ai_heading_error.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9, 4.5))
        if ref_heading_rad is not None:
            c_h_err = np.degrees((classical_traj.heading_rad - ref_heading_rad + np.pi) % (2*np.pi) - np.pi)
            ai_h_err = np.degrees((ai_traj.heading_rad - ref_heading_rad + np.pi) % (2*np.pi) - np.pi)
            plt.plot(classical_traj.timestamps_sec, c_h_err, label="Classical Heading Error", color="#1f77b4", lw=1.8)
            plt.plot(ai_traj.timestamps_sec, ai_h_err, label="AI-Corrected Heading Error", color="#2ca02c", lw=2)
            plt.ylabel("Heading Error (deg)")
        plt.title(f"[{sequence_id}] Heading Error Timeline Comparison")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p10 = os.path.join(output_dir, "classical_vs_ai_heading_error.png")
        plt.savefig(p10, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_heading_error"] = p10

        # ----------------------------------------------------------------------
        # 11. outage_comparison.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8.5, 4.5))
        durations = [r["duration_sec"] for r in outage_records]
        c_ates = [r["classical"]["outage_ate_rmse_m"] for r in outage_records]
        ai_ates = [r["ai_corrected"]["outage_ate_rmse_m"] for r in outage_records]
        x = np.arange(len(durations))
        width = 0.35
        plt.bar(x - width/2, c_ates, width, label="Classical Baseline A", color="#1f77b4")
        plt.bar(x + width/2, ai_ates, width, label="AI-Corrected Baseline", color="#2ca02c")
        plt.xticks(x, [f"{d}s Outage" for d in durations])
        plt.ylabel("Outage ATE RMSE (m)")
        plt.title(f"[{sequence_id}] Standardized GNSS Outage Comparison (Entry t = 20.0s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p11 = os.path.join(output_dir, "outage_comparison.png")
        plt.savefig(p11, dpi=dpi)
        plt.close()
        generated["outage_comparison"] = p11

        # ----------------------------------------------------------------------
        # 12. ablation_comparison.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9, 4.5))
        abl_names = list(ablation_records.keys())
        abl_labels = ["Classical Only", "Velocity Only", "Yaw Only", "Full Correction"]
        abl_ates = [ablation_records[k]["metrics"]["ate_rmse_m"] for k in abl_names]
        colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]
        bars = plt.bar(abl_labels, abl_ates, color=colors, edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, abl_ates):
            plt.text(bar.get_x() + bar.get_width()/2.0, val + 0.05, f"{val:.3f}m", ha="center", va="bottom", fontweight="bold")
        plt.title(f"[{sequence_id}] Scientific Ablation Study (ATE RMSE)")
        plt.ylabel("ATE RMSE (m)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p12 = os.path.join(output_dir, "ablation_comparison.png")
        plt.savefig(p12, dpi=dpi)
        plt.close()
        generated["ablation_comparison"] = p12

        return generated

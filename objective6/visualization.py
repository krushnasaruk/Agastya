"""
Diagnostic Visualization Engine for Objective 6.
Generates all 14 required engineering figures for closed-loop navigation, uncertainty calibration,
GNSS outage robustness, maneuver breakdown, and safety fallback telemetry.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navigation_engine.state import DeadReckoningTrajectory


class Objective6Visualizer:
    """
    Renders all 14 comprehensive engineering figures for Objective 6.
    """
    @classmethod
    def generate_all_plots(
        cls,
        exp_results: Dict[str, Any],
        ref_df: pd.DataFrame,
        output_dir: str = "artifacts/objective6/figures",
        sequence_id: str = "sync_02",
        dpi: int = 150
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        ref_east = ref_df["pos_east_m"].to_numpy()
        ref_north = ref_df["pos_north_m"].to_numpy()
        ref_speed = ref_df.get("ground_speed_ms", None)
        ref_heading = ref_df.get("heading_rad", None)

        c_traj: DeadReckoningTrajectory = exp_results["classical_traj"]
        o5_traj: DeadReckoningTrajectory = exp_results["obj5_v_traj"]
        o6_traj: DeadReckoningTrajectory = exp_results["obj6_traj"]
        yaw_traj: DeadReckoningTrajectory = exp_results["yaw_traj"]
        dec_df: pd.DataFrame = exp_results["decisions_df"]
        t_arr = c_traj.timestamps_sec

        # ----------------------------------------------------------------------
        # 1. classical_vs_ai_trajectory.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8.5, 7))
        plt.plot(ref_east, ref_north, label="Ground Truth (VBOX RTK)", color="black", lw=2)
        plt.plot(c_traj.p_east_m, c_traj.p_north_m, label="Classical Baseline A", color="#1f77b4", linestyle="--", lw=1.6)
        plt.plot(o5_traj.p_east_m, o5_traj.p_north_m, label="Objective 5 Velocity-Only", color="#ff7f0e", linestyle=":", lw=1.8)
        plt.plot(o6_traj.p_east_m, o6_traj.p_north_m, label="Objective 6 Selective Velocity", color="#2ca02c", linestyle="-.", lw=2)
        plt.scatter(ref_east[0], ref_north[0], color="green", s=60, marker="o", label="Start", zorder=5)
        plt.scatter(ref_east[-1], ref_north[-1], color="red", s=60, marker="x", label="End", zorder=5)
        plt.title(f"[{sequence_id}] Trajectory Comparison: Classical vs Obj5 vs Obj6")
        plt.xlabel("East Position (m)")
        plt.ylabel("North Position (m)")
        plt.legend()
        plt.axis("equal")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p1 = os.path.join(output_dir, "classical_vs_ai_trajectory.png")
        plt.savefig(p1, dpi=dpi)
        plt.close()
        generated["classical_vs_ai_trajectory"] = p1

        # ----------------------------------------------------------------------
        # 2. position_error_comparison.png
        # ----------------------------------------------------------------------
        c_pos_err = exp_results["pos_errors"]["classical"]
        o5_pos_err = exp_results["pos_errors"]["obj5_velocity"]
        o6_pos_err = exp_results["pos_errors"]["obj6_selective"]
        plt.figure(figsize=(9.5, 4.5))
        plt.plot(t_arr, c_pos_err, label=f"Classical Baseline A (Final: {c_pos_err[-1]:.2f}m)", color="#1f77b4", lw=1.6)
        plt.plot(t_arr, o5_pos_err, label=f"Obj5 Velocity (Final: {o5_pos_err[-1]:.2f}m)", color="#ff7f0e", linestyle=":", lw=1.8)
        plt.plot(t_arr, o6_pos_err, label=f"Obj6 Selective Velocity (Final: {o6_pos_err[-1]:.2f}m)", color="#2ca02c", lw=2)
        plt.title(f"[{sequence_id}] Position Error Timeline Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Euclidean Position Error (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p2 = os.path.join(output_dir, "position_error_comparison.png")
        plt.savefig(p2, dpi=dpi)
        plt.close()
        generated["position_error_comparison"] = p2

        # ----------------------------------------------------------------------
        # 3. velocity_residual_prediction.png
        # ----------------------------------------------------------------------
        raw_dv = dec_df["raw_delta_v"].to_numpy()
        app_dv = dec_df["applied_delta_v"].to_numpy()
        applied_mask = dec_df["is_applied"].to_numpy()
        plt.figure(figsize=(9.5, 4.5))
        plt.plot(t_arr, raw_dv, label="Raw AI Prediction (m/s)", color="#aec7e8", lw=1.2, alpha=0.8)
        plt.plot(t_arr, app_dv, label="Selectively Applied Residual (m/s)", color="#2ca02c", lw=1.8)
        plt.axhline(0, color="gray", linestyle=":", alpha=0.6)
        plt.title(f"[{sequence_id}] Velocity Residual: Raw Predictions vs Selectively Applied")
        plt.xlabel("Time (s)")
        plt.ylabel("Delta v (m/s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p3 = os.path.join(output_dir, "velocity_residual_prediction.png")
        plt.savefig(p3, dpi=dpi)
        plt.close()
        generated["velocity_residual_prediction"] = p3

        # ----------------------------------------------------------------------
        # 4. velocity_residual_error.png
        # ----------------------------------------------------------------------
        if ref_speed is not None:
            true_dv = ref_speed.to_numpy() - dec_df["classical_speed_ms"].to_numpy()
            v_pred_err = raw_dv - true_dv
            plt.figure(figsize=(9.5, 4.5))
            plt.plot(t_arr, v_pred_err, label="Raw Prediction Error (Pred - True)", color="#d62728", lw=1.5)
            plt.axhline(0, color="black", linestyle="--")
            plt.title(f"[{sequence_id}] Velocity Residual Prediction Error Timeline")
            plt.xlabel("Time (s)")
            plt.ylabel("Error (m/s)")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            p4 = os.path.join(output_dir, "velocity_residual_error.png")
            plt.savefig(p4, dpi=dpi)
            plt.close()
            generated["velocity_residual_error"] = p4

        # ----------------------------------------------------------------------
        # 5. confidence_vs_error.png
        # ----------------------------------------------------------------------
        calib_data = exp_results["experiment_j_calibration"]
        bins = calib_data.get("bins", [])
        if len(bins) > 0:
            plt.figure(figsize=(8, 4.5))
            bin_labels = [f"{b['range'][0]}-{b['range'][1]}" for b in bins]
            maes = [b["mean_absolute_error"] for b in bins]
            counts = [b["sample_count"] for b in bins]
            x_b = np.arange(len(bins))
            bars = plt.bar(x_b, maes, color="#9467bd", edgecolor="black", alpha=0.85)
            for bar, c in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.0002, f"N={c}", ha="center", va="bottom", fontsize=8)
            plt.xticks(x_b, bin_labels)
            plt.title(f"Predictive Confidence vs Residual MAE ({calib_data.get('calibration_status')})")
            plt.xlabel("Confidence Score Bin")
            plt.ylabel("Mean Absolute Error (m/s)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            p5 = os.path.join(output_dir, "confidence_vs_error.png")
            plt.savefig(p5, dpi=dpi)
            plt.close()
            generated["confidence_vs_error"] = p5

        # ----------------------------------------------------------------------
        # 6. ood_score_distribution.png
        # ----------------------------------------------------------------------
        ood_scores = dec_df["ood_score"].to_numpy()
        plt.figure(figsize=(8.5, 4.5))
        plt.hist(ood_scores, bins=35, color="#17becf", edgecolor="black", alpha=0.75)
        plt.axvline(3.5, color="red", linestyle="--", lw=2, label="OOD Fallback Threshold (3.5)")
        plt.title(f"[{sequence_id}] Feature-Space OOD Distance Distribution")
        plt.xlabel("Normalized Distance (d_OOD)")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p6 = os.path.join(output_dir, "ood_score_distribution.png")
        plt.savefig(p6, dpi=dpi)
        plt.close()
        generated["ood_score_distribution"] = p6

        # ----------------------------------------------------------------------
        # 7. correction_acceptance_rate.png
        # ----------------------------------------------------------------------
        cum_applied = np.cumsum(applied_mask) / (np.arange(len(applied_mask)) + 1) * 100.0
        plt.figure(figsize=(9.5, 4.5))
        plt.plot(t_arr, cum_applied, label=f"Cumulative AI Application Rate (Final: {cum_applied[-1]:.1f}%)", color="#2ca02c", lw=2)
        plt.title(f"[{sequence_id}] Cumulative AI Residual Application Rate Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Application Rate (%)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p7 = os.path.join(output_dir, "correction_acceptance_rate.png")
        plt.savefig(p7, dpi=dpi)
        plt.close()
        generated["correction_acceptance_rate"] = p7

        # ----------------------------------------------------------------------
        # 8. fallback_reason_distribution.png
        # ----------------------------------------------------------------------
        reason_counts = exp_results["experiment_i_ai_usage"]["fallback_reason_breakdown"]
        plt.figure(figsize=(8, 5))
        if len(reason_counts) > 0:
            labels = list(reason_counts.keys())
            vals = list(reason_counts.values())
            plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=140, colors=plt.cm.Set3.colors)
        else:
            plt.text(0.5, 0.5, "100% Accepted (No Fallback Triggered)", ha="center", va="center")
        plt.title(f"[{sequence_id}] Distribution of Fallback Reasons")
        plt.tight_layout()
        p8 = os.path.join(output_dir, "fallback_reason_distribution.png")
        plt.savefig(p8, dpi=dpi)
        plt.close()
        generated["fallback_reason_distribution"] = p8

        # ----------------------------------------------------------------------
        # 9. outage_duration_vs_ate.png
        # ----------------------------------------------------------------------
        outages = exp_results["experiment_g_outages"]
        plt.figure(figsize=(8.5, 4.5))
        durs = [o["duration_sec"] for o in outages]
        c_ates = [o["classical"]["ate_rmse_m"] for o in outages]
        o5_ates = [o["objective5_velocity"]["ate_rmse_m"] for o in outages]
        o6_ates = [o["objective6_selective"]["ate_rmse_m"] for o in outages]
        x_o = np.arange(len(durs))
        w = 0.25
        plt.bar(x_o - w, c_ates, w, label="Classical Baseline", color="#1f77b4")
        plt.bar(x_o, o5_ates, w, label="Obj5 Velocity-Only", color="#ff7f0e")
        plt.bar(x_o + w, o6_ates, w, label="Obj6 Selective Velocity", color="#2ca02c")
        plt.xticks(x_o, [f"{d}s" for d in durs])
        plt.title("Standardized GNSS Outage Duration vs ATE RMSE (Entry t = 20.0s)")
        plt.xlabel("Outage Duration")
        plt.ylabel("Outage ATE RMSE (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p9 = os.path.join(output_dir, "outage_duration_vs_ate.png")
        plt.savefig(p9, dpi=dpi)
        plt.close()
        generated["outage_duration_vs_ate"] = p9

        # ----------------------------------------------------------------------
        # 10. maneuver_wise_ate.png
        # ----------------------------------------------------------------------
        m_data = exp_results["experiment_h_maneuvers"]
        plt.figure(figsize=(9.5, 4.5))
        m_names = list(m_data["classical"].keys())
        m_c_ates = [m_data["classical"][k]["ate_rmse_m"] for k in m_names]
        m_o6_ates = [m_data["objective6_selective"][k]["ate_rmse_m"] for k in m_names]
        x_m = np.arange(len(m_names))
        w_m = 0.35
        plt.bar(x_m - w_m/2, m_c_ates, w_m, label="Classical Baseline", color="#1f77b4")
        plt.bar(x_m + w_m/2, m_o6_ates, w_m, label="Obj6 Selective", color="#2ca02c")
        plt.xticks(x_m, [m.replace("_", " ").title() for m in m_names], rotation=20)
        plt.title(f"[{sequence_id}] Maneuver-Stratified ATE RMSE Breakdown")
        plt.ylabel("ATE RMSE (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p10 = os.path.join(output_dir, "maneuver_wise_ate.png")
        plt.savefig(p10, dpi=dpi)
        plt.close()
        generated["maneuver_wise_ate"] = p10

        # ----------------------------------------------------------------------
        # 11. yaw_residual_failure_analysis.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9.5, 4.5))
        if ref_heading is not None:
            c_h_err = np.degrees((c_traj.heading_rad - ref_heading + np.pi) % (2*np.pi) - np.pi)
            yaw_h_err = np.degrees((yaw_traj.heading_rad - ref_heading + np.pi) % (2*np.pi) - np.pi)
            plt.plot(t_arr, c_h_err, label="Classical Heading Error (Baseline)", color="#1f77b4", lw=1.8)
            plt.plot(t_arr, yaw_h_err, label="Yaw AI Residual Heading Error (Degraded)", color="#d62728", lw=2)
            plt.ylabel("Heading Error (deg)")
        plt.title(f"[{sequence_id}] Yaw Residual Integration Failure Analysis (Heading Drift)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p11 = os.path.join(output_dir, "yaw_residual_failure_analysis.png")
        plt.savefig(p11, dpi=dpi)
        plt.close()
        generated["yaw_residual_failure_analysis"] = p11

        # ----------------------------------------------------------------------
        # 12. temporal_consistency_analysis.png
        # ----------------------------------------------------------------------
        v_jumps = dec_df["velocity_jump_ms"].to_numpy()
        plt.figure(figsize=(8.5, 4.5))
        plt.hist(v_jumps, bins=35, color="#8c564b", edgecolor="black", alpha=0.75)
        plt.axvline(0.60, color="red", linestyle="--", lw=2, label="Max Permissible Jump (0.60 m/s)")
        plt.title(f"[{sequence_id}] Temporal Velocity Jump Distribution")
        plt.xlabel("Step-to-Step Jump |dv[k] - dv[k-1]| (m/s)")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p12 = os.path.join(output_dir, "temporal_consistency_analysis.png")
        plt.savefig(p12, dpi=dpi)
        plt.close()
        generated["temporal_consistency_analysis"] = p12

        # ----------------------------------------------------------------------
        # 13. residual_histograms.png
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9, 4.5))
        plt.hist(raw_dv, bins=35, color="#2ca02c", alpha=0.7, label="Raw delta_v (m/s)", edgecolor="black")
        plt.hist(dec_df["raw_delta_w"].to_numpy(), bins=35, color="#9467bd", alpha=0.7, label="Raw delta_omega (rad/s)", edgecolor="black")
        plt.title(f"[{sequence_id}] Model Predicted Residual Distributions")
        plt.xlabel("Residual Value (SI Units)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p13 = os.path.join(output_dir, "residual_histograms.png")
        plt.savefig(p13, dpi=dpi)
        plt.close()
        generated["residual_histograms"] = p13

        # ----------------------------------------------------------------------
        # 14. navigation_drift_comparison.png
        # ----------------------------------------------------------------------
        c_dist = np.cumsum(np.insert(np.sqrt(np.diff(c_traj.p_east_m)**2 + np.diff(c_traj.p_north_m)**2), 0, 0))
        plt.figure(figsize=(9.5, 4.5))
        plt.plot(c_dist, c_pos_err, label="Classical Baseline A", color="#1f77b4", lw=1.6)
        plt.plot(c_dist, o5_pos_err, label="Obj5 Velocity-Only", color="#ff7f0e", linestyle=":", lw=1.8)
        plt.plot(c_dist, o6_pos_err, label="Obj6 Selective Velocity", color="#2ca02c", lw=2)
        plt.title(f"[{sequence_id}] Navigation Drift vs Cumulative Travel Distance")
        plt.xlabel("Cumulative Distance (m)")
        plt.ylabel("Position Error (m)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p14 = os.path.join(output_dir, "navigation_drift_comparison.png")
        plt.savefig(p14, dpi=dpi)
        plt.close()
        generated["navigation_drift_comparison"] = p14

        return generated

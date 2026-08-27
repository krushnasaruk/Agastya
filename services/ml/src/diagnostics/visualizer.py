"""
Diagnostic Visualizer for AI Error Modeling & Residual Formulation (Objective 4).
Renders 5 engineering diagnostic figures:
1. Candidate Residual Distributions & Box Plots
2. Residual vs Speed and Acceleration
3. Residual vs Yaw Rate and Curvature
4. Residual Autocorrelation Function (ACF)
5. Physical Error Decomposition Summary
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..targets.residual_targets import ResidualTargetsContainer


class Objective4DiagnosticsVisualizer:
    """
    Renders diagnostic figures for Objective 4 residual learning formulation.
    """
    @classmethod
    def generate_all_plots(
        cls,
        targets: ResidualTargetsContainer,
        classical_speed_ms: np.ndarray,
        classical_yaw_rate_rads: np.ndarray,
        accel_x_ms2: np.ndarray,
        output_dir: str = "data/processed",
        sequence_id: str = "sync_01",
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        Generate and save the 5 Objective 4 diagnostic figures.
        """
        fig_dir = os.path.join(output_dir, "reports", "figures", sequence_id, "ai_formulation")
        os.makedirs(fig_dir, exist_ok=True)
        t = targets.timestamps_sec
        generated: Dict[str, str] = {}

        # ----------------------------------------------------------------------
        # Figure 1: Candidate Residual Distributions (Target A vs Target B)
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].hist(targets.delta_velocity_ms, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
        axes[0].axvline(np.median(targets.delta_velocity_ms), color="red", linestyle="--", label=f"Median: {np.median(targets.delta_velocity_ms):.4f} m/s")
        axes[0].set_title("Target A: Velocity Residual Distribution (δv)")
        axes[0].set_xlabel("Velocity Residual (m/s)")
        axes[0].set_ylabel("Sample Count")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].hist(np.degrees(targets.delta_heading_rad), bins=30, color="#ff7f0e", edgecolor="black", alpha=0.7)
        axes[1].axvline(np.median(np.degrees(targets.delta_heading_rad)), color="red", linestyle="--", label=f"Median: {np.median(np.degrees(targets.delta_heading_rad)):.4f}°")
        axes[1].set_title("Target B: Heading Residual Distribution (δψ)")
        axes[1].set_xlabel("Heading Residual (deg)")
        axes[1].set_ylabel("Sample Count")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        p1 = os.path.join(fig_dir, "residual_distributions.png")
        plt.savefig(p1, dpi=dpi)
        plt.close()
        generated["residual_distributions"] = p1

        # ----------------------------------------------------------------------
        # Figure 2: Residual vs Forward Speed & Acceleration
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(classical_speed_ms, targets.delta_velocity_ms, color="#2ca02c", alpha=0.6, s=15)
        axes[0].set_title("Velocity Residual vs Forward Speed")
        axes[0].set_xlabel("Classical Forward Speed (m/s)")
        axes[0].set_ylabel("Velocity Residual δv (m/s)")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].scatter(accel_x_ms2, targets.delta_velocity_ms, color="#d62728", alpha=0.6, s=15)
        axes[1].set_title("Velocity Residual vs Longitudinal Accel")
        axes[1].set_xlabel("Longitudinal Accel a_x (m/s²)")
        axes[1].set_ylabel("Velocity Residual δv (m/s)")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        p2 = os.path.join(fig_dir, "residual_vs_kinematics.png")
        plt.savefig(p2, dpi=dpi)
        plt.close()
        generated["residual_vs_kinematics"] = p2

        # ----------------------------------------------------------------------
        # Figure 3: Residual vs Yaw Rate & Curvature
        # ----------------------------------------------------------------------
        curvature = np.abs(classical_yaw_rate_rads) / np.maximum(classical_speed_ms, 0.1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(classical_yaw_rate_rads, targets.delta_yaw_rate_rads, color="#9467bd", alpha=0.6, s=15)
        axes[0].set_title("Yaw Rate Residual vs Chassis Yaw Rate")
        axes[0].set_xlabel("Yaw Rate ω_z (rad/s)")
        axes[0].set_ylabel("Yaw Rate Residual δω (rad/s)")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].scatter(curvature, targets.delta_velocity_ms, color="#8c564b", alpha=0.6, s=15)
        axes[1].set_title("Velocity Residual vs Path Curvature")
        axes[1].set_xlabel("Path Curvature κ (1/m)")
        axes[1].set_ylabel("Velocity Residual δv (m/s)")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        p3 = os.path.join(fig_dir, "residual_vs_turning.png")
        plt.savefig(p3, dpi=dpi)
        plt.close()
        generated["residual_vs_turning"] = p3

        # ----------------------------------------------------------------------
        # Figure 4: Residual Autocorrelation Function (ACF)
        # ----------------------------------------------------------------------
        max_lags = min(30, len(targets.delta_velocity_ms) // 4)
        lags = np.arange(max_lags)
        v_clean = targets.delta_velocity_ms - np.mean(targets.delta_velocity_ms)
        var_v = np.sum(v_clean ** 2)
        acf = np.array([np.sum(v_clean[:len(v_clean)-l] * v_clean[l:]) / var_v if l > 0 else 1.0 for l in lags])

        plt.figure(figsize=(8.5, 4.5))
        plt.stem(lags, acf, basefmt=" ")
        plt.axhline(0, color="gray", linestyle="--")
        plt.axhline(0.2, color="red", linestyle=":", alpha=0.6)
        plt.axhline(-0.2, color="red", linestyle=":", alpha=0.6)
        plt.title(f"[{sequence_id}] Velocity Residual Autocorrelation Function (ACF)")
        plt.xlabel("Lag (Epochs @ 10 Hz)")
        plt.ylabel("Autocorrelation Coefficient")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p4 = os.path.join(fig_dir, "residual_autocorrelation.png")
        plt.savefig(p4, dpi=dpi)
        plt.close()
        generated["residual_autocorrelation"] = p4

        # ----------------------------------------------------------------------
        # Figure 5: Target Comparison Summary Box Plot
        # ----------------------------------------------------------------------
        plt.figure(figsize=(9, 4.5))
        data_to_plot = [
            targets.delta_velocity_ms,
            targets.delta_yaw_rate_rads,
            targets.delta_disp_east_m,
            targets.delta_disp_north_m
        ]
        plt.boxplot(data_to_plot, tick_labels=["Target A (δv)", "Target B1 (δω)", "Target C1 (δdE)", "Target C2 (δdN)"])
        plt.title(f"[{sequence_id}] Residual Target Spread & Outlier Comparison")
        plt.ylabel("Error Magnitude (SI Units)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p5 = os.path.join(fig_dir, "target_comparison_boxplot.png")
        plt.savefig(p5, dpi=dpi)
        plt.close()
        generated["target_comparison_boxplot"] = p5

        return generated

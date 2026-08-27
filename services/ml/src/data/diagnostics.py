"""
Diagnostic Visualization Engine for IO-VNBD Data Engineering.
Generates engineering diagnostic charts:
1. Sensor availability timeline
2. Reference trajectory in local ENU coordinates
3. Wheel-derived speed vs VBOX ground-truth speed
4. Measured gyro yaw rate vs kinematic wheel differential yaw rate
5. Quality mask timeline
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .pipeline import ProcessedSequencePackage


class DataQualityVisualizer:
    """
    Generates high-resolution engineering diagnostic figures for preprocessed sequences.
    """
    @classmethod
    def generate_diagnostic_plots(
        cls,
        package: ProcessedSequencePackage,
        output_dir: str,
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        Render and save the 5 mandatory engineering diagnostic plots.
        """
        fig_dir = os.path.join(output_dir, "reports", "figures", package.sequence_id)
        os.makedirs(fig_dir, exist_ok=True)
        nav_inputs = package.navigation_inputs_df
        ref_df = package.reference_trajectory_df
        q_df = package.quality_df
        t = package.timestamps_sec

        generated_files: Dict[str, str] = {}

        # ----------------------------------------------------------------------
        # Plot 1: Sensor Availability Timeline
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4))
        plt.plot(t, q_df["sensor_valid_mask"].astype(int), label="CAN Sensors", color="#1f77b4", linewidth=1.5)
        plt.plot(t, q_df["gps_valid_mask"].astype(int) * 0.9, label="GPS Reference", color="#2ca02c", linewidth=1.5)
        plt.plot(t, q_df["overall_valid_mask"].astype(int) * 0.8, label="Composite Valid", color="#d62728", linewidth=1.5, linestyle="--")
        plt.title(f"[{package.sequence_id}] Sensor & Reference Availability Timeline")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Availability Status (1 = Active, 0 = Inactive)")
        plt.ylim(-0.1, 1.2)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p1_path = os.path.join(fig_dir, "sensor_availability.png")
        plt.savefig(p1_path, dpi=dpi)
        plt.close()
        generated_files["sensor_availability"] = p1_path

        # ----------------------------------------------------------------------
        # Plot 2: 2D Reference Trajectory (Local ENU)
        # ----------------------------------------------------------------------
        if "pos_east_m" in ref_df and "pos_north_m" in ref_df:
            plt.figure(figsize=(7, 7))
            east = ref_df["pos_east_m"].to_numpy()
            north = ref_df["pos_north_m"].to_numpy()
            plt.plot(east, north, color="#2b5c8f", linewidth=2.0, label="Reference Path (VBOX)")
            plt.scatter(east[0], north[0], color="green", s=80, zorder=5, label="Start (0,0)")
            plt.scatter(east[-1], north[-1], color="red", s=80, zorder=5, label="End")
            plt.title(f"[{package.sequence_id}] 2D Metric Trajectory (Local ENU)\nTotal Dist: {package.reference_trajectory.total_distance_m:.1f}m")
            plt.xlabel("East Position (meters)")
            plt.ylabel("North Position (meters)")
            plt.axis("equal")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="best")
            plt.tight_layout()
            p2_path = os.path.join(fig_dir, "reference_trajectory_enu.png")
            plt.savefig(p2_path, dpi=dpi)
            plt.close()
            generated_files["reference_trajectory"] = p2_path

        # ----------------------------------------------------------------------
        # Plot 3: Wheel-Derived Speed vs VBOX Ground-Truth Speed
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.5))
        if "wheel_speed_rl_ms" in nav_inputs and "wheel_speed_rr_ms" in nav_inputs:
            v_wheel_avg = (nav_inputs["wheel_speed_rl_ms"] + nav_inputs["wheel_speed_rr_ms"]) * 0.5
            plt.plot(t, v_wheel_avg, label="Rear Wheel Average Speed", color="#ff7f0e", linewidth=1.5)
        if "ground_speed_ms" in ref_df:
            plt.plot(t, ref_df["ground_speed_ms"], label="VBOX GPS True Speed", color="#1f77b4", linewidth=1.5, linestyle="--")
        plt.title(f"[{package.sequence_id}] Wheel Odometry Speed vs VBOX Reference Ground Speed")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speed (m/s)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p3_path = os.path.join(fig_dir, "speed_comparison.png")
        plt.savefig(p3_path, dpi=dpi)
        plt.close()
        generated_files["speed_comparison"] = p3_path

        # ----------------------------------------------------------------------
        # Plot 4: Yaw Rate vs Time (Kinematic Differential vs Gyro)
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4.5))
        if "yaw_rate_rads" in nav_inputs:
            plt.plot(t, np.degrees(nav_inputs["yaw_rate_rads"]), label="CAN Gyro Yaw Rate", color="#9467bd", linewidth=1.5)
        if "wheel_speed_rl_ms" in nav_inputs and "wheel_speed_rr_ms" in nav_inputs:
            kin_yaw = (nav_inputs["wheel_speed_rr_ms"] - nav_inputs["wheel_speed_rl_ms"]) / 1.47
            plt.plot(t, np.degrees(kin_yaw), label="Wheel Differential Yaw Rate", color="#8c564b", linewidth=1.2, linestyle=":")
        plt.title(f"[{package.sequence_id}] Yaw Rate Dynamics (Gyro vs Kinematic Odometry)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Yaw Rate (deg/s)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p4_path = os.path.join(fig_dir, "yaw_rate_comparison.png")
        plt.savefig(p4_path, dpi=dpi)
        plt.close()
        generated_files["yaw_rate_comparison"] = p4_path

        # ----------------------------------------------------------------------
        # Plot 5: Data Quality & Anomaly Flags Timeline
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 4))
        plt.step(t, q_df["quality_flags"], where="post", color="#17becf", label="Quality Flag Code (0=Valid, 1=Missing, 2=Invalid, 3=Suspicious)", linewidth=1.5)
        if "wheel_slip_anomaly_mask" in q_df:
            slip_times = t[q_df["wheel_slip_anomaly_mask"]]
            if len(slip_times) > 0:
                plt.scatter(slip_times, [2.5] * len(slip_times), color="red", marker="x", s=30, label="Tire Slip Anomaly")
        plt.title(f"[{package.sequence_id}] Sample-by-Sample Data Quality Code Timeline")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Quality Code")
        plt.yticks([0, 1, 2, 3], ["VALID (0)", "MISSING (1)", "INVALID (2)", "SUSPICIOUS (3)"])
        plt.ylim(-0.5, 3.5)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p5_path = os.path.join(fig_dir, "quality_timeline.png")
        plt.savefig(p5_path, dpi=dpi)
        plt.close()
        generated_files["quality_timeline"] = p5_path

        return generated_files

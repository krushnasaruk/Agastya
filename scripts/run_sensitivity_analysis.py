
"""
Diagnostic Sensitivity Analysis for Vehicle Track Width (Objective 3 Audit).
Evaluates both Nominal Baseline A (CAN Gyro) and Gyro-Dropout Fallback (Differential Wheel Yaw)
under +/-2% and +/-5% track width variations.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine, ClassicalDeadReckoningConfig
from navigation_engine.evaluation import DeadReckoningEvaluator


def run_track_width_sensitivity(sequence_id: str = "sync_01"):
    seq_dir = os.path.join("data/processed", "sequences", sequence_id)
    nav_df = pd.read_parquet(os.path.join(seq_dir, "navigation_inputs.parquet"))
    ref_df = pd.read_parquet(os.path.join(seq_dir, "reference_trajectory.parquet"))

    init_heading = float(ref_df["heading_rad"].iloc[0])
    init_east = float(ref_df["pos_east_m"].iloc[0])
    init_north = float(ref_df["pos_north_m"].iloc[0])

    nominal_w = 1.47
    perturbations = [
        ("-5% (1.3965m)", nominal_w * 0.95),
        ("-2% (1.4406m)", nominal_w * 0.98),
        ("Nominal (1.4700m)", nominal_w),
        ("+2% (1.4994m)", nominal_w * 1.02),
        ("+5% (1.5435m)", nominal_w * 1.05),
    ]

    print("=" * 105)
    print(f"TRACK WIDTH SENSITIVITY ANALYSIS [Sequence: {sequence_id}]")
    print("=" * 105)

    # 1. Primary Baseline A (CAN Gyro active)
    print("\n[Case 1: Primary Baseline A (CAN Gyro Active)]")
    print(f"{'Perturbation':<20} | {'ATE RMSE (m)':<12} | {'Final Err (m)':<13} | {'Max Err (m)':<12} | {'Drift %':<8} | {'Heading RMSE':<12}")
    print("-" * 105)
    for label, w_val in perturbations:
        cfg = ClassicalDeadReckoningConfig(baseline_type="BASELINE_A", track_width_m=w_val)
        engine = ClassicalDeadReckoningEngine(config=cfg)
        traj = engine.run_sequence(nav_df, initial_heading_rad=init_heading, initial_p_east_m=init_east, initial_p_north_m=init_north)
        m, _, _ = DeadReckoningEvaluator.evaluate(
            traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            ref_df.get("heading_rad", None),
            ref_df.get("ground_speed_ms", None)
        )
        print(f"{label:<20} | {m.ate_rmse_m:<12.4f} | {m.final_position_error_m:<13.4f} | {m.max_position_error_m:<12.4f} | {m.drift_rate_pct:<8.3f} | {m.heading_rmse_deg:<12.3f}°")

    # 2. Case 2: Gyro Dropout Fallback (Differential Wheel Yaw Active)
    print("\n[Case 2: Gyro Dropout Fallback Mode (Differential Wheel Yaw Active)]")
    print(f"{'Perturbation':<20} | {'ATE RMSE (m)':<12} | {'Final Err (m)':<13} | {'Max Err (m)':<12} | {'Drift %':<8} | {'Heading RMSE':<12}")
    print("-" * 105)
    nav_df_no_gyro = nav_df.copy()
    nav_df_no_gyro["yaw_rate_rads"] = np.nan

    for label, w_val in perturbations:
        cfg = ClassicalDeadReckoningConfig(baseline_type="BASELINE_A", track_width_m=w_val)
        engine = ClassicalDeadReckoningEngine(config=cfg)
        traj = engine.run_sequence(nav_df_no_gyro, initial_heading_rad=init_heading, initial_p_east_m=init_east, initial_p_north_m=init_north)
        m, _, _ = DeadReckoningEvaluator.evaluate(
            traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            ref_df.get("heading_rad", None),
            ref_df.get("ground_speed_ms", None)
        )
        print(f"{label:<20} | {m.ate_rmse_m:<12.4f} | {m.final_position_error_m:<13.4f} | {m.max_position_error_m:<12.4f} | {m.drift_rate_pct:<8.3f} | {m.heading_rmse_deg:<12.3f}°")

    print("=" * 105)


if __name__ == "__main__":
    run_track_width_sensitivity("sync_01")

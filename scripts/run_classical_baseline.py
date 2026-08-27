"""
Command-Line Utility for Executing and Benchmarking Classical Dead-Reckoning Baselines.
Evaluates Baseline A, Baseline B, and Baseline C across IO-VNBD sequences and standardized GNSS outages.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine, ClassicalDeadReckoningConfig
from navigation_engine.outage import GNSSOutageSimulator, OutageScenario
from navigation_engine.evaluation import DeadReckoningEvaluator, OutageEvaluationMetrics, NavigationMetrics
from navigation_engine.diagnostics import ClassicalDiagnosticsVisualizer


def run_baseline_benchmarking(
    sequence_id: str = "sync_01",
    processed_dir: str = "data/processed",
    outage_start_sec: float = 20.0
):
    seq_dir = os.path.join(processed_dir, "sequences", sequence_id)
    nav_path = os.path.join(seq_dir, "navigation_inputs.parquet")
    ref_path = os.path.join(seq_dir, "reference_trajectory.parquet")

    if not os.path.exists(nav_path) or not os.path.exists(ref_path):
        raise FileNotFoundError(f"Processed sequence files not found in {seq_dir}. Please run Objective 2 preprocessing first.")

    print("=" * 90)
    print(f"AGASTYA OBJECTIVE 3: CLASSICAL DEAD-RECKONING BENCHMARKING [{sequence_id}]")
    print("=" * 90)

    # 1. Load Standardized Data
    nav_inputs_df = pd.read_parquet(nav_path)
    ref_df = pd.read_parquet(ref_path)

    init_heading = float(ref_df["heading_rad"].iloc[0]) if "heading_rad" in ref_df else 0.0
    init_east = float(ref_df["pos_east_m"].iloc[0]) if "pos_east_m" in ref_df else 0.0
    init_north = float(ref_df["pos_north_m"].iloc[0]) if "pos_north_m" in ref_df else 0.0

    print(f"[Initialization Protocol] Launch Position: ({init_east:.4f}, {init_north:.4f}) m | Launch Heading: {np.degrees(init_heading):.3f}°")
    print(f"[Causal Data] Loaded {len(nav_inputs_df)} onboard sensor epochs (~{ref_df['time_sec'].iloc[-1] - ref_df['time_sec'].iloc[0]:.2f}s duration)")

    # 2. Execute Multiple Classical Baselines
    baselines = ["BASELINE_A", "BASELINE_B", "BASELINE_C"]
    results_summary = {}

    for b_type in baselines:
        print(f"\n[Running Baseline] {b_type}...")
        engine = ClassicalDeadReckoningEngine(baseline_type=b_type)
        traj = engine.run_sequence(
            nav_inputs_df,
            initial_heading_rad=init_heading,
            initial_p_east_m=init_east,
            initial_p_north_m=init_north
        )

        metrics, pos_err, head_err = DeadReckoningEvaluator.evaluate(
            estimated_traj=traj,
            reference_p_east_m=ref_df["pos_east_m"].to_numpy(),
            reference_p_north_m=ref_df["pos_north_m"].to_numpy(),
            reference_heading_rad=ref_df.get("heading_rad", None),
            reference_speed_ms=ref_df.get("ground_speed_ms", None)
        )

        results_summary[b_type] = {
            "metrics": metrics.to_dict(),
            "trajectory": traj,
            "pos_errors": pos_err,
            "head_errors": head_err
        }

        print(f"  + ATE RMSE:              {metrics.ate_rmse_m:.4f} m")
        print(f"  + Final Position Error:  {metrics.final_position_error_m:.4f} m")
        print(f"  + Max Position Error:    {metrics.max_position_error_m:.4f} m")
        print(f"  + Drift Rate:            {metrics.drift_rate_pct:.3f}% of total distance ({metrics.total_trajectory_distance_m:.2f} m)")
        print(f"  + Heading RMSE:          {metrics.heading_rmse_deg:.3f}°")
        print(f"  + Velocity RMSE:         {metrics.velocity_rmse_ms:.5f} m/s (Unrounded)")

    # 3. Standardized GNSS Outage Simulation Experiments (Same Entry Timestamp t=20.0s)
    print(f"\n[Standardized Outage Simulation] Evaluating Monotonic Drift Growth from t = {outage_start_sec:.1f}s...")
    outage_sim = GNSSOutageSimulator(default_durations_sec=[5.0, 10.0, 30.0])
    scenarios = outage_sim.generate_standardized_start_scenarios(
        nav_inputs_df["time_sec"].to_numpy(),
        start_time_sec=outage_start_sec,
        yaw_rate_rads=nav_inputs_df.get("yaw_rate_rads", None).to_numpy()
    )

    primary_traj = results_summary["BASELINE_A"]["trajectory"]
    outage_results = []

    for sc in scenarios:
        acc_drift = DeadReckoningEvaluator.compute_outage_accumulated_drift(
            primary_traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            sc.start_index,
            sc.end_index
        )
        max_drift = DeadReckoningEvaluator.compute_outage_max_drift(
            primary_traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            sc.start_index,
            sc.end_index
        )
        outage_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(
            primary_traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            sc.start_index,
            sc.end_index
        )

        d_e = ref_df["pos_east_m"].to_numpy()
        d_n = ref_df["pos_north_m"].to_numpy()
        dist_outage = float(np.sum(np.sqrt(np.diff(d_e[sc.start_index:sc.end_index+1])**2 + np.diff(d_n[sc.start_index:sc.end_index+1])**2)))
        drift_pct = (acc_drift / max(dist_outage, 1.0)) * 100.0

        out_metrics = OutageEvaluationMetrics(
            outage_id=sc.outage_id,
            duration_sec=sc.duration_sec,
            start_time_sec=sc.start_time_sec,
            end_time_sec=sc.end_time_sec,
            accumulated_drift_m=round(acc_drift, 4),
            max_drift_m=round(max_drift, 4),
            outage_ate_rmse_m=round(outage_ate, 4),
            drift_rate_pct=round(drift_pct, 3),
            distance_traveled_m=round(dist_outage, 2),
            maneuver_type=sc.maneuver_type
        )
        outage_results.append(out_metrics.to_dict())

        print(f"  * Outage {sc.duration_sec:4.1f}s [{sc.start_time_sec:.1f}s -> {sc.end_time_sec:.1f}s | {sc.maneuver_type}]: "
              f"Accumulated Drift = {acc_drift:.4f} m | Max Drift = {max_drift:.4f} m | Outage ATE = {outage_ate:.4f} m | Drift Rate = {drift_pct:.2f}%")

    # 4. Generate Diagnostic Figures
    print("\n[Visualization] Rendering diagnostic performance figures...")
    figs = ClassicalDiagnosticsVisualizer.generate_baseline_plots(
        estimated_traj=primary_traj,
        reference_p_east_m=ref_df["pos_east_m"].to_numpy(),
        reference_p_north_m=ref_df["pos_north_m"].to_numpy(),
        pos_errors_m=results_summary["BASELINE_A"]["pos_errors"],
        head_errors_deg=results_summary["BASELINE_A"]["head_errors"],
        reference_speed_ms=ref_df.get("ground_speed_ms", None).to_numpy(),
        metrics=DeadReckoningEvaluator.evaluate(
            primary_traj,
            ref_df["pos_east_m"].to_numpy(),
            ref_df["pos_north_m"].to_numpy(),
            ref_df.get("heading_rad", None),
            ref_df.get("ground_speed_ms", None)
        )[0],
        output_dir=processed_dir,
        sequence_id=sequence_id
    )

    for k, p in figs.items():
        print(f"  + {k}: {p}")

    # 5. Export JSON Report
    report_path = os.path.join(processed_dir, "reports", f"{sequence_id}_classical_baseline_report.json")
    final_report = {
        "sequence_id": sequence_id,
        "baselines": {
            "BASELINE_A": results_summary["BASELINE_A"]["metrics"],
            "BASELINE_B": results_summary["BASELINE_B"]["metrics"],
            "BASELINE_C": results_summary["BASELINE_C"]["metrics"]
        },
        "gnss_outage_experiments": outage_results
    }
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Report] Exported audited evaluation report to: {report_path}")

    print("\n" + "=" * 90)
    print("OBJECTIVE 3 CLASSICAL DEAD RECKONING BENCHMARKING: SUCCESS")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Objective 3 Classical Dead Reckoning Benchmarking")
    parser.add_argument("--sequence-id", type=str, default="sync_01")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--outage-start-sec", type=float, default=20.0)
    args = parser.parse_args()

    run_baseline_benchmarking(args.sequence_id, args.processed_dir, args.outage_start_sec)

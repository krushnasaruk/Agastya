"""
Command-Line Utility for Executing Objective 4 AI Error Modeling & Formulation Analysis.
Evaluates candidate residual targets, computes statistical diagnostics, performs error decomposition,
and generates diagnostic figures.
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

from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from services.ml.src.targets.residual_targets import ResidualTargetExtractor, ResidualTargetsContainer
from services.ml.src.targets.statistics import ResidualStatisticsAnalyzer
from services.ml.src.features.causal_features import CausalFeatureExtractor, CAUSAL_FEATURE_REGISTRY
from services.ml.src.features.window_builder import CausalWindowBuilder
from services.ml.src.data.alignment import TrajectoryTargetAligner
from services.ml.src.data.splits import DatasetSplitManager
from services.ml.src.analysis.error_decomposition import PhysicalErrorDecomposer
from services.ml.src.analysis.temporal_windows import TemporalWindowAnalyzer
from services.ml.src.correction.interface import AICorrectionSafetyGuard
from services.ml.src.diagnostics.visualizer import Objective4DiagnosticsVisualizer


def run_objective4_analysis(
    sequence_id: str = "sync_01",
    processed_dir: str = "data/processed"
):
    seq_dir = os.path.join(processed_dir, "sequences", sequence_id)
    nav_path = os.path.join(seq_dir, "navigation_inputs.parquet")
    ref_path = os.path.join(seq_dir, "reference_trajectory.parquet")

    if not os.path.exists(nav_path) or not os.path.exists(ref_path):
        raise FileNotFoundError(f"Processed sequence files not found in {seq_dir}.")

    print("=" * 95)
    print(f"AGASTYA OBJECTIVE 4: AI ERROR MODELING & RESIDUAL LEARNING FORMULATION [{sequence_id}]")
    print("=" * 95)

    # 1. Load Data
    nav_df = pd.read_parquet(nav_path)
    ref_df = pd.read_parquet(ref_path)

    # 2. Run Baseline A Classical Engine to obtain baseline estimates
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    init_heading = float(ref_df["heading_rad"].iloc[0])
    init_east = float(ref_df["pos_east_m"].iloc[0])
    init_north = float(ref_df["pos_north_m"].iloc[0])
    traj = engine.run_sequence(nav_df, initial_heading_rad=init_heading, initial_p_east_m=init_east, initial_p_north_m=init_north)
    traj_df = traj.to_dataframe()

    # 3. Deterministic Alignment
    aligned = TrajectoryTargetAligner.align(nav_df, traj_df, ref_df)
    print(f"[Alignment] Successfully aligned {aligned.num_aligned_samples} epochs between classical estimate and offline ground truth.")

    # 4. Extract Candidate Residual Targets
    v_wheel_raw = nav_df["wheel_speed_rear_mean_ms"].to_numpy() if "wheel_speed_rear_mean_ms" in nav_df else (nav_df["wheel_speed_rl_ms"] + nav_df["wheel_speed_rr_ms"]) * 0.5
    targets = ResidualTargetExtractor.extract_all_targets(
        time_sec=aligned.time_sec,
        dt_sec=aligned.dt_sec,
        classical_pos_east_m=aligned.classical_p_east_m,
        classical_pos_north_m=aligned.classical_p_north_m,
        classical_heading_rad=aligned.classical_heading_rad,
        classical_speed_ms=aligned.classical_speed_ms,
        classical_yaw_rate_rads=aligned.classical_yaw_rate_rads,
        raw_wheel_speed_ms=v_wheel_raw,
        ref_pos_east_m=aligned.ref_p_east_m,
        ref_pos_north_m=aligned.ref_p_north_m,
        ref_heading_rad=aligned.ref_heading_rad,
        ref_speed_ms=aligned.ref_speed_ms
    )

    # 5. Extract Causal Features
    causal_features_df = CausalFeatureExtractor.extract_features(nav_df, classical_speed_ms=aligned.classical_speed_ms)
    print(f"[Features] Extracted {len(causal_features_df.columns)} strictly causal features ({len(CAUSAL_FEATURE_REGISTRY)} cataloged in registry).")

    # 6. Statistical Diagnostics on Candidate Targets
    print("\n" + "-" * 95)
    print("STATISTICAL DESCRIPTORS OF CANDIDATE RESIDUAL TARGETS")
    print("-" * 95)
    print(f"{'Target Name':<28} | {'Mean':<10} | {'Std':<10} | {'Median':<10} | {'MAD':<10} | {'Outlier%':<9} | {'ACF(Lag1)':<10}")
    print("-" * 95)

    stats_summary = {}
    target_arrays = {
        "Target A (delta_v_ms)": targets.delta_velocity_ms,
        "Target B1 (delta_yaw_rads)": targets.delta_yaw_rate_rads,
        "Target B2 (delta_heading_rad)": targets.delta_heading_rad,
        "Target C1 (delta_disp_east_m)": targets.delta_disp_east_m,
        "Target C2 (delta_disp_north_m)": targets.delta_disp_north_m,
        "Target D1 (delta_pos_east_m)": targets.delta_pos_east_m,
        "Target D2 (delta_pos_north_m)": targets.delta_pos_north_m,
        "Target E (delta_wheel_ms)": targets.delta_wheel_speed_ms
    }

    for name, arr in target_arrays.items():
        st = ResidualStatisticsAnalyzer.analyze_target(arr, name)
        stats_summary[name] = st.to_dict()
        print(f"{name:<28} | {st.mean:<10.4f} | {st.std:<10.4f} | {st.median:<10.4f} | {st.mad:<10.4f} | {st.outlier_ratio_pct:<9.2f}% | {st.autocorr_lag1:<10.4f}")

    # 7. Causal Feature - Target Correlations
    corrs = ResidualStatisticsAnalyzer.compute_feature_correlations(
        causal_features_df,
        targets.delta_velocity_ms,
        "Target A (delta_v_ms)"
    )
    print("\n[Feature Correlations with Primary Target A (delta_v)]")
    top_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, r in top_corrs:
        print(f"  * {feat:<32}: r = {r:+.4f}")

    # 8. Physical Error Decomposition
    print("\n" + "-" * 95)
    print("PHYSICAL ERROR DECOMPOSITION")
    print("-" * 95)
    decomp = PhysicalErrorDecomposer.decompose(
        time_sec=aligned.time_sec,
        dt_sec=aligned.dt_sec,
        v_wheel_rear_ms=v_wheel_raw,
        yaw_rate_can_rads=nav_df["yaw_rate_rads"].to_numpy(),
        accel_x_ms2=nav_df["accel_x_ms2"].to_numpy(),
        v_ref_ms=aligned.ref_speed_ms,
        v_classical_ms=aligned.classical_speed_ms,
        heading_classical_rad=aligned.classical_heading_rad,
        heading_ref_rad=aligned.ref_heading_rad
    )
    for f in decomp.findings_summary:
        print(f"  + {f}")

    # 9. Temporal Window Analysis
    print("\n" + "-" * 95)
    print("TEMPORAL WINDOW TRADE-OFF ANALYSIS")
    print("-" * 95)
    window_reports = TemporalWindowAnalyzer.evaluate_all_windows(total_sequence_samples=len(nav_df))
    for wr in window_reports:
        print(f"  * Window {wr.duration_sec:.1f}s ({wr.num_epochs} epochs | Latency: {wr.latency_ms:.0f}ms): {wr.recommendation_status} — {wr.rationale}")

    # 10. Dataset Split Validation
    split_cfg = DatasetSplitManager.get_default_split()
    DatasetSplitManager.validate_no_leakage(split_cfg)
    print(f"\n[Dataset Split] Validated zero-leakage trajectory partitioning: Train={split_cfg.train_sequences}, Val={split_cfg.val_sequences}, Test={split_cfg.test_sequences}")

    # 11. Generate Diagnostic Figures
    print("\n[Visualization] Rendering Objective 4 diagnostic figures...")
    figs = Objective4DiagnosticsVisualizer.generate_all_plots(
        targets=targets,
        classical_speed_ms=aligned.classical_speed_ms,
        classical_yaw_rate_rads=aligned.classical_yaw_rate_rads,
        accel_x_ms2=nav_df["accel_x_ms2"].to_numpy(),
        output_dir=processed_dir,
        sequence_id=sequence_id
    )
    for k, p in figs.items():
        print(f"  + {k}: {p}")

    # 12. Export JSON Report
    report_path = os.path.join(processed_dir, "reports", f"{sequence_id}_objective4_formulation_report.json")
    final_report = {
        "sequence_id": sequence_id,
        "target_statistics": stats_summary,
        "feature_correlations_target_a": corrs,
        "physical_error_decomposition": decomp.to_dict(),
        "temporal_window_analysis": [wr.to_dict() for wr in window_reports],
        "dataset_split_configuration": split_cfg.to_dict()
    }
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Report] Exported full formulation report to: {report_path}")

    print("\n" + "=" * 95)
    print("OBJECTIVE 4 RESIDUAL LEARNING FORMULATION: SUCCESS")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Objective 4 Formulation Analysis")
    parser.add_argument("--sequence-id", type=str, default="sync_01")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    run_objective4_analysis(args.sequence_id, args.processed_dir)

"""
Command-Line Utility for Training, Validating, and Evaluating Causal Residual Models (Objective 5).
Executes strictly controlled training on sync_01, early stopping on v_standalone_03,
and held-out evaluation on sync_02.
"""

import os
import sys
import json
import argparse
import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from navigation_engine.evaluation import DeadReckoningEvaluator
from services.ml.src.targets.residual_targets import ResidualTargetExtractor
from services.ml.src.features.causal_features import CausalFeatureExtractor
from services.ml.src.data.alignment import TrajectoryTargetAligner

from ai_residual.feature_registry import CANONICAL_FEATURE_NAMES, validate_feature_matrix_columns
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from ai_residual.dataset import CausalWindowDataset
from ai_residual.model import CausalResidualGRU
from ai_residual.safety import SafetyGuard
from ai_residual.trainer import ResidualModelTrainer, set_seed
from ai_residual.evaluator import ResidualEvaluator
from ai_residual.rollout import AIRolloutEngine
from ai_residual.outage_eval import OutageComparator
from ai_residual.ablations import AblationRunner
from ai_residual.diagnostics import Objective5Visualizer


def prepare_sequence_data(
    sequence_id: str,
    processed_dir: str = "data/processed"
):
    seq_dir = os.path.join(processed_dir, "sequences", sequence_id)
    nav_path = os.path.join(seq_dir, "navigation_inputs.parquet")
    ref_path = os.path.join(seq_dir, "reference_trajectory.parquet")

    nav_df = pd.read_parquet(nav_path)
    ref_df = pd.read_parquet(ref_path)

    # Run Baseline A to get classical unassisted estimates
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    init_heading = float(ref_df["heading_rad"].iloc[0]) if "heading_rad" in ref_df else 0.0
    init_east = float(ref_df["pos_east_m"].iloc[0]) if "pos_east_m" in ref_df else 0.0
    init_north = float(ref_df["pos_north_m"].iloc[0]) if "pos_north_m" in ref_df else 0.0

    traj = engine.run_sequence(
        nav_df,
        initial_heading_rad=init_heading,
        initial_p_east_m=init_east,
        initial_p_north_m=init_north
    )
    traj_df = traj.to_dataframe()

    aligned = TrajectoryTargetAligner.align(nav_df, traj_df, ref_df)
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

    causal_feats_df = CausalFeatureExtractor.extract_features(nav_df, classical_speed_ms=aligned.classical_speed_ms)
    causal_feats_df = causal_feats_df[CANONICAL_FEATURE_NAMES]
    validate_feature_matrix_columns(list(causal_feats_df.columns))

    targets_matrix = np.column_stack([targets.delta_velocity_ms, targets.delta_yaw_rate_rads])

    return {
        "sequence_id": sequence_id,
        "nav_df": nav_df,
        "ref_df": ref_df,
        "aligned": aligned,
        "classical_traj": traj,
        "causal_feats_df": causal_feats_df,
        "targets_matrix": targets_matrix,
        "timestamps": aligned.time_sec
    }


def run_training_pipeline(
    train_seq: str = "sync_01",
    val_seq: str = "v_standalone_03",
    test_seq: str = "sync_02",
    artifacts_dir: str = "artifacts/objective5",
    processed_dir: str = "data/processed",
    max_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 15,
    seed: int = 42
):
    print("=" * 105)
    print("AGASTYA OBJECTIVE 5: CAUSAL RESIDUAL LEARNING MODEL TRAINING & HELD-OUT VALIDATION")
    print("=" * 105)
    set_seed(seed)
    os.makedirs(artifacts_dir, exist_ok=True)
    fig_dir = os.path.join(artifacts_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware & Environment] Device: {device} | Seed: {seed} | PyTorch: {torch.__version__} | NumPy: {np.__version__}")

    # 1. Load & Prepare Sequences
    print("\n[Data Ingestion] Preparing train, validation, and held-out test sequences...")
    train_data = prepare_sequence_data(train_seq, processed_dir)
    val_data = prepare_sequence_data(val_seq, processed_dir)
    test_data = prepare_sequence_data(test_seq, processed_dir)

    print(f"  + Training Set:   '{train_seq}' -> {len(train_data['nav_df'])} samples")
    print(f"  + Validation Set: '{val_seq}' -> {len(val_data['nav_df'])} samples")
    print(f"  + Held-Out Test:  '{test_seq}' -> {len(test_data['nav_df'])} samples (HELD-OUT UNSEEN)")

    # 2. Fit Scalers STRICTLY on Training Sequence (sync_01)
    print("\n[Scaler Fitting] Fitting feature and target scalers strictly on training sequence...")
    feat_scaler = TrainOnlyScaler().fit(train_data["causal_feats_df"], sequence_id=train_seq)
    target_scaler = TargetScaler().fit(train_data["targets_matrix"], sequence_id=train_seq)

    feat_scaler_path = os.path.join(artifacts_dir, "feature_scaler.json")
    target_scaler_path = os.path.join(artifacts_dir, "target_scaler.json")
    feat_scaler.save_json(feat_scaler_path)
    target_scaler.save_json(target_scaler_path)
    print(f"  + Serialized feature scaler: {feat_scaler_path}")
    print(f"  + Serialized target scaler:  {target_scaler_path}")

    # 3. Standardize Features and Targets
    x_train_norm = feat_scaler.transform(train_data["causal_feats_df"])
    y_train_norm = target_scaler.transform(train_data["targets_matrix"])

    x_val_norm = feat_scaler.transform(val_data["causal_feats_df"])
    y_val_norm = target_scaler.transform(val_data["targets_matrix"])

    x_test_norm = feat_scaler.transform(test_data["causal_feats_df"])
    y_test_norm = target_scaler.transform(test_data["targets_matrix"])

    # 4. Construct Causal Window Datasets (W = 10)
    w_size = 10
    train_dataset = CausalWindowDataset(x_train_norm, y_train_norm, window_size=w_size, timestamps_sec=train_data["timestamps"], sequence_id=train_seq)
    val_dataset = CausalWindowDataset(x_val_norm, y_val_norm, window_size=w_size, timestamps_sec=val_data["timestamps"], sequence_id=val_seq)
    test_dataset = CausalWindowDataset(x_test_norm, y_test_norm, window_size=w_size, timestamps_sec=test_data["timestamps"], sequence_id=test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"  + Causal Windows [W={w_size}]: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 5. Initialize Model
    model = CausalResidualGRU(input_dim=16, hidden_dim=64, mlp_dim=32, output_dim=2, num_gru_layers=1)
    model_config = model.get_model_config()
    model_config["window_size"] = w_size
    model_config["training_sequence"] = train_seq
    model_config["validation_sequence"] = val_seq
    model_config["test_sequence"] = test_seq
    model_config["seed"] = seed

    config_path = os.path.join(artifacts_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n[Model Initialized] {model_config['model_type']} ({model_config['total_parameters']} parameters)")

    # 6. Train with Early Stopping on Validation Loss
    print("\n[Training Execution] Starting deterministic multi-task training loop...")
    trainer = ResidualModelTrainer(model=model, learning_rate=learning_rate, device=device)
    train_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=artifacts_dir,
        verbose=True
    )
    print(f"\n[Training Complete] Best Epoch: {train_result['best_epoch']} | Best Val Loss: {train_result['best_val_loss']:.6f}")

    # 7. Evaluate Physical Residual Metrics on Held-Out Test Set (sync_02)
    print("\n" + "-" * 105)
    print(f"HELD-OUT RESIDUAL PREDICTION METRICS [{test_seq}]")
    print("-" * 105)
    res_metrics, y_true_phys, y_pred_phys, test_timestamps = ResidualEvaluator.evaluate_dataset(
        model=model,
        dataloader=test_loader,
        target_scaler=target_scaler,
        device=device
    )

    test_metrics_export = {}
    for name, met in res_metrics.items():
        test_metrics_export[name] = met.to_dict()
        print(f"{name:<25} | MAE: {met.mae:.5f} | RMSE: {met.rmse:.5f} | Bias: {met.bias:+.5f} | R²: {met.r2_score:+.4f} | r: {met.pearson_correlation:+.4f} | Zero-RMSE: {met.trivial_zero_rmse:.5f}")

    test_metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_export, f, indent=2)

    # 8. Closed-Loop Navigation Rollout on Held-Out Test Sequence (sync_02)
    print("\n" + "-" * 105)
    print(f"CLOSED-LOOP NAVIGATION ROLLOUT & BENCHMARK COMPARISON [{test_seq}]")
    print("-" * 105)
    rollout_engine = AIRolloutEngine(
        model=model,
        feature_scaler=feat_scaler,
        target_scaler=target_scaler,
        safety_guard=SafetyGuard(),
        window_size=w_size,
        device=device
    )

    init_h = float(test_data["ref_df"]["heading_rad"].iloc[0]) if "heading_rad" in test_data["ref_df"] else 0.0
    init_e = float(test_data["ref_df"]["pos_east_m"].iloc[0]) if "pos_east_m" in test_data["ref_df"] else 0.0
    init_n = float(test_data["ref_df"]["pos_north_m"].iloc[0]) if "pos_north_m" in test_data["ref_df"] else 0.0

    ai_traj = rollout_engine.run_rollout(
        test_data["nav_df"],
        test_data["causal_feats_df"],
        initial_p_east_m=init_e,
        initial_p_north_m=init_n,
        initial_heading_rad=init_h
    )

    ref_e = test_data["ref_df"]["pos_east_m"].to_numpy()
    ref_n = test_data["ref_df"]["pos_north_m"].to_numpy()
    ref_h = test_data["ref_df"].get("heading_rad", None)
    ref_v = test_data["ref_df"].get("ground_speed_ms", None)

    class_metrics, _, _ = DeadReckoningEvaluator.evaluate(test_data["classical_traj"], ref_e, ref_n, ref_h, ref_v)
    ai_metrics, _, _ = DeadReckoningEvaluator.evaluate(ai_traj, ref_e, ref_n, ref_h, ref_v)

    print(f"{'Metric':<30} | {'Classical Baseline A':<22} | {'AI-Corrected Baseline':<22} | {'Improvement %':<15}")
    print("-" * 105)
    ate_imp = ((class_metrics.ate_rmse_m - ai_metrics.ate_rmse_m) / class_metrics.ate_rmse_m) * 100.0
    fin_imp = ((class_metrics.final_position_error_m - ai_metrics.final_position_error_m) / class_metrics.final_position_error_m) * 100.0
    max_imp = ((class_metrics.max_position_error_m - ai_metrics.max_position_error_m) / class_metrics.max_position_error_m) * 100.0
    v_imp = ((class_metrics.velocity_rmse_ms - ai_metrics.velocity_rmse_ms) / class_metrics.velocity_rmse_ms) * 100.0
    h_imp = ((class_metrics.heading_rmse_deg - ai_metrics.heading_rmse_deg) / class_metrics.heading_rmse_deg) * 100.0

    print(f"{'ATE RMSE (m)':<30} | {class_metrics.ate_rmse_m:<22.4f} | {ai_metrics.ate_rmse_m:<22.4f} | {ate_imp:+14.2f}%")
    print(f"{'Final Position Error (m)':<30} | {class_metrics.final_position_error_m:<22.4f} | {ai_metrics.final_position_error_m:<22.4f} | {fin_imp:+14.2f}%")
    print(f"{'Max Position Error (m)':<30} | {class_metrics.max_position_error_m:<22.4f} | {ai_metrics.max_position_error_m:<22.4f} | {max_imp:+14.2f}%")
    print(f"{'Drift Rate (% distance)':<30} | {class_metrics.drift_rate_pct:<22.3f}% | {ai_metrics.drift_rate_pct:<22.3f}% | {((class_metrics.drift_rate_pct - ai_metrics.drift_rate_pct)/class_metrics.drift_rate_pct)*100:+14.2f}%")
    print(f"{'Heading RMSE (deg)':<30} | {class_metrics.heading_rmse_deg:<22.4f}° | {ai_metrics.heading_rmse_deg:<22.4f}° | {h_imp:+14.2f}%")
    print(f"{'Velocity RMSE (m/s)':<30} | {class_metrics.velocity_rmse_ms:<22.5f} | {ai_metrics.velocity_rmse_ms:<22.5f} | {v_imp:+14.2f}%")

    # 9. Standardized GNSS Outage Comparison (t = 20.0s)
    print("\n" + "-" * 105)
    print("STANDARDIZED GNSS OUTAGE COMPARISON (Entry t = 20.0s)")
    print("-" * 105)
    outage_results = OutageComparator.evaluate_outages(
        classical_traj=test_data["classical_traj"],
        ai_traj=ai_traj,
        ref_east_m=ref_e,
        ref_north_m=ref_n,
        start_time_sec=20.0,
        durations=[5.0, 10.0, 30.0]
    )
    for out_rec in outage_results:
        d = out_rec["duration_sec"]
        c_ate = out_rec["classical"]["outage_ate_rmse_m"]
        ai_ate = out_rec["ai_corrected"]["outage_ate_rmse_m"]
        imp = out_rec["ate_improvement_pct"]
        print(f"  * Outage {d:4.1f}s ({out_rec['maneuver_type']}) | Classical ATE = {c_ate:.4f}m | AI-Corrected ATE = {ai_ate:.4f}m | ATE Improvement = {imp:+.2f}%")

    # 10. Controlled Scientific Ablations
    print("\n" + "-" * 105)
    print("SCIENTIFIC ABLATION STUDY")
    print("-" * 105)
    abl_results = AblationRunner.run_ablations(
        model=model,
        feature_scaler=feat_scaler,
        target_scaler=target_scaler,
        nav_inputs_df=test_data["nav_df"],
        causal_features_df=test_data["causal_feats_df"],
        ref_df=test_data["ref_df"],
        sequence_id=test_seq,
        device=device
    )

    abl_export = {}
    for name, res in abl_results.items():
        m = res["metrics"]
        abl_export[name] = m
        print(f"  * {name:<22} | ATE RMSE: {m['ate_rmse_m']:.4f} m | Final Err: {m['final_position_error_m']:.4f} m | Heading RMSE: {m['heading_rmse_deg']:.3f}° | Velocity RMSE: {m['velocity_rmse_ms']:.4f} m/s")

    abl_path = os.path.join(artifacts_dir, "ablation_metrics.json")
    with open(abl_path, "w") as f:
        json.dump(abl_export, f, indent=2)

    # 11. Render All 12 Diagnostic Figures
    print("\n[Visualization] Rendering all 12 Objective 5 diagnostic figures...")
    figs = Objective5Visualizer.generate_all_plots(
        training_history=train_result,
        y_true_phys=y_true_phys,
        y_pred_phys=y_pred_phys,
        timestamps_sec=test_timestamps,
        classical_traj=test_data["classical_traj"],
        ai_traj=ai_traj,
        ref_east_m=ref_e,
        ref_north_m=ref_n,
        ref_speed_ms=ref_v.to_numpy() if ref_v is not None else None,
        ref_heading_rad=ref_h.to_numpy() if ref_h is not None else None,
        outage_records=outage_results,
        ablation_records=abl_results,
        output_dir=fig_dir,
        sequence_id=test_seq
    )
    for k, p in figs.items():
        print(f"  + {k}: {p}")

    # 12. Export Objective 5 Manifest
    manifest = {
        "project": "AGASTYA",
        "objective": 5,
        "model_name": model_config["model_type"],
        "feature_count": 16,
        "feature_order": CANONICAL_FEATURE_NAMES,
        "window_size": w_size,
        "training_sequence": train_seq,
        "validation_sequence": val_seq,
        "test_sequence": test_seq,
        "seed": seed,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": train_result["best_epoch"],
        "final_training_loss": train_result["history"]["train_loss"][-1],
        "final_validation_loss": train_result["best_val_loss"],
        "residual_test_metrics": test_metrics_export,
        "classical_navigation_metrics": class_metrics.to_dict(),
        "ai_navigation_metrics": ai_metrics.to_dict(),
        "gnss_outage_comparison": outage_results,
        "ablation_metrics": abl_export,
        "timestamp": datetime.datetime.now().isoformat()
    }
    manifest_path = os.path.join(artifacts_dir, "objective5_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Manifest] Exported comprehensive experiment manifest to: {manifest_path}")

    # 13. Final Reproducibility Summary
    print("\n" + "=" * 105)
    print("OBJECTIVE 5 TRAINING COMPLETE")
    print("=" * 105)
    print(f"Model:                    {model_config['model_type']}")
    print(f"Training trajectory:      {train_seq}")
    print(f"Validation trajectory:    {val_seq}")
    print(f"Held-out test trajectory: {test_seq}")
    print(f"Best epoch:               {train_result['best_epoch']}")
    print(f"Validation loss:          {train_result['best_val_loss']:.6f}")
    print(f"Classical ATE RMSE:       {class_metrics.ate_rmse_m:.4f} m")
    print(f"AI ATE RMSE:              {ai_metrics.ate_rmse_m:.4f} m ({ate_imp:+.2f}%)")
    print(f"Classical final error:    {class_metrics.final_position_error_m:.4f} m")
    print(f"AI final error:           {ai_metrics.final_position_error_m:.4f} m ({fin_imp:+.2f}%)")
    print(f"Classical drift rate:     {class_metrics.drift_rate_pct:.3f}%")
    print(f"AI drift rate:            {ai_metrics.drift_rate_pct:.3f}%")
    print(f"Velocity residual RMSE:   {res_metrics['delta_velocity_ms'].rmse:.5f} m/s (Zero-RMSE: {res_metrics['delta_velocity_ms'].trivial_zero_rmse:.5f})")
    print(f"Yaw residual RMSE:        {res_metrics['delta_yaw_rate_rads'].rmse:.5f} rad/s")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Objective 5 Model Training and Evaluation")
    parser.add_argument("--train-seq", type=str, default="sync_01")
    parser.add_argument("--val-seq", type=str, default="v_standalone_03")
    parser.add_argument("--test-seq", type=str, default="sync_02")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/objective5")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training_pipeline(
        train_seq=args.train_seq,
        val_seq=args.val_seq,
        test_seq=args.test_seq,
        artifacts_dir=args.artifacts_dir,
        processed_dir=args.processed_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed
    )

"""
Standalone Real-Time Sequence Replay CLI for Objective 7.
Replays a recorded trajectory through the RealtimeNavigationEngine at 10 Hz.
"""

import os
import sys
import argparse
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from scripts.train_residual_model import prepare_sequence_data
from objective6.distribution_monitor import TrainingDistributionMonitor
from objective7.realtime_engine import RealtimeNavigationEngine
from objective7.replay_engine import RealtimeReplayEngine


def main():
    parser = argparse.ArgumentParser(description="Objective 7 Sequence Replay")
    parser.add_argument("--sequence", type=str, default="sync_02")
    parser.add_argument("--obj5-dir", type=str, default="artifacts/objective5")
    parser.add_argument("--obj6-dir", type=str, default="artifacts/objective6")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    model = CausalResidualGRU(input_dim=16, hidden_dim=64, mlp_dim=32, output_dim=2)
    model.load_state_dict(torch.load(os.path.join(args.obj5_dir, "best_model.pt"), map_location=torch.device("cpu")))
    model.eval()

    feat_scaler = TrainOnlyScaler.load_json(os.path.join(args.obj5_dir, "feature_scaler.json"))
    target_scaler = TargetScaler.load_json(os.path.join(args.obj5_dir, "target_scaler.json"))
    dist_monitor = TrainingDistributionMonitor.load_json(os.path.join(args.obj6_dir, "feature_distribution.json"))

    engine = RealtimeNavigationEngine(model, feat_scaler, target_scaler, dist_monitor)
    data = prepare_sequence_data(args.sequence, args.data_dir)

    init_h = float(data["ref_df"]["heading_rad"].iloc[0]) if "heading_rad" in data["ref_df"] else 0.0
    init_e = float(data["ref_df"]["pos_east_m"].iloc[0]) if "pos_east_m" in data["ref_df"] else 0.0
    init_n = float(data["ref_df"]["pos_north_m"].iloc[0]) if "pos_north_m" in data["ref_df"] else 0.0

    print(f"Replaying sequence '{args.sequence}' through RealtimeNavigationEngine...")
    res = RealtimeReplayEngine.run_replay(engine, data["nav_df"], data["ref_df"], init_e, init_n, init_h)

    print("=" * 60)
    print(f"Replay Results for '{args.sequence}':")
    print(f"  + ATE RMSE:              {res.metrics.ate_rmse_m:.4f} m")
    print(f"  + Final Position Error:  {res.metrics.final_position_error_m:.4f} m")
    print(f"  + Heading RMSE:          {res.metrics.heading_rmse_deg:.4f} deg")
    print(f"  + AI Application Rate:   {res.application_rate_pct:.1f}%")
    print(f"  + Fallback Rate:         {res.fallback_rate_pct:.1f}%")
    print(f"  + Mean Latency:          {res.latency_summary['total_latency']['mean_ms']:.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()

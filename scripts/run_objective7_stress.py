"""
Long-Duration Stress Testing and Fault-Injection CLI for Objective 7.
"""

import os
import sys
import argparse
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from objective6.distribution_monitor import TrainingDistributionMonitor
from objective7.realtime_engine import RealtimeNavigationEngine
from objective7.numerical_stability import NumericalStabilityMonitor


def main():
    parser = argparse.ArgumentParser(description="Objective 7 Stress Test")
    parser.add_argument("--epochs", type=int, default=5000)
    args = parser.parse_args()

    model = CausalResidualGRU(input_dim=16, hidden_dim=64, mlp_dim=32, output_dim=2)
    model.eval()
    feat_scaler = TrainOnlyScaler.load_json("artifacts/objective5/feature_scaler.json")
    target_scaler = TargetScaler.load_json("artifacts/objective5/target_scaler.json")
    dist_monitor = TrainingDistributionMonitor.load_json("artifacts/objective6/feature_distribution.json")

    engine = RealtimeNavigationEngine(model, feat_scaler, target_scaler, dist_monitor)
    stab = NumericalStabilityMonitor()

    print(f"Running long-duration continuous stress test ({args.epochs} epochs)...")
    engine.initialize()
    for i in range(args.epochs):
        st = engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.05, 0.005)
        stab.check_state(st.p_east_m, st.p_north_m, st.heading_rad, st.forward_speed_ms)

    res = stab.get_summary()
    print("=" * 60)
    print("NUMERICAL STABILITY SUMMARY:")
    print(f"  + NaNs:                  {res['nan_occurrences']}")
    print(f"  + Infs:                  {res['inf_occurrences']}")
    print(f"  + Heading Violations:    {res['heading_wrapping_violations']}")
    print(f"  + State Explosions:      {res['state_explosion_events']}")
    print(f"  + Status:                {res['stability_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

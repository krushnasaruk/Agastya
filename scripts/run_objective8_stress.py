#!/usr/bin/env python3
"""
Long-Duration Stress Testing CLI Script for Objective 8.
Usage:
    python scripts/run_objective8_stress.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from objective8.hardware_ready_engine import HardwareReadyNavigationEngine
from objective8.long_duration_runner import LongDurationRunner


def main():
    model = CausalResidualGRU()
    model.load_state_dict(torch.load("artifacts/objective5/best_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    f_scaler = TrainOnlyScaler.load("artifacts/objective5/feature_scaler.json")
    t_scaler = TargetScaler.load("artifacts/objective5/target_scaler.json")

    engine = HardwareReadyNavigationEngine(
        model=model,
        feature_scaler=f_scaler,
        target_scaler=t_scaler,
        deployment_mode="MODE_B_INT8"
    )

    print("Running 10,000 continuous navigation epochs stress test...")
    res = LongDurationRunner.run_stress_test(engine, num_epochs=10000)

    print("=" * 70)
    print("OBJECTIVE 8 LONG-DURATION STRESS SUMMARY (10,000 EPOCHS)")
    print("=" * 70)
    print(f"Total Epochs:              {res['total_stress_epochs']:,}")
    print(f"Drive Duration:            {res['simulated_drive_duration_sec']:.1f} s ({res['simulated_drive_duration_sec']/60.0:.1f} min)")
    print(f"Peak Memory:               {res['resource_summary']['memory_profile']['peak_rss_mb']:.2f} MB")
    print(f"Memory Bounded:            {res['resource_summary']['memory_profile']['is_bounded']}")
    print(f"NaN Occurrences:           {res['stability_summary']['nan_occurrences']}")
    print(f"Inf Occurrences:           {res['stability_summary']['inf_occurrences']}")
    print(f"Status:                    {res['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

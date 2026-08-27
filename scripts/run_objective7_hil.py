"""
Software-HIL Validation Runner for Objective 7.
Simulates real-time 10-Hz pacing, measures inter-frame timing jitter, and benchmarks dropped epochs.
"""

import os
import sys
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from objective7.hil_runner import HILRunner


def main():
    parser = argparse.ArgumentParser(description="Software-HIL Benchmark")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Executing SOFTWARE-HIL streaming benchmark ({args.epochs} epochs @ {args.hz} Hz)...")
    runner = HILRunner(target_frequency_hz=args.hz)
    res = runner.run_stream_benchmark(num_epochs=args.epochs)

    print("=" * 60)
    print("SOFTWARE-HIL RESULTS:")
    print(f"  + Mean Jitter:       {res['mean_jitter_ms']:.3f} ms")
    print(f"  + p95 Jitter:        {res['p95_jitter_ms']:.3f} ms")
    print(f"  + p99 Jitter:        {res['p99_jitter_ms']:.3f} ms")
    print(f"  + Max Jitter:        {res['max_jitter_ms']:.3f} ms")
    print(f"  + Dropped Epochs:    {res['dropped_epochs']}")
    print(f"  + Status:            {res['software_hil_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

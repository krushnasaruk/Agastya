#!/usr/bin/env python3
"""
Software-HIL Benchmark CLI Script for Objective 8.
Usage:
    python scripts/run_objective8_hil.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from objective8.hil_runner import HILRunner


def main():
    runner = HILRunner(target_frequency_hz=10.0)
    print("Executing 10-Hz Software-HIL continuous stream benchmark...")
    res = runner.run_stream_benchmark(num_epochs=100)

    print("=" * 70)
    print("OBJECTIVE 8 SOFTWARE-HIL STREAM BENCHMARK")
    print("=" * 70)
    print(f"Target Frequency:          {res['target_frequency_hz']} Hz ({res['nominal_period_ms']:.1f} ms period)")
    print(f"Total Streamed Epochs:     {res['total_streamed_epochs']}")
    print(f"Mean Jitter:               {res['mean_jitter_ms']:.3f} ms")
    print(f"p95 Jitter:                {res['p95_jitter_ms']:.3f} ms")
    print(f"p99 Jitter:                {res['p99_jitter_ms']:.3f} ms")
    print(f"Dropped Frames:            {res['dropped_packets_count']}")
    print(f"Hardware Status:           {res['hardware_validation_label']}")
    print(f"Status:                    {res['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

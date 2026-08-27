#!/usr/bin/env python3
"""
AGASTYA Command-Line Interface (CLI).
Provides developer tools for system diagnostics, latency benchmarks,
trajectory error evaluation, and automated unit test execution.
"""

import sys
import os
import time
import argparse
import platform
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def cmd_system_info():
    """Display runtime environment, hardware platform, and project modules."""
    print("=" * 60)
    print("  PROJECT AGASTYA - SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print(f"  OS:              {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python:          {platform.python_version()} ({sys.executable})")
    print(f"  Workspace:       {BASE_DIR}")

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        print(f"  PyTorch:         {torch.__version__} (CUDA: {cuda_avail} - {dev_name})")
    except ImportError:
        print("  PyTorch:         Not installed")

    try:
        import numpy
        print(f"  NumPy:           {numpy.__version__}")
    except ImportError:
        pass

    try:
        import scipy
        print(f"  SciPy:           {scipy.__version__}")
    except ImportError:
        pass

    print("=" * 60)


def cmd_benchmark(num_iterations: int = 1000):
    """Benchmark Strapdown Mechanization and ES-EKF latency."""
    print("=" * 60)
    print(f"  AGASTYA NAVIGATION ENGINE BENCHMARK ({num_iterations} steps)")
    print("=" * 60)

    try:
        from navigation_engine.estimation.kalman import ErrorStateKalmanFilter
        from navigation_engine.estimation.state import NavigationState
    except ImportError:
        from src.estimation.kalman import ErrorStateKalmanFilter
        from src.estimation.state import NavigationState

    ekf = ErrorStateKalmanFilter()
    state = NavigationState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(15) * 0.1
    )

    accel = np.array([0.0, 0.0, -9.80665])
    gyro = np.array([0.01, -0.01, 0.02])

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        state = ekf.predict(state, accel, gyro, dt=0.01)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    lat_arr = np.array(latencies)
    mean_lat = np.mean(lat_arr)
    p95_lat = np.percentile(lat_arr, 95)
    p99_lat = np.percentile(lat_arr, 99)
    max_lat = np.max(lat_arr)
    fps = 1000.0 / mean_lat if mean_lat > 0 else float("inf")

    print(f"  Mean Latency:    {mean_lat:.4f} ms")
    print(f"  95th Percentile: {p95_lat:.4f} ms")
    print(f"  99th Percentile: {p99_lat:.4f} ms")
    print(f"  Max Latency:     {max_lat:.4f} ms")
    print(f"  Throughput:      {fps:.1f} Hz (Budget: 100 Hz)")
    print(f"  Status:          {'PASS (Real-time capable)' if mean_lat < 1.0 else 'WARN'}")
    print("=" * 60)


def cmd_zupt_check():
    """Verify ZUPT detector on synthetic stationary and dynamic periods."""
    print("=" * 60)
    print("  AGASTYA ZERO-VELOCITY DETECTOR (ZUPT) VALIDATION")
    print("=" * 60)

    try:
        from navigation_engine.estimation.zupt import ZeroVelocityDetector
    except ImportError:
        from src.estimation.zupt import ZeroVelocityDetector

    detector = ZeroVelocityDetector()
    
    # 1. Stationary test
    for _ in range(12):
        acc = np.array([0.0, 0.0, -9.80665]) + np.random.normal(0, 0.005, 3)
        gyr = np.random.normal(0, 0.002, 3)
        detector.add_reading(acc, gyr)

    stat, conf = detector.is_stationary()
    print(f"  Stationary Phase Detection: {'PASS' if stat else 'FAIL'} (Confidence: {conf:.2%})")

    # 2. Dynamic test
    for _ in range(12):
        acc = np.array([3.5, 1.2, -9.80665]) + np.random.normal(0, 0.2, 3)
        gyr = np.array([0.4, -0.3, 0.5])
        detector.add_reading(acc, gyr)

    stat_dyn, conf_dyn = detector.is_stationary()
    print(f"  Dynamic Phase Detection:    {'PASS' if not stat_dyn else 'FAIL'} (Confidence: {conf_dyn:.2%})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AGASTYA Navigation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="Show system environment and diagnostic info")
    
    bench_parser = subparsers.add_parser("benchmark", help="Run EKF latency benchmark")
    bench_parser.add_argument("--steps", type=int, default=1000, help="Number of benchmark steps")

    subparsers.add_parser("zupt", help="Validate Zero-Velocity detector")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        cmd_system_info()
    elif args.command == "benchmark":
        cmd_benchmark(num_iterations=args.steps)
    elif args.command == "zupt":
        cmd_zupt_check()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

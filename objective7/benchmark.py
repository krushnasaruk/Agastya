"""
Master Benchmarking Engine for Objective 7.
Evaluates Cold/Warm latency, CPU inference, Throughput scaling (10-100Hz), and Memory stability over 1,000+ iterations.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from .realtime_engine import RealtimeNavigationEngine
from .memory_monitor import MemoryMonitor


class Objective7BenchmarkSuite:
    """
    Executes comprehensive latency, throughput, and memory stress tests.
    """
    @classmethod
    def run_cold_vs_warm_benchmark(
        cls,
        engine: RealtimeNavigationEngine,
        num_warm_epochs: int = 1000
    ) -> Dict[str, Any]:
        """
        Measure first epoch (cold start) vs sustained warm execution.
        """
        engine.reset()
        # Cold start epoch
        t0 = time.perf_counter()
        engine.process_sensor_sample(0.0, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
        cold_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Warm epochs
        for i in range(1, num_warm_epochs + 1):
            engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)

        warm_stats = engine.latency_monitor.get_summary_statistics()
        return {
            "cold_start_latency_ms": round(cold_latency_ms, 4),
            "warm_execution_summary": warm_stats
        }

    @classmethod
    def run_throughput_load_test(
        cls,
        engine: RealtimeNavigationEngine,
        target_frequencies: List[float] = [10.0, 20.0, 50.0, 100.0],
        samples_per_freq: int = 500
    ) -> Dict[str, Any]:
        """
        Measure maximum sustained throughput and CPU utilization scaling.
        """
        throughput_records = {}

        for hz in target_frequencies:
            engine.reset()
            t_start = time.perf_counter()
            for i in range(samples_per_freq):
                engine.process_sensor_sample(i * 0.1, 0.1, 12.0, 12.0, 12.0, 12.0, 0.1, 0.01)
            t_elapsed = time.perf_counter() - t_start

            actual_hz = samples_per_freq / max(t_elapsed, 1e-6)
            lat_sum = engine.latency_monitor.get_summary_statistics()

            throughput_records[f"{int(hz)}Hz_target"] = {
                "target_hz": hz,
                "achieved_throughput_hz": round(actual_hz, 2),
                "total_samples": samples_per_freq,
                "elapsed_time_sec": round(t_elapsed, 4),
                "mean_latency_ms": lat_sum["total_latency"]["mean_ms"],
                "p99_latency_ms": lat_sum["p99_total_ms"],
                "is_realtime_capable": bool(actual_hz >= hz)
            }

        return throughput_records

    @classmethod
    def run_memory_stability_test(
        cls,
        engine: RealtimeNavigationEngine,
        total_epochs: int = 3000
    ) -> Dict[str, Any]:
        """
        Simulate 5+ minutes continuous stream and track RSS memory footprint.
        """
        mem = MemoryMonitor()
        mem.record_snapshot("startup", buffer_length=0)

        engine.reset()
        for i in range(total_epochs):
            engine.process_sensor_sample(i * 0.1, 0.1, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0)
            if (i + 1) % 600 == 0:  # Every 1 minute of 10Hz data
                mins = (i + 1) // 600
                mem.record_snapshot(f"{mins}_minute_mark", buffer_length=len(engine.window_buffer))

        mem.record_snapshot("final_completion", buffer_length=len(engine.window_buffer))
        return mem.evaluate_memory_stability()

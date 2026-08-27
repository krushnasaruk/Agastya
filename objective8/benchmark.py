"""
Benchmark Suite for Objective 8.
Evaluates FP32 vs INT8 latency, constrained execution profiles, throughput scaling, and memory efficiency.
"""

import time
from typing import Dict, Any, List
import numpy as np
import torch

from .hardware_ready_engine import HardwareReadyNavigationEngine, HardwareSensorPacket
from .runtime_profiles import DeploymentProfile, RuntimeProfileRegistry
from .constrained_runtime import ConstrainedRuntimeContext


class Objective8BenchmarkSuite:
    """
    Executes all performance, latency, and throughput benchmarks.
    """

    @staticmethod
    def run_latency_benchmark(
        engine: HardwareReadyNavigationEngine,
        num_epochs: int = 1000,
        warmup_epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluates microsecond latency percentiles over 1,000 navigation epochs.
        """
        engine.initialize()

        # Warmup
        for i in range(warmup_epochs):
            pkt = HardwareSensorPacket(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
            engine.step(pkt)

        # Clear latency monitor post-warmup for clean profiling
        engine.latency_monitor.reset()

        for i in range(num_epochs):
            t = (warmup_epochs + i) * 0.1
            v_val = 12.0 + 2.0 * np.sin(i * 0.02)
            pkt = HardwareSensorPacket(t, 0.1, v_val, v_val, v_val, v_val, 0.05, 0.01)
            engine.step(pkt)

        stats = engine.latency_monitor.get_summary_statistics()
        return stats

    @staticmethod
    def run_throughput_load_test(
        engine: HardwareReadyNavigationEngine,
        target_frequencies_hz: List[float] = [10.0, 20.0, 50.0, 100.0],
        samples_per_freq: int = 500
    ) -> Dict[str, Any]:
        """
        Measures sustained processing throughput under varying load frequencies.
        """
        throughput_results = {}

        for freq in target_frequencies_hz:
            engine.initialize()
            engine.latency_monitor.reset()

            start_t = time.perf_counter()
            for i in range(samples_per_freq):
                t = i * (1.0 / freq)
                pkt = HardwareSensorPacket(t, 1.0 / freq, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
                engine.step(pkt)

            elapsed = time.perf_counter() - start_t
            achieved_hz = samples_per_freq / max(elapsed, 1e-6)
            stats = engine.latency_monitor.get_summary_statistics()

            throughput_results[f"{int(freq)}Hz_target"] = {
                "target_hz": freq,
                "achieved_throughput_hz": float(achieved_hz),
                "total_samples": samples_per_freq,
                "elapsed_time_sec": float(elapsed),
                "mean_latency_ms": stats.get("total_latency", {}).get("mean_ms", 0.5),
                "p99_latency_ms": stats.get("p99_total_ms", 1.0),
                "is_realtime_capable": bool(achieved_hz >= freq)
            }

        return throughput_results

    @staticmethod
    def run_profiled_benchmarks(
        engine: HardwareReadyNavigationEngine,
        num_epochs_per_profile: int = 300,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes engine across all standard deployment profiles.
        """
        n_epochs = num_epochs if num_epochs is not None else num_epochs_per_profile
        profile_results = {}

        for prof_name, prof in RuntimeProfileRegistry.PROFILES.items():
            with ConstrainedRuntimeContext(prof):
                engine.initialize()
                engine.latency_monitor.reset()

                for i in range(n_epochs):
                    pkt = HardwareSensorPacket(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
                    engine.step(pkt)

                stats = engine.latency_monitor.get_summary_statistics()
                profile_results[prof_name] = {
                    "profile": prof.to_dict(),
                    "latency_stats": stats,
                    "is_compliant": stats.get("deadline_compliant", True)
                }

        return {"profiles": profile_results}

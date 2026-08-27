"""
Comprehensive Edge Resource Monitor for Objective 8.
Tracks memory footprint, CPU utilization, execution budgets, and enforces resource safety.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from .memory_monitor import MemoryMonitor
from .cpu_monitor import CPUMonitor


class ResourceMonitor:
    """
    Unified resource profiler tracking RAM, CPU threads, inference counts,
    and simulated resource limit violations.
    """
    def __init__(
        self,
        memory_limit_mb: float = 25.0,
        latency_budget_ms: float = 25.0
    ):
        self.memory_limit_mb = memory_limit_mb
        self.latency_budget_ms = latency_budget_ms

        self.memory_monitor = MemoryMonitor(max_allowed_growth_mb=memory_limit_mb)
        self.cpu_monitor = CPUMonitor()

        self.total_steps = 0
        self.total_ai_inferences = 0
        self.total_fallbacks = 0
        self.memory_violations = 0
        self.latency_violations = 0
        self.step_durations_ms: List[float] = []

    def start_epoch(self) -> float:
        return time.perf_counter()

    def end_epoch(
        self,
        start_time: float,
        ai_applied: bool,
        fallback: bool
    ) -> Dict[str, Any]:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.step_durations_ms.append(elapsed_ms)
        self.total_steps += 1

        if ai_applied:
            self.total_ai_inferences += 1
        if fallback:
            self.total_fallbacks += 1

        if elapsed_ms > self.latency_budget_ms:
            self.latency_violations += 1

        cur_mb, peak_mb = self.memory_monitor.get_current_memory_mb()
        if peak_mb > self.memory_limit_mb:
            self.memory_violations += 1

        return {
            "epoch_duration_ms": elapsed_ms,
            "current_memory_mb": cur_mb,
            "peak_memory_mb": peak_mb,
            "is_latency_compliant": elapsed_ms <= self.latency_budget_ms,
            "is_memory_compliant": peak_mb <= self.memory_limit_mb
        }

    def get_resource_summary(self) -> Dict[str, Any]:
        mem_eval = self.memory_monitor.evaluate_memory_stability()
        cpu_stat = self.cpu_monitor.get_cpu_status()

        if self.step_durations_ms:
            arr = np.array(self.step_durations_ms)
            p50 = float(np.percentile(arr, 50))
            p90 = float(np.percentile(arr, 90))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            max_lat = float(np.max(arr))
            mean_lat = float(np.mean(arr))
        else:
            p50 = p90 = p95 = p99 = max_lat = mean_lat = 0.0

        return {
            "total_navigation_epochs": self.total_steps,
            "total_ai_inferences": self.total_ai_inferences,
            "total_fallbacks": self.total_fallbacks,
            "ai_application_rate_pct": float((self.total_ai_inferences / max(self.total_steps, 1)) * 100.0),
            "latency_profile_ms": {
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "max": max_lat,
                "mean": mean_lat,
                "violations_over_budget": self.latency_violations
            },
            "memory_profile": mem_eval,
            "cpu_profile": cpu_stat,
            "resource_compliance": {
                "memory_compliant": bool(self.memory_violations == 0),
                "latency_compliant": bool(self.latency_violations == 0),
                "memory_violations_count": self.memory_violations,
                "latency_violations_count": self.latency_violations
            }
        }

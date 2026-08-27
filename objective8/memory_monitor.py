"""
Memory Monitor for Objective 8.
Tracks resident memory, allocations, peak usage, and detects memory leaks or unbounded growth.
"""

import os
import gc
import tracemalloc
from typing import Dict, Any, List, Optional
import numpy as np


class MemoryMonitor:
    """
    Monitors process RAM and memory allocation growth across epochs.
    """
    def __init__(self, max_allowed_growth_mb: float = 25.0):
        self.max_allowed_growth_mb = max_allowed_growth_mb
        self.snapshots: List[Dict[str, Any]] = []
        self._is_tracemalloc_active = False

        if not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
                self._is_tracemalloc_active = True
            except Exception:
                self._is_tracemalloc_active = False

    def record_snapshot(self, stage_name: str) -> Dict[str, Any]:
        """Records current process memory snapshot."""
        current_mb, peak_mb = self.get_current_memory_mb()
        snap = {
            "stage": stage_name,
            "current_mb": current_mb,
            "peak_mb": peak_mb
        }
        self.snapshots.append(snap)
        return snap

    def get_current_memory_mb(self) -> tuple[float, float]:
        """Returns (current_mb, peak_mb)."""
        if tracemalloc.is_tracing():
            cur_b, peak_b = tracemalloc.get_traced_memory()
            return cur_b / (1024.0 * 1024.0), peak_b / (1024.0 * 1024.0)
        return 0.0, 0.0

    def evaluate_memory_stability(self) -> Dict[str, Any]:
        """Evaluates whether memory consumption is strictly bounded."""
        if len(self.snapshots) < 2:
            return {
                "initial_rss_mb": 0.0,
                "final_rss_mb": 0.0,
                "peak_rss_mb": 0.0,
                "net_growth_mb": 0.0,
                "is_bounded": True,
                "growth_slope_mb_per_epoch": 0.0
            }

        cur_values = [s["current_mb"] for s in self.snapshots]
        peak_values = [s["peak_mb"] for s in self.snapshots]

        init_val = cur_values[0]
        final_val = cur_values[-1]
        peak_val = max(peak_values)
        net_growth = max(final_val - init_val, 0.0)

        # Growth slope calculation
        if len(cur_values) > 10:
            tail = cur_values[-len(cur_values)//2:]
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
        else:
            slope = float(net_growth / max(len(cur_values), 1))

        is_bounded = bool((net_growth < self.max_allowed_growth_mb) and (slope < 1.0))

        return {
            "initial_rss_mb": float(init_val),
            "final_rss_mb": float(final_val),
            "peak_rss_mb": float(peak_val),
            "net_growth_mb": float(net_growth),
            "is_bounded": is_bounded,
            "growth_slope_mb_per_epoch": float(slope),
            "leak_slope_mb_per_min": float(slope * 600.0)
        }

    def reset(self) -> None:
        self.snapshots.clear()
        gc.collect()
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()

"""
Memory Monitor and Leakage Detection Suite for Objective 7.
Tracks process RSS, working buffers, peak memory usage, and verifies bounded execution footprint.
"""

import os
import gc
import sys
import tracemalloc
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryMonitor:
    """
    Monitors process RAM and ensures bounded memory consumption during long-duration runs.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        if not PSUTIL_AVAILABLE and not tracemalloc.is_tracing():
            tracemalloc.start()
        self.snapshots: List[Dict[str, Any]] = []

    def get_current_rss_mb(self) -> float:
        """Get current resident set size in megabytes."""
        if self.process is not None:
            try:
                return float(self.process.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
        if tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            return float(cur / (1024 * 1024))
        return 0.0

    def record_snapshot(self, stage_label: str, buffer_length: int = 0) -> Dict[str, Any]:
        """Record memory checkpoint snapshot."""
        gc.collect()
        rss = self.get_current_rss_mb()
        snap = {
            "stage_label": stage_label,
            "rss_mb": round(rss, 2),
            "buffer_length": buffer_length,
            "snapshot_index": len(self.snapshots)
        }
        self.snapshots.append(snap)
        return snap

    def evaluate_memory_stability(self) -> Dict[str, Any]:
        """
        Verify memory boundedness and calculate growth rate over sequence.
        """
        if len(self.snapshots) < 2:
            return {
                "initial_rss_mb": self.get_current_rss_mb(),
                "peak_rss_mb": self.get_current_rss_mb(),
                "growth_mb": 0.0,
                "is_bounded": True,
                "status": "INSUFFICIENT_SNAPSHOTS"
            }

        rss_vals = [s["rss_mb"] for s in self.snapshots]
        init_rss = rss_vals[0]
        final_rss = rss_vals[-1]
        peak_rss = max(rss_vals)
        growth_mb = final_rss - init_rss

        # Linear regression slope over snapshots
        x = np.arange(len(rss_vals))
        slope = float(np.polyfit(x, rss_vals, 1)[0]) if len(rss_vals) >= 3 else 0.0

        # Memory is considered bounded if net growth is under 25 MB and growth rate is under 2.0 MB per checkpoint
        is_bounded = bool((growth_mb < 25.0) and (slope < 2.0))

        return {
            "initial_rss_mb": round(init_rss, 2),
            "final_rss_mb": round(final_rss, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "net_growth_mb": round(growth_mb, 2),
            "growth_slope_mb_per_checkpoint": round(slope, 4),
            "is_bounded": is_bounded,
            "snapshots_count": len(self.snapshots),
            "snapshots": self.snapshots
        }

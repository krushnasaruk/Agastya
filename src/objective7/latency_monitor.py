"""
High-Resolution Latency Monitor and Stage Breakdown for Objective 7.
Measures per-epoch processing stages and computes strict latency percentiles (p50, p90, p95, p99, max).
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class EpochLatencyBreakdown:
    sensor_validation_ms: float
    classical_physics_ms: float
    feature_extraction_ms: float
    window_update_ms: float
    neural_inference_ms: float
    policy_evaluation_ms: float
    telemetry_ms: float
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyMonitor:
    """
    Tracks microsecond-precision timing across all navigation pipeline stages.
    """
    def __init__(self, deadline_ms: float = 100.0, preferred_target_ms: float = 50.0):
        self.deadline_ms = deadline_ms
        self.preferred_target_ms = preferred_target_ms
        self.records: List[EpochLatencyBreakdown] = []

    def reset(self) -> None:
        self.records.clear()

    def record_epoch(self, breakdown: EpochLatencyBreakdown) -> None:
        self.records.append(breakdown)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute latency percentiles and deadline compliance.
        """
        if not self.records:
            return {"sample_count": 0, "status": "NO_RECORDS"}

        n = len(self.records)
        totals = np.array([r.total_latency_ms for r in self.records])
        infer = np.array([r.neural_inference_ms for r in self.records])
        class_phys = np.array([r.classical_physics_ms for r in self.records])
        feats = np.array([r.feature_extraction_ms for r in self.records])
        policy = np.array([r.policy_evaluation_ms for r in self.records])
        s_val = np.array([r.sensor_validation_ms for r in self.records])
        telem = np.array([r.telemetry_ms for r in self.records])

        def _stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean_ms": round(float(np.mean(arr)), 4),
                "std_ms": round(float(np.std(arr)), 4),
                "median_ms": round(float(np.median(arr)), 4),
                "p90_ms": round(float(np.percentile(arr, 90)), 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 4),
                "p99_ms": round(float(np.percentile(arr, 99)), 4),
                "max_ms": round(float(np.max(arr)), 4),
                "min_ms": round(float(np.min(arr)), 4)
            }

        p99_total = float(np.percentile(totals, 99))
        p99_infer = float(np.percentile(infer, 99))
        deadline_violations = int(np.sum(totals > self.deadline_ms))
        target_violations = int(np.sum(totals > self.preferred_target_ms))

        return {
            "total_epochs": n,
            "deadline_ms": self.deadline_ms,
            "preferred_target_ms": self.preferred_target_ms,
            "p99_total_ms": round(p99_total, 4),
            "p99_inference_ms": round(p99_infer, 4),
            "deadline_violation_count": deadline_violations,
            "deadline_violation_pct": round((deadline_violations / n) * 100.0, 3),
            "target_violation_count": target_violations,
            "target_violation_pct": round((target_violations / n) * 100.0, 3),
            "deadline_compliant": bool(p99_total < self.deadline_ms),
            "preferred_target_compliant": bool(p99_total < self.preferred_target_ms),
            "total_latency": _stats(totals),
            "neural_inference": _stats(infer),
            "classical_physics": _stats(class_phys),
            "feature_extraction": _stats(feats),
            "policy_evaluation": _stats(policy),
            "sensor_validation": _stats(s_val),
            "telemetry": _stats(telem)
        }

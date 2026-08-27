"""
Timestamp & Timing Jitter Analysis Module for IO-VNBD Data Engineering.
Computes dynamic sample-by-sample dt_k statistics directly from raw timestamps,
detects non-monotonicities, clock resets, duplicate timestamps, and gap anomalies.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class TimestampStats:
    num_samples: int
    duration_sec: float
    nominal_rate_hz: float
    observed_rate_hz: float
    mean_dt_sec: float
    median_dt_sec: float
    std_dt_sec: float
    min_dt_sec: float
    max_dt_sec: float
    p25_dt_sec: float
    p75_dt_sec: float
    p95_dt_sec: float
    p99_dt_sec: float
    is_strictly_monotonic: bool
    num_duplicates: int          # dt == 0
    num_retrograde: int          # dt < 0 (clock resets)
    num_large_gaps: int          # dt > threshold
    max_gap_sec: float
    num_fallback_substitutions: int
    fallback_dt_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TimestampAnalyzer:
    """
    Analyzes temporal consistency, sample spacing, and clock jitter for sensor streams.
    Preserves actual timestamps and dynamic dt_k without silent unmasked substitutions.
    """
    def __init__(
        self,
        gap_threshold_sec: float = 0.25,
        nominal_rate_hz: float = 10.0,
        fallback_dt_sec: float = 0.10
    ):
        self.gap_threshold_sec = gap_threshold_sec
        self.nominal_rate_hz = nominal_rate_hz
        self.fallback_dt_sec = fallback_dt_sec

    def analyze(self, raw_timestamps_ms: np.ndarray) -> Tuple[TimestampStats, np.ndarray, np.ndarray]:
        """
        Analyze an array of millisecond timestamps.

        Parameters:
            raw_timestamps_ms: Array of raw timestamps in milliseconds

        Returns:
            stats: Comprehensive TimestampStats summary
            dt_array: Sample-by-sample delta-t array in seconds (length N, dt[0] = median_dt)
            valid_time_mask: Boolean mask marking temporally valid, strictly monotonic steps
        """
        raw_arr = np.asarray(raw_timestamps_ms, dtype=np.float64)
        n = len(raw_arr)

        if n < 2:
            stats = TimestampStats(
                num_samples=n,
                duration_sec=0.0,
                nominal_rate_hz=self.nominal_rate_hz,
                observed_rate_hz=self.nominal_rate_hz,
                mean_dt_sec=self.fallback_dt_sec,
                median_dt_sec=self.fallback_dt_sec,
                std_dt_sec=0.0,
                min_dt_sec=0.0,
                max_dt_sec=0.0,
                p25_dt_sec=0.0,
                p75_dt_sec=0.0,
                p95_dt_sec=0.0,
                p99_dt_sec=0.0,
                is_strictly_monotonic=True,
                num_duplicates=0,
                num_retrograde=0,
                num_large_gaps=0,
                max_gap_sec=0.0,
                num_fallback_substitutions=0,
                fallback_dt_sec=self.fallback_dt_sec
            )
            return stats, np.array([self.fallback_dt_sec] * n), np.ones(n, dtype=bool)

        # Convert to seconds
        t_sec = raw_arr / 1000.0
        raw_deltas = np.diff(t_sec)  # Length N - 1

        # Monotonicity & Anomaly counters
        num_duplicates = int(np.sum(raw_deltas == 0.0))
        num_retrograde = int(np.sum(raw_deltas < 0.0))
        num_large_gaps = int(np.sum(raw_deltas > self.gap_threshold_sec))
        is_monotonic = bool(np.all(raw_deltas > 0.0))

        # Filter strictly positive deltas for statistics
        valid_deltas = raw_deltas[raw_deltas > 0.0]
        if len(valid_deltas) == 0:
            median_dt = self.fallback_dt_sec
            mean_dt = median_dt
            std_dt = 0.0
            min_dt = 0.0
            max_dt = 0.0
            p25 = p75 = p95 = p99 = median_dt
        else:
            mean_dt = float(np.mean(valid_deltas))
            median_dt = float(np.median(valid_deltas))
            std_dt = float(np.std(valid_deltas))
            min_dt = float(np.min(valid_deltas))
            max_dt = float(np.max(valid_deltas))
            p25 = float(np.percentile(valid_deltas, 25))
            p75 = float(np.percentile(valid_deltas, 75))
            p95 = float(np.percentile(valid_deltas, 95))
            p99 = float(np.percentile(valid_deltas, 99))

        duration_sec = float(t_sec[-1] - t_sec[0]) if is_monotonic else float(np.sum(valid_deltas))
        observed_rate = float(n / duration_sec) if duration_sec > 0 else self.nominal_rate_hz

        # Construct full dynamic dt array
        # dt[0] is set to initial step based on median dt
        dt_array = np.empty(n, dtype=np.float64)
        dt_array[0] = median_dt

        # For steps k >= 1, use actual delta. If delta <= 0 (duplicate/retrograde),
        # preserve actual delta for audit while marking invalid in mask.
        dt_array[1:] = raw_deltas

        # Count fallback substitutions if any non-positive deltas are found
        num_fallbacks = int(np.sum(raw_deltas <= 0.0))

        # Validity mask: strictly positive and within gap threshold
        valid_time_mask = np.ones(n, dtype=bool)
        valid_time_mask[1:] = (raw_deltas > 0.0) & (raw_deltas <= self.gap_threshold_sec * 2.0)

        stats = TimestampStats(
            num_samples=n,
            duration_sec=round(duration_sec, 3),
            nominal_rate_hz=self.nominal_rate_hz,
            observed_rate_hz=round(observed_rate, 2),
            mean_dt_sec=round(mean_dt, 5),
            median_dt_sec=round(median_dt, 5),
            std_dt_sec=round(std_dt, 5),
            min_dt_sec=round(min_dt, 5),
            max_dt_sec=round(max_dt, 5),
            p25_dt_sec=round(p25, 5),
            p75_dt_sec=round(p75, 5),
            p95_dt_sec=round(p95, 5),
            p99_dt_sec=round(p99, 5),
            is_strictly_monotonic=is_monotonic,
            num_duplicates=num_duplicates,
            num_retrograde=num_retrograde,
            num_large_gaps=num_large_gaps,
            max_gap_sec=round(float(np.max(raw_deltas)), 5) if len(raw_deltas) > 0 else 0.0,
            num_fallback_substitutions=num_fallbacks,
            fallback_dt_sec=self.fallback_dt_sec
        )

        return stats, dt_array, valid_time_mask

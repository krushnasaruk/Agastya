"""
Data Quality Assessment & Masking Engine for Project AGASTYA.
Generates non-destructive boolean quality masks (VALID, MISSING, INVALID, SUSPICIOUS)
and audit statistics without altering underlying raw observations.
"""

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Dict, Any, Tuple, Optional
import numpy as np


class QualityFlag(IntEnum):
    VALID = 0
    MISSING = 1
    INVALID = 2
    SUSPICIOUS = 3
    INTERPOLATED = 4


@dataclass
class QualitySummary:
    total_samples: int
    valid_samples_count: int
    valid_fraction_pct: float
    missing_samples_count: int
    invalid_samples_count: int
    suspicious_samples_count: int
    sensor_valid_fraction_pct: float
    gps_valid_fraction_pct: float
    reference_valid_fraction_pct: float
    max_consecutive_valid_samples: int
    num_isolated_dropouts: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataQualityManager:
    """
    Evaluates signal integrity and builds non-destructive bitmasks.
    """
    def __init__(
        self,
        max_speed_ms: float = 70.0,      # ~250 km/h
        max_accel_ms2: float = 25.0,     # ~2.5g
        max_yaw_rate_rads: float = 3.0,  # ~170 deg/s
        max_gps_accuracy_m: float = 30.0,
        min_satellites: int = 4
    ):
        self.max_speed_ms = max_speed_ms
        self.max_accel_ms2 = max_accel_ms2
        self.max_yaw_rate_rads = max_yaw_rate_rads
        self.max_gps_accuracy_m = max_gps_accuracy_m
        self.min_satellites = min_satellites

    def evaluate_quality(
        self,
        data: Dict[str, np.ndarray],
        time_mask: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], QualitySummary]:
        """
        Evaluate full multi-channel dataset dictionary and construct quality masks.

        Returns:
            masks: Dict of boolean masks ('sensor_valid', 'gps_valid', 'reference_valid', 'overall_valid', 'quality_code')
            summary: QualitySummary statistics
        """
        n = len(next(iter(data.values()))) if data else 0
        if n == 0:
            empty_mask = np.zeros(0, dtype=bool)
            summary = QualitySummary(0, 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0)
            return {
                "sensor_valid_mask": empty_mask,
                "gps_valid_mask": empty_mask,
                "reference_valid_mask": empty_mask,
                "overall_valid_mask": empty_mask,
                "quality_flags": np.zeros(0, dtype=np.int8)
            }, summary

        # 1. Base NaN / Inf Check
        has_nan = np.zeros(n, dtype=bool)
        for key, arr in data.items():
            if np.issubdtype(arr.dtype, np.floating):
                has_nan |= np.isnan(arr) | np.isinf(arr)

        # 2. Sensor Validity Mask (Wheel Speeds, Accel, Yaw Rate)
        sensor_valid = ~has_nan.copy()
        
        # Check wheel speeds if present
        for col in ["wheel_speed_fl_ms", "wheel_speed_fr_ms", "wheel_speed_rl_ms", "wheel_speed_rr_ms"]:
            if col in data:
                ws = data[col]
                sensor_valid &= (ws >= -2.0) & (ws <= self.max_speed_ms)

        # Check longitudinal acceleration
        if "accel_x_ms2" in data:
            ax = data["accel_x_ms2"]
            sensor_valid &= (np.abs(ax) <= self.max_accel_ms2)

        # Check yaw rate
        if "yaw_rate_rads" in data:
            yr = data["yaw_rate_rads"]
            sensor_valid &= (np.abs(yr) <= self.max_yaw_rate_rads)

        # 3. GPS / Reference Validity Mask
        gps_valid = np.ones(n, dtype=bool)
        if "latitude_deg" in data and "longitude_deg" in data:
            lat = data["latitude_deg"]
            lon = data["longitude_deg"]
            gps_valid &= ~np.isnan(lat) & ~np.isnan(lon)
            gps_valid &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
            # Flag origin (0, 0) GPS failure
            gps_valid &= ~((np.abs(lat) < 1e-4) & (np.abs(lon) < 1e-4))

        if "gps_accuracy_m" in data:
            acc = data["gps_accuracy_m"]
            gps_valid &= (acc <= self.max_gps_accuracy_m)

        if "satellites_count" in data:
            sats = data["satellites_count"]
            gps_valid &= (sats >= self.min_satellites)

        reference_valid = gps_valid.copy()
        if "gps_speed_ms" in data:
            gs = data["gps_speed_ms"]
            reference_valid &= ~np.isnan(gs) & (gs >= 0.0) & (gs <= self.max_speed_ms)

        # 4. Overall Master Validity Mask
        t_mask = time_mask if time_mask is not None else np.ones(n, dtype=bool)
        overall_valid = sensor_valid & reference_valid & t_mask

        # 5. Categorical Quality Codes
        quality_codes = np.full(n, QualityFlag.VALID, dtype=np.int8)
        quality_codes[has_nan] = QualityFlag.MISSING
        quality_codes[~sensor_valid & ~has_nan] = QualityFlag.INVALID
        quality_codes[~reference_valid & sensor_valid & ~has_nan] = QualityFlag.SUSPICIOUS

        # 6. Compute Quality Statistics
        valid_count = int(np.sum(overall_valid))
        missing_count = int(np.sum(has_nan))
        invalid_count = int(np.sum(quality_codes == QualityFlag.INVALID))
        suspicious_count = int(np.sum(quality_codes == QualityFlag.SUSPICIOUS))

        # Max consecutive valid samples
        max_consec = 0
        curr_consec = 0
        for val in overall_valid:
            if val:
                curr_consec += 1
                if curr_consec > max_consec:
                    max_consec = curr_consec
            else:
                curr_consec = 0

        # Isolated dropouts (single invalid between valids)
        isolated_dropouts = 0
        if n >= 3:
            for i in range(1, n - 1):
                if not overall_valid[i] and overall_valid[i - 1] and overall_valid[i + 1]:
                    isolated_dropouts += 1

        summary = QualitySummary(
            total_samples=n,
            valid_samples_count=valid_count,
            valid_fraction_pct=round(float(valid_count / n * 100.0), 2),
            missing_samples_count=missing_count,
            invalid_samples_count=invalid_count,
            suspicious_samples_count=suspicious_count,
            sensor_valid_fraction_pct=round(float(np.sum(sensor_valid) / n * 100.0), 2),
            gps_valid_fraction_pct=round(float(np.sum(gps_valid) / n * 100.0), 2),
            reference_valid_fraction_pct=round(float(np.sum(reference_valid) / n * 100.0), 2),
            max_consecutive_valid_samples=max_consec,
            num_isolated_dropouts=isolated_dropouts
        )

        masks = {
            "sensor_valid_mask": sensor_valid,
            "gps_valid_mask": gps_valid,
            "reference_valid_mask": reference_valid,
            "timestamp_valid_mask": t_mask,
            "overall_valid_mask": overall_valid,
            "quality_flags": quality_codes
        }

        return masks, summary

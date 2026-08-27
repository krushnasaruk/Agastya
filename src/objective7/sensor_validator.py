"""
Sensor Validator and Sanity Gate for Real-Time Navigation Engine (Objective 7).
Enforces robust input validation, detects dropouts, NaNs, Infs, and timestamp/dt anomalies.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class SensorValidationResult:
    is_valid: bool
    is_degraded: bool
    status_code: str
    cleaned_wheel_fl: float
    cleaned_wheel_fr: float
    cleaned_wheel_rl: float
    cleaned_wheel_rr: float
    cleaned_accel_x: float
    cleaned_yaw_rate: float
    cleaned_dt: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensorValidator:
    """
    Validates single sensor frames in real time with zero latency penalty.
    """
    def __init__(
        self,
        min_dt_sec: float = 0.001,
        max_dt_sec: float = 1.0,
        max_speed_ms: float = 100.0,
        max_accel_ms2: float = 25.0,
        max_yaw_rate_rads: float = 5.0
    ):
        self.min_dt = min_dt_sec
        self.max_dt = max_dt_sec
        self.max_speed = max_speed_ms
        self.max_accel = max_accel_ms2
        self.max_yaw_rate = max_yaw_rate_rads
        self.prev_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset validation state between sequences."""
        self.prev_timestamp = None

    def validate_sample(
        self,
        timestamp_sec: float,
        dt_sec: float,
        wheel_fl: Optional[float],
        wheel_fr: Optional[float],
        wheel_rl: Optional[float],
        wheel_rr: Optional[float],
        accel_x: Optional[float],
        yaw_rate: Optional[float]
    ) -> SensorValidationResult:
        """
        Validate incoming real-time sensor frame.
        """
        # 1. Timestamp validation
        if timestamp_sec is None or np.isnan(timestamp_sec) or np.isinf(timestamp_sec):
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="INVALID_TIMESTAMP",
                cleaned_wheel_fl=0.0, cleaned_wheel_fr=0.0, cleaned_wheel_rl=0.0, cleaned_wheel_rr=0.0,
                cleaned_accel_x=0.0, cleaned_yaw_rate=0.0, cleaned_dt=0.1,
                error_message="Timestamp is NaN or Inf"
            )

        if self.prev_timestamp is not None and timestamp_sec < self.prev_timestamp:
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="NON_MONOTONIC_TIMESTAMP",
                cleaned_wheel_fl=0.0, cleaned_wheel_fr=0.0, cleaned_wheel_rl=0.0, cleaned_wheel_rr=0.0,
                cleaned_accel_x=0.0, cleaned_yaw_rate=0.0, cleaned_dt=0.1,
                error_message=f"Timestamp decreased from {self.prev_timestamp} to {timestamp_sec}"
            )
        self.prev_timestamp = timestamp_sec

        # 2. dt validation
        if dt_sec is None or np.isnan(dt_sec) or np.isinf(dt_sec) or dt_sec < self.min_dt or dt_sec > self.max_dt:
            cleaned_dt = 0.10 if (dt_sec is None or np.isnan(dt_sec) or dt_sec <= 0) else min(dt_sec, self.max_dt)
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="INVALID_DT",
                cleaned_wheel_fl=0.0, cleaned_wheel_fr=0.0, cleaned_wheel_rl=0.0, cleaned_wheel_rr=0.0,
                cleaned_accel_x=0.0, cleaned_yaw_rate=0.0, cleaned_dt=cleaned_dt,
                error_message=f"Invalid dt: {dt_sec}"
            )
        cleaned_dt = float(dt_sec)

        # 3. Wheel speed validation
        wheels = [wheel_fl, wheel_fr, wheel_rl, wheel_rr]
        if any(w is None or np.isnan(w) or np.isinf(w) or w < -5.0 or w > self.max_speed for w in wheels):
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="INVALID_WHEEL_SPEED",
                cleaned_wheel_fl=0.0, cleaned_wheel_fr=0.0, cleaned_wheel_rl=0.0, cleaned_wheel_rr=0.0,
                cleaned_accel_x=0.0, cleaned_yaw_rate=0.0, cleaned_dt=cleaned_dt,
                error_message="One or more wheel speeds are invalid or out of physical bounds"
            )
        c_fl, c_fr, c_rl, c_rr = [float(w) for w in wheels]

        # 4. IMU validation
        if accel_x is None or np.isnan(accel_x) or np.isinf(accel_x) or abs(accel_x) > self.max_accel:
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="INVALID_ACCELERATION",
                cleaned_wheel_fl=c_fl, cleaned_wheel_fr=c_fr, cleaned_wheel_rl=c_rl, cleaned_wheel_rr=c_rr,
                cleaned_accel_x=0.0, cleaned_yaw_rate=0.0, cleaned_dt=cleaned_dt,
                error_message=f"Acceleration {accel_x} out of physical bounds"
            )
        c_ax = float(accel_x)

        if yaw_rate is None or np.isnan(yaw_rate) or np.isinf(yaw_rate) or abs(yaw_rate) > self.max_yaw_rate:
            return SensorValidationResult(
                is_valid=False, is_degraded=True, status_code="INVALID_YAW_RATE",
                cleaned_wheel_fl=c_fl, cleaned_wheel_fr=c_fr, cleaned_wheel_rl=c_rl, cleaned_wheel_rr=c_rr,
                cleaned_accel_x=c_ax, cleaned_yaw_rate=0.0, cleaned_dt=cleaned_dt,
                error_message=f"Yaw rate {yaw_rate} out of physical bounds"
            )
        c_yr = float(yaw_rate)

        return SensorValidationResult(
            is_valid=True,
            is_degraded=False,
            status_code="SENSOR_HEALTHY",
            cleaned_wheel_fl=c_fl,
            cleaned_wheel_fr=c_fr,
            cleaned_wheel_rl=c_rl,
            cleaned_wheel_rr=c_rr,
            cleaned_accel_x=c_ax,
            cleaned_yaw_rate=c_yr,
            cleaned_dt=cleaned_dt,
            error_message=None
        )

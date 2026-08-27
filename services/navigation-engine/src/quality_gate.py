"""
Causal Sensor Quality Gate for Project AGASTYA (Objective 3).
Validates raw onboard measurements against physical bounds and Objective 2 masks
before propagation in the classical dead-reckoning engine.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class SanitizedSensorInput:
    time_sec: float
    dt_sec: float
    wheel_speed_fl_ms: Optional[float]
    wheel_speed_fr_ms: Optional[float]
    wheel_speed_rl_ms: Optional[float]
    wheel_speed_rr_ms: Optional[float]
    accel_x_ms2: Optional[float]
    yaw_rate_rads: Optional[float]
    is_valid_epoch: bool
    quality_status: str


class CausalQualityGate:
    """
    Guards navigation state against NaN/Inf, sensor clipping, and invalid time steps.
    """
    def __init__(
        self,
        max_speed_ms: float = 70.0,
        max_accel_ms2: float = 20.0,
        max_yaw_rate_rads: float = 3.0,
        min_dt_sec: float = 0.005,
        max_dt_sec: float = 0.50
    ):
        self.max_speed_ms = max_speed_ms
        self.max_accel_ms2 = max_accel_ms2
        self.max_yaw_rate_rads = max_yaw_rate_rads
        self.min_dt_sec = min_dt_sec
        self.max_dt_sec = max_dt_sec

    def sanitize_epoch(
        self,
        time_sec: float,
        dt_sec: float,
        v_fl: Optional[float] = None,
        v_fr: Optional[float] = None,
        v_rl: Optional[float] = None,
        v_rr: Optional[float] = None,
        accel_x: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        mask_valid: bool = True
    ) -> SanitizedSensorInput:
        """
        Sanitize single-epoch sensor readings strictly causally.
        """
        # 1. Timestamp validation
        if np.isnan(dt_sec) or dt_sec < self.min_dt_sec or dt_sec > self.max_dt_sec or not mask_valid:
            valid_dt = max(min(dt_sec if not np.isnan(dt_sec) else 0.10, self.max_dt_sec), self.min_dt_sec)
            return SanitizedSensorInput(
                time_sec=time_sec,
                dt_sec=valid_dt,
                wheel_speed_fl_ms=None,
                wheel_speed_fr_ms=None,
                wheel_speed_rl_ms=None,
                wheel_speed_rr_ms=None,
                accel_x_ms2=None,
                yaw_rate_rads=None,
                is_valid_epoch=False,
                quality_status="INVALID_TIMESTAMP_OR_MASK"
            )

        # 2. Wheel speed validation
        def clean_speed(v):
            if v is not None and not np.isnan(v) and 0.0 <= v <= self.max_speed_ms:
                return float(v)
            return None

        c_fl = clean_speed(v_fl)
        c_fr = clean_speed(v_fr)
        c_rl = clean_speed(v_rl)
        c_rr = clean_speed(v_rr)

        # 3. Longitudinal Accel validation
        c_ax = None
        if accel_x is not None and not np.isnan(accel_x) and abs(accel_x) <= self.max_accel_ms2:
            c_ax = float(accel_x)

        # 4. Yaw rate validation
        c_yr = None
        if yaw_rate is not None and not np.isnan(yaw_rate) and abs(yaw_rate) <= self.max_yaw_rate_rads:
            c_yr = float(yaw_rate)

        has_sensors = any(x is not None for x in [c_fl, c_fr, c_rl, c_rr, c_ax, c_yr])
        status = "VALID" if has_sensors else "DROPOUT"

        return SanitizedSensorInput(
            time_sec=float(time_sec),
            dt_sec=float(dt_sec),
            wheel_speed_fl_ms=c_fl,
            wheel_speed_fr_ms=c_fr,
            wheel_speed_rl_ms=c_rl,
            wheel_speed_rr_ms=c_rr,
            accel_x_ms2=c_ax,
            yaw_rate_rads=c_yr,
            is_valid_epoch=has_sensors,
            quality_status=status
        )

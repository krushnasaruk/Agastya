"""
Heading & Yaw Propagation Module for Project AGASTYA (Objective 3).
Integrates chassis yaw rates using dynamic dt_k with angle wrapping [0, 2*pi),
zero-velocity stationary drift freezing, and kinematic wheel fallback.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .state import wrap_to_2pi, wrap_to_pi


class YawPropagator:
    """
    Propagates vehicle heading using dynamic timesteps and multi-source yaw rates.
    """
    def __init__(
        self,
        initial_heading_rad: float = 0.0,
        gyro_bias_rads: float = 0.0,
        max_yaw_rate_rads: float = 3.0
    ):
        self.heading_rad = wrap_to_2pi(initial_heading_rad)
        self.gyro_bias_rads = gyro_bias_rads
        self.max_yaw_rate_rads = max_yaw_rate_rads

    def reset(self, initial_heading_rad: float = 0.0) -> None:
        self.heading_rad = wrap_to_2pi(initial_heading_rad)

    def step(
        self,
        yaw_rate_can_rads: Optional[float],
        dt_sec: float,
        is_stationary: bool = False,
        kinematic_yaw_fallback_rads: Optional[float] = None
    ) -> Tuple[float, float, str]:
        """
        Propagate heading by dt_sec using active yaw rate.

        Returns:
            updated_heading_rad: New heading in [0, 2*pi) radians
            effective_yaw_rate: The rate used for propagation
            source: Source of yaw rate ('CAN_GYRO', 'WHEEL_DIFFERENTIAL', 'ZERO_STATIONARY', 'HELD')
        """
        if dt_sec <= 0.0:
            return self.heading_rad, 0.0, "INVALID_DT"

        # 1. Zero-Velocity Lock (ZUPT)
        if is_stationary:
            return self.heading_rad, 0.0, "ZERO_STATIONARY"

        # 2. Select Valid Yaw Rate
        source = "CAN_GYRO"
        if yaw_rate_can_rads is not None and not np.isnan(yaw_rate_can_rads) and abs(yaw_rate_can_rads) <= self.max_yaw_rate_rads:
            raw_rate = float(yaw_rate_can_rads)
            unbiased_rate = raw_rate - self.gyro_bias_rads
        elif kinematic_yaw_fallback_rads is not None and not np.isnan(kinematic_yaw_fallback_rads):
            unbiased_rate = float(kinematic_yaw_fallback_rads)
            source = "WHEEL_DIFFERENTIAL"
        else:
            unbiased_rate = 0.0
            source = "HELD_ZERO"

        # 3. Integrate with exact dynamic dt
        delta_heading = unbiased_rate * dt_sec
        self.heading_rad = wrap_to_2pi(self.heading_rad + delta_heading)

        return self.heading_rad, unbiased_rate, source

"""
Zero-Velocity Update (ZUPT) Detector and Corrector.
Monitors multi-axis inertial signal energy to detect stationary stance phases
and applies pseudo-measurement velocity corrections to arrest dead reckoning drift.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional
from .state import NavigationState
from .kalman import ErrorStateKalmanFilter


@dataclass
class ZUPTConfig:
    accel_variance_threshold: float = 0.04    # (m/s^2)^2
    gyro_magnitude_threshold: float = 0.08    # rad/s
    window_size: int = 8
    measurement_noise_std: float = 0.01       # m/s


class ZeroVelocityDetector:
    """
    Sliding-window energy detector evaluating body acceleration variance
    and angular velocity magnitude against stationary thresholds.
    """
    def __init__(self, config: Optional[ZUPTConfig] = None):
        self.config = config or ZUPTConfig()
        self.accel_buffer = deque(maxlen=self.config.window_size)
        self.gyro_buffer = deque(maxlen=self.config.window_size)

    def add_reading(self, accel: np.ndarray, gyro: np.ndarray):
        self.accel_buffer.append(np.asarray(accel, dtype=np.float64))
        self.gyro_buffer.append(np.asarray(gyro, dtype=np.float64))

    def is_stationary(self) -> Tuple[bool, float]:
        """
        Returns (is_stationary, confidence_metric).
        Confidence metric is normalized in [0, 1] where 1 is absolute standstill.
        """
        if len(self.accel_buffer) < self.config.window_size:
            return False, 0.0

        acc_arr = np.array(self.accel_buffer)
        gyr_arr = np.array(self.gyro_buffer)

        # 1. Acceleration magnitude variance
        acc_norms = np.linalg.norm(acc_arr, axis=1)
        acc_var = float(np.var(acc_norms))

        # 2. Gyroscope mean magnitude
        gyr_norms = np.linalg.norm(gyr_arr, axis=1)
        gyr_mean = float(np.mean(gyr_norms))

        is_stat = (acc_var < self.config.accel_variance_threshold) and                   (gyr_mean < self.config.gyro_magnitude_threshold)

        # Compute metric score
        var_ratio = min(1.0, acc_var / (self.config.accel_variance_threshold + 1e-9))
        gyr_ratio = min(1.0, gyr_mean / (self.config.gyro_magnitude_threshold + 1e-9))
        confidence = float(np.clip(1.0 - 0.5 * (var_ratio + gyr_ratio), 0.0, 1.0))

        return is_stat, confidence

    def reset(self):
        self.accel_buffer.clear()
        self.gyro_buffer.clear()


class ZUPTCorrector:
    """
    Coordinates detection and ES-EKF velocity zeroing.
    """
    def __init__(self, filter_instance: ErrorStateKalmanFilter, config: Optional[ZUPTConfig] = None):
        self.filter = filter_instance
        self.detector = ZeroVelocityDetector(config)
        self.r_matrix = np.eye(3, dtype=np.float64) * (self.detector.config.measurement_noise_std ** 2)

    def process(
        self,
        state: NavigationState,
        accel: np.ndarray,
        gyro: np.ndarray
    ) -> Tuple[NavigationState, bool]:
        """
        Applies ZUPT measurement update if stationary condition is detected.
        """
        self.detector.add_reading(accel, gyro)
        is_stat, _ = self.detector.is_stationary()

        if is_stat:
            updated_state, _ = self.filter.update_zero_velocity(state, R_zupt=self.r_matrix)
            return updated_state, True
        return state, False

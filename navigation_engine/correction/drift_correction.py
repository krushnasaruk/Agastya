"""
Drift Correction Algorithms for Dead Reckoning.
Includes:
- Zero-Velocity Update (ZUPT) detection using Generalized Likelihood Ratio Test (GLRT) / ARET
- Non-Holonomic Constraints (NHC) for wheeled / ground vehicles
- AI Velocity & Displacement correction blending
"""

import numpy as np
from typing import Optional, Tuple, Deque
from collections import deque


class DriftCorrector:
    def __init__(
        self,
        zupt_accel_std_threshold: float = 0.25,     # m/s^2
        zupt_gyro_norm_threshold: float = 0.08,      # rad/s
        zupt_window_size: int = 15,
        enable_nhc: bool = False                     # Non-holonomic constraints (lateral/vertical = 0)
    ):
        self.zupt_accel_std_threshold = zupt_accel_std_threshold
        self.zupt_gyro_norm_threshold = zupt_gyro_norm_threshold
        self.zupt_window_size = zupt_window_size
        self.enable_nhc = enable_nhc

        self.accel_history: Deque[np.ndarray] = deque(maxlen=zupt_window_size)
        self.gyro_history: Deque[np.ndarray] = deque(maxlen=zupt_window_size)

    def update_sensor_window(self, accel: np.ndarray, gyro: np.ndarray):
        """Append latest IMU sample to sliding detector window."""
        self.accel_history.append(accel.copy())
        self.gyro_history.append(gyro.copy())

    def detect_zero_velocity(self) -> Tuple[bool, float]:
        """
        GLRT-based stationary detector.
        Evaluates acceleration variance and angular rate magnitude over window.
        Returns: (is_stationary, detector_metric)
        """
        if len(self.accel_history) < self.zupt_window_size:
            return False, 0.0

        acc_arr = np.array(self.accel_history)  # (W, 3)
        gyro_arr = np.array(self.gyro_history)  # (W, 3)

        # Acceleration magnitude variance
        acc_norms = np.linalg.norm(acc_arr, axis=1)
        acc_std = float(np.std(acc_norms))

        # Mean gyro magnitude
        gyro_norms = np.linalg.norm(gyro_arr, axis=1)
        gyro_mean = float(np.mean(gyro_norms))

        is_stationary = (
            acc_std < self.zupt_accel_std_threshold and
            gyro_mean < self.zupt_gyro_norm_threshold
        )

        detector_metric = acc_std + gyro_mean
        return is_stationary, detector_metric

    def apply_nhc_body(self, velocity_body: np.ndarray) -> np.ndarray:
        """
        Apply Non-Holonomic Constraints: For ground vehicles,
        lateral (v_y) and vertical (v_z) velocities in body frame are near zero.
        """
        if not self.enable_nhc:
            return velocity_body

        corrected = velocity_body.copy()
        corrected[1] = 0.0  # Zero lateral slip
        corrected[2] = 0.0  # Zero vertical jump
        return corrected

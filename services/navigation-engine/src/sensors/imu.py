"""
IMU (Inertial Measurement Unit) Sensor Model.
Simulates and models a 6-DOF accelerometer and gyroscope with realistic noise,
Allan variance bias drift, scale factor errors, and gravity projection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class IMUReading:
    timestamp: float
    accel: np.ndarray        # Specific force in body frame (m/s^2), shape (3,)
    gyro: np.ndarray         # Angular rate in body frame (rad/s), shape (3,)
    temperature: float = 25.0
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "accel": [float(x) for x in self.accel],
            "gyro": [float(x) for x in self.gyro],
            "temperature": float(self.temperature),
            "is_valid": self.is_valid
        }


class IMUSensor:
    def __init__(
        self,
        accel_noise_std: float = 0.05,       # m/s^2
        gyro_noise_std: float = 0.005,       # rad/s
        accel_bias_instability: float = 0.001,  # m/s^3
        gyro_bias_instability: float = 0.0001,  # rad/s^2
        accel_init_bias: Optional[np.ndarray] = None,
        gyro_init_bias: Optional[np.ndarray] = None,
        scale_factor_error: float = 0.001,   # 1000 ppm
        seed: Optional[int] = None
    ):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability
        
        self.rng = np.random.RandomState(seed)
        
        # Biases
        if accel_init_bias is not None:
            self.accel_bias = np.array(accel_init_bias, dtype=np.float64)
        else:
            self.accel_bias = self.rng.normal(0, 0.02, 3)
            
        if gyro_init_bias is not None:
            self.gyro_bias = np.array(gyro_init_bias, dtype=np.float64)
        else:
            self.gyro_bias = self.rng.normal(0, 0.002, 3)
            
        # Scale factor matrix
        self.scale_factor_acc = np.eye(3) + self.rng.normal(0, scale_factor_error, (3, 3))
        self.scale_factor_gyro = np.eye(3) + self.rng.normal(0, scale_factor_error, (3, 3))
        
    def step(
        self,
        timestamp: float,
        dt: float,
        true_accel_body: np.ndarray,
        true_gyro_body: np.ndarray
    ) -> IMUReading:
        """
        Generate a realistic noisy IMU reading given ground truth body-frame dynamics.
        """
        # Bias random walk update
        self.accel_bias += self.rng.normal(0, self.accel_bias_instability * np.sqrt(dt), 3)
        self.gyro_bias += self.rng.normal(0, self.gyro_bias_instability * np.sqrt(dt), 3)
        
        # White noise
        acc_noise = self.rng.normal(0, self.accel_noise_std, 3)
        gyro_noise = self.rng.normal(0, self.gyro_noise_std, 3)
        
        # Measurement synthesis
        meas_accel = self.scale_factor_acc @ true_accel_body + self.accel_bias + acc_noise
        meas_gyro = self.scale_factor_gyro @ true_gyro_body + self.gyro_bias + gyro_noise
        
        return IMUReading(
            timestamp=timestamp,
            accel=meas_accel,
            gyro=meas_gyro,
            temperature=25.0 + 0.1 * np.sin(timestamp * 0.01)
        )
        
    def inject_bias_jump(self, accel_delta: np.ndarray, gyro_delta: np.ndarray):
        """Simulate sudden thermal shock or mechanical bias shift."""
        self.accel_bias += np.array(accel_delta, dtype=np.float64)
        self.gyro_bias += np.array(gyro_delta, dtype=np.float64)
        
    def get_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.accel_bias.copy(), self.gyro_bias.copy()

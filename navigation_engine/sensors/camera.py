"""
Visual Odometry (Camera) Sensor Model.
Simulates optical flow, keypoint feature matching, scale drift, and tracking confidence.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisualOdometryReading:
    timestamp: float
    velocity_body: np.ndarray       # Relative body-frame velocity (m/s), shape (3,)
    displacement_body: np.ndarray   # Relative displacement over dt (m), shape (3,)
    confidence: float               # Quality metric [0.0, 1.0]
    inlier_count: int
    covariance: np.ndarray          # 3x3 covariance matrix
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "velocity_body": [float(x) for x in self.velocity_body],
            "displacement_body": [float(x) for x in self.displacement_body],
            "confidence": float(self.confidence),
            "inlier_count": int(self.inlier_count),
            "cov_diag": [float(self.covariance[i, i]) for i in range(3)],
            "is_valid": self.is_valid
        }


class VisualOdometrySensor:
    def __init__(
        self,
        base_velocity_std: float = 0.08,    # m/s
        scale_drift_rate: float = 0.005,    # 0.5% scale drift
        seed: Optional[int] = None
    ):
        self.base_velocity_std = base_velocity_std
        self.scale_factor = 1.0
        self.scale_drift_rate = scale_drift_rate
        self.tracking_loss = False
        self.rng = np.random.RandomState(seed)

    def set_tracking_loss(self, loss: bool):
        self.tracking_loss = loss

    def step(
        self,
        timestamp: float,
        dt: float,
        true_velocity_body: np.ndarray
    ) -> VisualOdometryReading:
        if self.tracking_loss:
            return VisualOdometryReading(
                timestamp=timestamp,
                velocity_body=np.zeros(3),
                displacement_body=np.zeros(3),
                confidence=0.0,
                inlier_count=0,
                covariance=np.eye(3) * 1e4,
                is_valid=False
            )

        # Scale factor random walk
        self.scale_factor += self.rng.normal(0, self.scale_drift_rate * np.sqrt(dt))
        self.scale_factor = np.clip(self.scale_factor, 0.9, 1.1)

        # Inlier features & confidence
        inliers = int(self.rng.normal(250, 30))
        confidence = float(np.clip(inliers / 300.0, 0.3, 1.0))

        # Effective noise scaled by confidence
        eff_std = self.base_velocity_std / confidence
        vel_noise = self.rng.normal(0, eff_std, 3)

        meas_vel = true_velocity_body * self.scale_factor + vel_noise
        meas_disp = meas_vel * dt
        cov = np.eye(3) * (eff_std ** 2)

        return VisualOdometryReading(
            timestamp=timestamp,
            velocity_body=meas_vel,
            displacement_body=meas_disp,
            confidence=confidence,
            inlier_count=inliers,
            covariance=cov,
            is_valid=True
        )

"""
Navigation State representation and Coordinate/Quaternion Mathematics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [qw, qx, qy, qz]."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q * r."""
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw
    ], dtype=np.float64)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix C_b^n."""
    q = quat_normalize(q)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float64)


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return quat_normalize(np.array([qw, qx, qy, qz]))


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw) in degrees.
    ZYX rotation convention (Yaw -> Pitch -> Roll).
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw) % 360.0)


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert Euler angles (degrees) to unit quaternion [qw, qx, qy, qz]."""
    r = np.radians(roll_deg) / 2.0
    p = np.radians(pitch_deg) / 2.0
    y = np.radians(yaw_deg) / 2.0

    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    cy = np.cos(y)
    sy = np.sin(y)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return quat_normalize(np.array([qw, qx, qy, qz]))


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric matrix [v]x."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)


@dataclass
class NavigationState:
    """
    Complete Navigation State vector and associated covariance.
    State representation:
      - Position: [North, East, Down] (meters)
      - Velocity: [v_North, v_East, v_Down] (m/s)
      - Quaternion: [qw, qx, qy, qz]
      - Accel Bias: [ba_x, ba_y, ba_z] (m/s^2)
      - Gyro Bias: [bg_x, bg_y, bg_z] (rad/s)
      - Covariance P: 15x15 error state covariance
    """
    timestamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(15, dtype=np.float64) * 0.1)
    mode: str = "ai_enhanced_ekf"
    gnss_valid: bool = True

    def get_euler(self) -> Tuple[float, float, float]:
        """Returns (roll, pitch, yaw) in degrees."""
        return quat_to_euler(self.quaternion)

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns 3x3 rotation matrix C_b^n."""
        return quat_to_rotation_matrix(self.quaternion)

    def clone(self) -> "NavigationState":
        return NavigationState(
            timestamp=float(self.timestamp),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            covariance=self.covariance.copy(),
            mode=str(self.mode),
            gnss_valid=bool(self.gnss_valid)
        )

    def to_dict(self) -> Dict[str, Any]:
        roll, pitch, yaw = self.get_euler()
        return {
            "timestamp": float(self.timestamp),
            "position": [float(x) for x in self.position],
            "velocity": [float(x) for x in self.velocity],
            "quaternion": [float(x) for x in self.quaternion],
            "euler": {
                "roll": round(roll, 2),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2)
            },
            "accel_bias": [float(x) for x in self.accel_bias],
            "gyro_bias": [float(x) for x in self.gyro_bias],
            "cov_diag": [float(self.covariance[i, i]) for i in range(15)],
            "mode": self.mode,
            "gnss_valid": self.gnss_valid
        }

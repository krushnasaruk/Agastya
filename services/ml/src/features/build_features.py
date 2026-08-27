"""
Feature Engineering and Signal Transformations for Inertial Navigation.
Extracts temporal statistics, frequency descriptors, jerk, and coordinate transformations.
"""

from typing import Dict, Any
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [qw, qx, qy, qz]."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix C_b^n."""
    q = quat_normalize(q)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float64)


def extract_window_features(imu_window: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Extract high-dimensional feature descriptor from an IMU window (W, 6).
    Returns 1D feature vector.
    """
    acc = imu_window[:, 0:3]
    gyro = imu_window[:, 3:6]

    # 1. Statistical Moments
    acc_mean = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    acc_min = np.min(acc, axis=0)
    acc_max = np.max(acc, axis=0)

    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)
    gyro_min = np.min(gyro, axis=0)
    gyro_max = np.max(gyro, axis=0)

    # 2. Magnitudes & Norms
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    norm_features = np.array([
        np.mean(acc_norm), np.std(acc_norm),
        np.mean(gyro_norm), np.std(gyro_norm)
    ])

    # 3. Jerk (da/dt) & Angular Acceleration (dw/dt)
    jerk = np.diff(acc, axis=0) / dt
    jerk_std = np.std(jerk, axis=0) if len(jerk) > 0 else np.zeros(3)

    ang_acc = np.diff(gyro, axis=0) / dt
    ang_acc_std = np.std(ang_acc, axis=0) if len(ang_acc) > 0 else np.zeros(3)

    # 4. Concatenate
    features = np.concatenate([
        acc_mean, acc_std, acc_min, acc_max,
        gyro_mean, gyro_std, gyro_min, gyro_max,
        norm_features,
        jerk_std, ang_acc_std
    ])

    return features.astype(np.float32)


def remove_gravity_body(accel_meas: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
    """
    Subtract gravity vector projected into body frame.
    """
    C_b_n = quat_to_rotation_matrix(orientation_quat)
    g_n = np.array([0.0, 0.0, 9.80665])
    g_b = C_b_n.T @ g_n
    return accel_meas - g_b

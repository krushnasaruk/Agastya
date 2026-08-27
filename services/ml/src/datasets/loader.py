"""
PyTorch Dataset and DataLoader for Sequential IMU Dead Reckoning.
Supports sliding-window raw 6-DOF IMU sequences mapped to ground truth velocity vectors.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class IMUDataset(Dataset):
    """
    Sliding window dataset for IMU measurements.
    Features X: (Window_Size, 6) -> [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Target Y: (3,) -> Body-frame velocity vector [v_x, v_y, v_z]
    Target Bias (Optional): (3,) -> [ba_x, ba_y, ba_z]
    """
    def __init__(
        self,
        imu_readings: np.ndarray,      # Shape (N, 6)
        ground_truth_vel: np.ndarray,  # Shape (N, 3)
        window_size: int = 100,        # 1.0s window at 100Hz
        step_size: int = 10,           # Stride
        accel_biases: Optional[np.ndarray] = None
    ):
        self.window_size = window_size
        self.step_size = step_size
        
        self.imu_readings = imu_readings.astype(np.float32)
        self.ground_truth_vel = ground_truth_vel.astype(np.float32)
        self.accel_biases = accel_biases.astype(np.float32) if accel_biases is not None else None

        self.samples = []
        n_total = len(self.imu_readings)
        for i in range(0, n_total - window_size + 1, step_size):
            self.samples.append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.samples[idx]
        end_idx = start_idx + self.window_size

        x_window = self.imu_readings[start_idx:end_idx]  # (W, 6)
        y_vel = self.ground_truth_vel[end_idx - 1]        # (3,)

        item = {
            "imu": torch.from_numpy(x_window),
            "velocity": torch.from_numpy(y_vel)
        }

        if self.accel_biases is not None:
            item["bias"] = torch.from_numpy(self.accel_biases[end_idx - 1])

        return item


def generate_synthetic_flight_dataset(
    num_trajectories: int = 10,
    trajectory_duration_sec: float = 30.0,
    dt: float = 0.01,
    window_size: int = 100,
    step_size: int = 10
) -> Tuple[IMUDataset, IMUDataset]:
    """
    Generate rich synthetic 3D kinematic trajectories (climbing turns, figure-8s, evasive maneuvers)
    with realistic IMU noise for training and validation.
    """
    all_imu = []
    all_vel = []
    all_bias = []

    steps = int(trajectory_duration_sec / dt)
    rng = np.random.RandomState(1337)

    for _ in range(num_trajectories):
        # Maneuver parameters
        speed = rng.uniform(10.0, 30.0)
        freq_turn = rng.uniform(0.05, 0.2)
        freq_pitch = rng.uniform(0.02, 0.1)
        climb_rate = rng.uniform(-3.0, 5.0)

        acc_bias = rng.normal(0, 0.03, 3)
        gyro_bias = rng.normal(0, 0.003, 3)

        for step in range(steps):
            t = step * dt

            # Attitude rates
            yaw_rate = np.sin(2 * np.pi * freq_turn * t) * 0.3
            pitch_rate = np.cos(2 * np.pi * freq_pitch * t) * 0.15
            roll_rate = yaw_rate * 0.5

            # Body velocity
            v_x = speed + np.sin(0.5 * t) * 2.0
            v_y = np.sin(t) * 0.5
            v_z = climb_rate + np.cos(0.3 * t)

            # Specific force in body frame (centripetal + gravity reaction)
            f_x = np.cos(t) * 0.5
            f_y = speed * yaw_rate
            f_z = -9.80665 - speed * pitch_rate

            # Add sensor noise & bias
            acc_meas = np.array([f_x, f_y, f_z]) + acc_bias + rng.normal(0, 0.05, 3)
            gyro_meas = np.array([roll_rate, pitch_rate, yaw_rate]) + gyro_bias + rng.normal(0, 0.005, 3)

            all_imu.append(np.concatenate([acc_meas, gyro_meas]))
            all_vel.append([v_x, v_y, v_z])
            all_bias.append(acc_bias)

    imu_arr = np.array(all_imu)
    vel_arr = np.array(all_vel)
    bias_arr = np.array(all_bias)

    # 80/20 train/val split
    split_idx = int(0.8 * len(imu_arr))
    train_ds = IMUDataset(imu_arr[:split_idx], vel_arr[:split_idx], window_size, step_size, bias_arr[:split_idx])
    val_ds = IMUDataset(imu_arr[split_idx:], vel_arr[split_idx:], window_size, step_size, bias_arr[split_idx:])

    return train_ds, val_ds

"""
High-Fidelity 3D Trajectory and Multi-Sensor Simulator.
Generates ground-truth flight & ground vehicle dynamics, along with realistic noisy
and degraded sensor observations (IMU, GNSS, Visual Odometry).
"""

import sys
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAV_ENGINE_DIR = os.path.join(BASE_DIR, "services", "navigation-engine")
for p in [BASE_DIR, NAV_ENGINE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from navigation_engine.sensors.imu import IMUSensor, IMUReading
from navigation_engine.sensors.gnss import GNSSReceiver, GNSSReading, GNSSFixType
from navigation_engine.sensors.camera import VisualOdometrySensor, VisualOdometryReading
from navigation_engine.estimation.state import (
    NavigationState,
    euler_to_quat,
    quat_to_euler,
    quat_to_rotation_matrix,
    quat_multiply
)
from navigation_engine.fusion.sensor_fusion import SensorFusionEngine


@dataclass
class SimulationFrame:
    timestamp: float
    true_position: np.ndarray        # [North, East, Down] in meters (3,)
    true_velocity: np.ndarray        # [v_N, v_E, v_D] in m/s (3,)
    true_velocity_body: np.ndarray   # [v_x, v_y, v_z] in body frame (3,)
    true_orientation_quat: np.ndarray # [qw, qx, qy, qz] (4,)
    true_euler_deg: tuple            # (roll, pitch, yaw) in deg
    imu: IMUReading
    gnss: Optional[GNSSReading]
    vo: Optional[VisualOdometryReading]
    gnss_available: bool
    scenario_progress: float         # [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "true_position": [round(float(x), 3) for x in self.true_position],
            "true_velocity": [round(float(x), 3) for x in self.true_velocity],
            "true_velocity_body": [round(float(x), 3) for x in self.true_velocity_body],
            "true_euler": {
                "roll": round(self.true_euler_deg[0], 2),
                "pitch": round(self.true_euler_deg[1], 2),
                "yaw": round(self.true_euler_deg[2], 2)
            },
            "imu": self.imu.to_dict(),
            "gnss": self.gnss.to_dict() if self.gnss else None,
            "vo": self.vo.to_dict() if self.vo else None,
            "gnss_available": self.gnss_available,
            "scenario_progress": round(self.scenario_progress, 3)
        }


class TrajectorySimulator:
    def __init__(
        self,
        scenario_path: Optional[str] = None,
        dt: float = 0.01,
        seed: int = 42
    ):
        self.dt = dt
        self.time = 0.0
        self.step_count = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Default Scenario Configuration
        self.config = {
            "name": "default",
            "duration_sec": 45.0,
            "trajectory_type": "figure_8",
            "speed_mps": 20.0,
            "climb_rate_mps": 0.5,
            "gnss": {
                "enabled": True,
                "rate_hz": 5,
                "base_horizontal_std": 1.2,
                "base_vertical_std": 2.2,
                "multipath": False,
                "outages": []
            },
            "imu": {
                "accel_noise_std": 0.03,
                "gyro_noise_std": 0.002,
                "accel_bias": [0.02, -0.015, 0.025],
                "gyro_bias": [0.001, -0.002, 0.001]
            },
            "vo": {
                "enabled": True,
                "rate_hz": 20,
                "base_velocity_std": 0.08
            }
        }

        if scenario_path and os.path.exists(scenario_path):
            with open(scenario_path, "r") as f:
                loaded = json.load(f)
                self.config.update(loaded)

        self.duration_sec = self.config.get("duration_sec", 45.0)
        self.trajectory_type = self.config.get("trajectory_type", "figure_8")
        self.speed = self.config.get("speed_mps", 20.0)

        # Instantiate Sensor Models
        imu_cfg = self.config.get("imu", {})
        self.imu_sensor = IMUSensor(
            accel_noise_std=imu_cfg.get("accel_noise_std", 0.03),
            gyro_noise_std=imu_cfg.get("gyro_noise_std", 0.002),
            accel_init_bias=np.array(imu_cfg.get("accel_bias", [0.02, -0.01, 0.02])),
            gyro_init_bias=np.array(imu_cfg.get("gyro_bias", [0.001, -0.002, 0.001])),
            seed=seed
        )

        gnss_cfg = self.config.get("gnss", {})
        self.gnss_receiver = GNSSReceiver(
            base_horizontal_std=gnss_cfg.get("base_horizontal_std", 1.2),
            base_vertical_std=gnss_cfg.get("base_vertical_std", 2.2),
            seed=seed + 1
        )
        if gnss_cfg.get("multipath", False):
            self.gnss_receiver.set_multipath(True, bias_magnitude=12.0)

        vo_cfg = self.config.get("vo", {})
        self.vo_sensor = VisualOdometrySensor(
            base_velocity_std=vo_cfg.get("base_velocity_std", 0.08),
            seed=seed + 2
        )

        # Frequencies
        self.gnss_interval = max(1, int((1.0 / gnss_cfg.get("rate_hz", 5)) / self.dt))
        self.vo_interval = max(1, int((1.0 / vo_cfg.get("rate_hz", 20)) / self.dt))

        # Ground truth tracking
        self.current_pos_ned = np.array([0.0, 0.0, -100.0], dtype=np.float64)  # 100m altitude
        self.current_vel_ned = np.array([self.speed, 0.0, 0.0], dtype=np.float64)
        self.current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self):
        self.time = 0.0
        self.step_count = 0
        self.current_pos_ned = np.array([0.0, 0.0, -100.0], dtype=np.float64)
        self.current_vel_ned = np.array([self.speed, 0.0, 0.0], dtype=np.float64)
        self.current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _compute_analytical_pose(self, t: float) -> tuple:
        """Computes analytical position, velocity, and orientation at time t."""
        g = 9.80665

        if self.trajectory_type == "figure_8":
            T = 30.0
            omega = 2 * np.pi / T
            A = (self.speed / (2 * np.pi)) * T * 0.7

            x = A * np.sin(omega * t)
            y = A * np.sin(omega * t) * np.cos(omega * t)
            z = -100.0 - 0.5 * t

            vx = A * omega * np.cos(omega * t)
            vy = A * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
            vz = -0.5

            ax = -A * (omega**2) * np.sin(omega * t)
            ay = -4 * A * (omega**2) * np.sin(omega * t) * np.cos(omega * t)
            az = 0.0

            yaw = np.arctan2(vy, vx)
            horiz_speed = np.sqrt(vx**2 + vy**2)
            pitch = -np.arctan2(vz, max(horiz_speed, 1e-3))
            yaw_rate = (vx * ay - vy * ax) / max(horiz_speed**2, 1e-3)
            roll = np.arctan2(horiz_speed * yaw_rate, g)

        elif self.trajectory_type == "helical_climb":
            radius = 120.0
            omega = self.speed / radius
            climb = 2.5

            x = radius * np.sin(omega * t)
            y = radius * (1.0 - np.cos(omega * t))
            z = -50.0 - climb * t

            vx = radius * omega * np.cos(omega * t)
            vy = radius * omega * np.sin(omega * t)
            vz = -climb

            ax = -radius * (omega**2) * np.sin(omega * t)
            ay = radius * (omega**2) * np.cos(omega * t)
            az = 0.0

            yaw = np.arctan2(vy, vx)
            pitch = -np.arctan2(vz, self.speed)
            roll = np.arctan2(self.speed * omega, g)

        elif self.trajectory_type == "urban_grid":
            turn_interval = 12.0
            turn_duration = 3.0
            phase = t % turn_interval
            corner_idx = int(t / turn_interval) % 4
            base_heading = corner_idx * (np.pi / 2)

            if phase < (turn_interval - turn_duration):
                yaw = base_heading
                vx = self.speed * np.cos(yaw)
                vy = self.speed * np.sin(yaw)
                vz = 0.0
                ax, ay, az = 0.0, 0.0, 0.0
                roll, pitch = 0.0, 0.0
            else:
                turn_progress = (phase - (turn_interval - turn_duration)) / turn_duration
                yaw = base_heading + turn_progress * (np.pi / 2)
                yaw_rate = (np.pi / 2) / turn_duration
                vx = self.speed * np.cos(yaw)
                vy = self.speed * np.sin(yaw)
                vz = 0.0
                ax = -self.speed * yaw_rate * np.sin(yaw)
                ay = self.speed * yaw_rate * np.cos(yaw)
                az = 0.0
                roll, pitch = 0.0, 0.0

            x = self.current_pos_ned[0] + vx * self.dt
            y = self.current_pos_ned[1] + vy * self.dt
            z = 0.0

        else:
            x = self.speed * t
            y = 0.0
            z = -100.0
            vx, vy, vz = self.speed, 0.0, 0.0
            ax, ay, az = 0.0, 0.0, 0.0
            roll, pitch, yaw = 0.0, 0.0, 0.0

        pos_ned = np.array([x, y, z])
        vel_ned = np.array([vx, vy, vz])
        r_deg = float(np.degrees(roll))
        p_deg = float(np.degrees(pitch))
        y_deg = float(np.degrees(yaw))
        quat = euler_to_quat(r_deg, p_deg, y_deg)

        return pos_ned, vel_ned, np.array([ax, ay, az]), quat, (r_deg, p_deg, y_deg)

    def _compute_ground_truth(self, t: float) -> tuple:
        """
        Calculates analytical 3D trajectory dynamics and exact body sensor forces.
        Returns: (pos_ned, vel_ned, vel_body, quat, euler_deg, accel_body, gyro_body)
        """
        g = 9.80665
        pos_ned, vel_ned, accel_ned, quat, (r_deg, p_deg, y_deg) = self._compute_analytical_pose(t)
        C_b_n = quat_to_rotation_matrix(quat)

        # Body frame velocity
        vel_body = C_b_n.T @ vel_ned

        # Body frame specific force (f_b = C_n_b * (a_n - g_n))
        g_n = np.array([0.0, 0.0, g])
        accel_body = C_b_n.T @ (accel_ned - g_n)

        # Exact Body Angular Rate via quaternion derivative
        h = 1e-4
        _, _, _, q_plus, _ = self._compute_analytical_pose(t + h)
        _, _, _, q_minus, _ = self._compute_analytical_pose(t - h)

        if np.dot(q_plus, q_minus) < 0:
            q_plus = -q_plus
        quat_eval = -quat if np.dot(quat, q_minus) < 0 else quat

        q_dot = (q_plus - q_minus) / (2.0 * h)
        q_conj = np.array([quat_eval[0], -quat_eval[1], -quat_eval[2], -quat_eval[3]])
        omega_quat = 2.0 * quat_multiply(q_conj, q_dot)
        gyro_body = omega_quat[1:4]

        return pos_ned, vel_ned, vel_body, quat, (r_deg, p_deg, y_deg), accel_body, gyro_body

    def is_gnss_jammed(self, t: float) -> bool:
        """Check if current timestamp falls within a programmed GNSS outage window."""
        gnss_cfg = self.config.get("gnss", {})
        if not gnss_cfg.get("enabled", True):
            return True

        outages = gnss_cfg.get("outages", [])
        for outg in outages:
            st = outg.get("start_time", 0.0)
            et = outg.get("end_time", 0.0)
            if st <= t <= et:
                return True
        return False

    def step(self) -> SimulationFrame:
        """Advance simulation by dt seconds and return full frame."""
        t = self.time
        pos_ned, vel_ned, vel_body, quat, euler_deg, acc_body, gyro_body = self._compute_ground_truth(t)

        self.current_pos_ned = pos_ned
        self.current_vel_ned = vel_ned
        self.current_quat = quat

        # 1. IMU Reading (100Hz)
        imu_reading = self.imu_sensor.step(t, self.dt, acc_body, gyro_body)

        # 2. GNSS Reading (5Hz)
        gnss_reading = None
        jammed = self.is_gnss_jammed(t)
        self.gnss_receiver.set_jamming(jammed)

        if self.step_count % self.gnss_interval == 0:
            gnss_reading = self.gnss_receiver.step(t, pos_ned, vel_ned)

        # 3. Visual Odometry Reading (20Hz)
        vo_reading = None
        if self.step_count % self.vo_interval == 0:
            vo_reading = self.vo_sensor.step(t, self.vo_interval * self.dt, vel_body)

        frame = SimulationFrame(
            timestamp=t,
            true_position=pos_ned,
            true_velocity=vel_ned,
            true_velocity_body=vel_body,
            true_orientation_quat=quat,
            true_euler_deg=euler_deg,
            imu=imu_reading,
            gnss=gnss_reading,
            vo=vo_reading,
            gnss_available=not jammed,
            scenario_progress=min(1.0, t / max(self.duration_sec, 1.0))
        )

        self.time += self.dt
        self.step_count += 1
        return frame

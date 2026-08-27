"""
Sensor Fusion Engine.
Orchestrates multi-rate asynchronous sensor integration:
- 100Hz IMU for ES-EKF propagation
- 5Hz GNSS for global position/velocity correction
- 20Hz Visual Odometry
- 20Hz AI Neural Inertial velocity corrections
- Comparative Pure Dead Reckoning baseline
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from ..sensors.imu import IMUReading
from ..sensors.gnss import GNSSReading, GNSSFixType
from ..sensors.camera import VisualOdometryReading
from ..estimation.state import NavigationState
from ..estimation.dead_reckoning import StrapdownDeadReckoning
from ..estimation.kalman import ErrorStateKalmanFilter
from ..correction.drift_correction import DriftCorrector


class SensorFusionEngine:
    def __init__(
        self,
        mode: str = "ai_enhanced_ekf",
        enable_zupt: bool = True,
        ai_velocity_noise_std: float = 0.25  # m/s
    ):
        self.mode = mode
        self.enable_zupt = enable_zupt
        self.ai_velocity_noise_std = ai_velocity_noise_std

        # Estimation Cores
        self.filter = ErrorStateKalmanFilter()
        self.pure_dr_engine = StrapdownDeadReckoning()
        self.drift_corrector = DriftCorrector()

        # States
        self.fused_state = NavigationState(mode=self.mode)
        self.pure_dr_state = NavigationState(mode="pure_dr")
        
        self.last_imu_time: Optional[float] = None
        self.last_gnss_fix_time: Optional[float] = None
        self.gnss_available = True
        self.total_distance_travelled = 0.0

    def reset(self, initial_state: Optional[NavigationState] = None):
        """Reset filter and baseline states."""
        if initial_state is not None:
            self.fused_state = initial_state.clone()
            self.fused_state.mode = self.mode
            self.pure_dr_state = initial_state.clone()
            self.pure_dr_state.mode = "pure_dr"
        else:
            self.fused_state = NavigationState(mode=self.mode)
            self.pure_dr_state = NavigationState(mode="pure_dr")

        self.last_imu_time = None
        self.last_gnss_fix_time = None
        self.gnss_available = True
        self.total_distance_travelled = 0.0

    def set_mode(self, mode: str):
        """Switch navigation mode."""
        self.mode = mode
        self.fused_state.mode = mode

    def process_imu(self, imu: IMUReading) -> NavigationState:
        """
        Process high-frequency (100Hz+) IMU reading.
        Propagates both fused ES-EKF state and uncorrected baseline SINS state.
        """
        if self.last_imu_time is None:
            self.last_imu_time = imu.timestamp
            self.fused_state.timestamp = imu.timestamp
            self.pure_dr_state.timestamp = imu.timestamp
            return self.fused_state

        dt = float(imu.timestamp - self.last_imu_time)
        if dt <= 0.0 or dt > 0.5:
            dt = 0.01  # Fallback default

        self.last_imu_time = imu.timestamp

        # 1. Update drift corrector window
        self.drift_corrector.update_sensor_window(imu.accel, imu.gyro)

        # 2. Propagate Pure Dead Reckoning baseline (never corrected)
        self.pure_dr_state = self.pure_dr_engine.step(
            self.pure_dr_state, imu.accel, imu.gyro, dt
        )

        # 3. Propagate Fused State via ES-EKF
        if self.mode == "pure_dr":
            self.fused_state = self.pure_dr_engine.step(
                self.fused_state, imu.accel, imu.gyro, dt
            )
        else:
            self.fused_state = self.filter.predict(
                self.fused_state, imu.accel, imu.gyro, dt
            )

        # 4. Check for Zero Velocity (ZUPT)
        if self.enable_zupt and self.mode in ["ai_enhanced_ekf", "classical_ekf"]:
            is_stat, _ = self.drift_corrector.detect_zero_velocity()
            if is_stat:
                R_zupt = np.eye(3) * 1e-4  # Very confident zero velocity
                self.fused_state, _ = self.filter.update_velocity(
                    self.fused_state, np.zeros(3), R_zupt
                )

        # Track cumulative distance
        vel_mag = float(np.linalg.norm(self.fused_state.velocity))
        self.total_distance_travelled += vel_mag * dt

        # Update GNSS timeout (if no fix received in last 0.35s, declare GNSS unavailable)
        if self.last_gnss_fix_time is not None:
            if (imu.timestamp - self.last_gnss_fix_time) > 0.35:
                self.gnss_available = False
                self.fused_state.gnss_valid = False

        return self.fused_state

    def process_gnss(self, gnss: GNSSReading) -> Tuple[NavigationState, bool]:
        """
        Process GNSS position & velocity fix (5Hz).
        """
        self.gnss_available = gnss.is_valid and (gnss.fix_type >= GNSSFixType.FIX_3D)
        self.fused_state.gnss_valid = self.gnss_available

        if self.gnss_available:
            self.last_gnss_fix_time = gnss.timestamp

        if self.mode == "pure_dr" or not self.gnss_available:
            return self.fused_state, False

        # Apply GNSS PVA measurement update
        R_pos = gnss.covariance
        R_vel = np.eye(3) * (0.2 ** 2)

        self.fused_state, mahalanobis_sq = self.filter.update_gnss_pva(
            self.fused_state, gnss.position, gnss.velocity, R_pos, R_vel
        )

        return self.fused_state, True

    def process_visual_odometry(self, vo: VisualOdometryReading) -> Tuple[NavigationState, bool]:
        """
        Process Visual Odometry body-frame velocity (20Hz).
        """
        if self.mode == "pure_dr" or not vo.is_valid:
            return self.fused_state, False

        # Transform body velocity to navigation NED frame: v_ned = C_b_n * v_b
        C_b_n = self.fused_state.get_rotation_matrix()
        v_ned = C_b_n @ vo.velocity_body

        # Transform covariance: R_ned = C_b_n * R_b * C_b_n^T
        R_ned = C_b_n @ vo.covariance @ C_b_n.T

        self.fused_state, _ = self.filter.update_velocity(
            self.fused_state, v_ned, R_ned
        )
        return self.fused_state, True

    def process_ai_velocity(
        self,
        predicted_velocity_body: np.ndarray,
        confidence: float = 0.95,
        force: bool = False
    ) -> Tuple[NavigationState, bool]:
        """
        Process AI Neural Inertial inferred velocity vector during GNSS degradation.
        """
        if self.mode not in ["ai_enhanced_ekf", "ai_only"]:
            return self.fused_state, False

        # Only apply heavy AI correction when GNSS is lost/degraded or forced
        if self.gnss_available and not force and self.mode != "ai_only":
            return self.fused_state, False

        # Body to NED frame
        C_b_n = self.fused_state.get_rotation_matrix()
        v_ned = C_b_n @ predicted_velocity_body

        eff_noise = (self.ai_velocity_noise_std / max(confidence, 0.1)) ** 2
        R_ai = np.eye(3) * eff_noise

        self.fused_state, _ = self.filter.update_velocity(
            self.fused_state, v_ned, R_ai
        )
        return self.fused_state, True

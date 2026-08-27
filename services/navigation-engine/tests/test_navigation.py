"""
Unit and Integration Tests for AGASTYA Navigation Engine.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.estimation.state import (
    NavigationState,
    quat_normalize,
    quat_multiply,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    quat_to_euler,
    euler_to_quat
)
from src.estimation.dead_reckoning import StrapdownDeadReckoning
from src.estimation.kalman import ErrorStateKalmanFilter
from src.sensors.imu import IMUSensor
from src.sensors.gnss import GNSSReceiver, GNSSFixType
from src.sensors.camera import VisualOdometrySensor
from src.correction.drift_correction import DriftCorrector
from src.fusion.sensor_fusion import SensorFusionEngine


def test_quaternion_math():
    # Unit normalization
    q = np.array([3.0, 4.0, 0.0, 0.0])
    q_norm = quat_normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)

    # Identity rotation
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_rotation_matrix(q_id)
    assert np.allclose(R_id, np.eye(3))

    # Euler conversions roundtrip
    roll, pitch, yaw = 15.0, -25.0, 130.0
    q_rot = euler_to_quat(roll, pitch, yaw)
    r_out, p_out, y_out = quat_to_euler(q_rot)
    assert np.isclose(roll, r_out, atol=1e-3)
    assert np.isclose(pitch, p_out, atol=1e-3)
    assert np.isclose(yaw, y_out, atol=1e-3)


def test_strapdown_stationary_gravity_cancellation():
    sins = StrapdownDeadReckoning(gravity_accel=9.80665)
    state = NavigationState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0])
    )

    # Stationary accelerometer measures upward reaction force: [0, 0, -9.80665]
    accel_meas = np.array([0.0, 0.0, -9.80665])
    gyro_meas = np.zeros(3)

    next_state = sins.step(state, accel_meas, gyro_meas, dt=0.01)

    # After 1 step, velocity should remain approximately zero
    assert np.allclose(next_state.velocity, np.zeros(3), atol=1e-4)
    assert np.allclose(next_state.position, np.zeros(3), atol=1e-4)


def test_esekf_covariance_and_update():
    ekf = ErrorStateKalmanFilter()
    state = NavigationState(
        position=np.array([10.0, 20.0, -30.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        covariance=np.eye(15) * 1.0
    )

    # Prediction step
    accel = np.array([0.0, 0.0, -9.80665])
    gyro = np.zeros(3)
    pred_state = ekf.predict(state, accel, gyro, dt=0.1)

    # Covariance should grow with process noise
    assert pred_state.covariance[0, 0] > state.covariance[0, 0]

    # GNSS Measurement update
    gnss_pos = np.array([10.2, 19.8, -30.1])
    gnss_vel = np.array([5.05, 0.02, 0.01])
    R_pos = np.eye(3) * 0.5
    R_vel = np.eye(3) * 0.1

    upd_state, mahalanobis = ekf.update_gnss_pva(pred_state, gnss_pos, gnss_vel, R_pos, R_vel)

    # Covariance should contract after valid measurement update
    assert upd_state.covariance[0, 0] < pred_state.covariance[0, 0]
    assert np.all(np.linalg.eigvals(upd_state.covariance) >= 0)  # Positive semi-definite


def test_zupt_detector():
    corrector = DriftCorrector(zupt_window_size=5)

    # Simulate 10 stationary samples
    for _ in range(10):
        acc = np.array([0.0, 0.0, -9.81]) + np.random.normal(0, 0.01, 3)
        gyro = np.random.normal(0, 0.005, 3)
        corrector.update_sensor_window(acc, gyro)

    is_stat, metric = corrector.detect_zero_velocity()
    assert is_stat is True

    # Simulate active motion
    for _ in range(10):
        acc = np.array([5.0, 2.0, -9.81]) + np.random.normal(0, 0.5, 3)
        gyro = np.array([0.5, -0.2, 0.8])
        corrector.update_sensor_window(acc, gyro)

    is_stat, metric = corrector.detect_zero_velocity()
    assert is_stat is False


def test_sensor_fusion_engine_pipeline():
    fusion = SensorFusionEngine(mode="ai_enhanced_ekf")
    imu = IMUSensor(seed=1)
    gnss = GNSSReceiver(seed=1)

    # Step IMU
    imu_pkt = imu.step(0.01, 0.01, np.array([0.0, 0.0, -9.80665]), np.zeros(3))
    st = fusion.process_imu(imu_pkt)
    assert st is not None

    # Step GNSS
    gnss_pkt = gnss.step(0.01, np.zeros(3), np.zeros(3))
    st, accepted = fusion.process_gnss(gnss_pkt)
    assert accepted is True
    assert st.gnss_valid is True

    # Test Jamming
    gnss.set_jamming(True)
    gnss_pkt_jam = gnss.step(0.02, np.zeros(3), np.zeros(3))
    st, accepted = fusion.process_gnss(gnss_pkt_jam)
    assert accepted is False
    assert st.gnss_valid is False

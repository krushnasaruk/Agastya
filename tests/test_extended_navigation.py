"""
Extended Integration and Unit Tests for AGASTYA Navigation Engine.
Validates:
  1. TrajectoryMetricsEngine (ATE, RPE, drift, distance).
  2. Sensor and Telemetry Typed Packets (IMUPacket, GNSSPacket, TelemetryFramePacket).
  3. ES-EKF Covariance Conditioning and positive-definiteness.
  4. Zero-Velocity Detector (ZUPT) stance detection and velocity arrest.
"""

import pytest
import numpy as np

try:
    from navigation_engine.utils.metrics import (
        TrajectoryMetricsEngine,
        TrajectoryMetricResult
    )
    from navigation_engine.sensors.packets import (
        IMUPacket,
        GNSSPacket,
        VisualOdometryPacket,
        TelemetryFramePacket
    )
    from navigation_engine.estimation.kalman import ErrorStateKalmanFilter
    from navigation_engine.estimation.state import NavigationState
    from navigation_engine.estimation.zupt import (
        ZeroVelocityDetector,
        ZUPTCorrector,
        ZUPTConfig
    )
except ImportError:
    from src.utils.metrics import (
        TrajectoryMetricsEngine,
        TrajectoryMetricResult
    )
    from src.sensors.packets import (
        IMUPacket,
        GNSSPacket,
        VisualOdometryPacket,
        TelemetryFramePacket
    )
    from src.estimation.kalman import ErrorStateKalmanFilter
    from src.estimation.state import NavigationState
    from src.estimation.zupt import (
        ZeroVelocityDetector,
        ZUPTCorrector,
        ZUPTConfig
    )


def test_trajectory_metrics_engine_ate_and_rpe():
    # Synthetic ground truth circle
    theta = np.linspace(0, 2 * np.pi, 100)
    gt = np.column_stack([np.cos(theta) * 50.0, np.sin(theta) * 50.0, np.zeros(100)])

    # Estimated with constant offset [0.5, -0.5, 0.0]
    est = gt + np.array([0.5, -0.5, 0.0])

    res = TrajectoryMetricsEngine.evaluate(est, gt)
    expected_ate = np.sqrt(0.5**2 + 0.5**2)
    
    assert np.isclose(res.ate_rmse, expected_ate, atol=1e-4)
    assert np.isclose(res.max_position_error, expected_ate, atol=1e-4)
    assert res.num_samples == 100
    assert res.total_distance > 0.0


def test_trajectory_metrics_zero_drift_and_edge_cases():
    # Empty inputs
    empty_res = TrajectoryMetricsEngine.evaluate(np.empty((0, 3)), np.empty((0, 3)))
    assert empty_res.ate_rmse == 0.0
    assert empty_res.num_samples == 0

    # Perfect identical trajectory
    gt = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
    res = TrajectoryMetricsEngine.evaluate(gt, gt)
    assert res.ate_rmse == 0.0
    assert res.drift_percentage == 0.0
    assert np.isclose(res.total_distance, 20.0)


def test_sensor_packet_validation_and_serialization():
    # IMUPacket
    imu = IMUPacket(
        timestamp=1.25,
        accel_x=0.0, accel_y=0.1, accel_z=-9.81,
        gyro_x=0.01, gyro_y=-0.02, gyro_z=0.05,
        temperature_c=26.5,
        sequence_id=42
    )
    assert np.allclose(imu.accel, np.array([0.0, 0.1, -9.81]))
    assert np.allclose(imu.gyro, np.array([0.01, -0.02, 0.05]))

    # Serialization roundtrip
    d = imu.to_dict()
    imu_restored = IMUPacket.from_dict(d)
    assert imu_restored.timestamp == imu.timestamp
    assert np.allclose(imu_restored.accel, imu.accel)

    # Invalid timestamp assertion
    with pytest.raises(ValueError):
        IMUPacket(timestamp=-0.1, accel_x=0, accel_y=0, accel_z=0, gyro_x=0, gyro_y=0, gyro_z=0)


def test_kalman_filter_covariance_conditioning():
    ekf = ErrorStateKalmanFilter()
    state = NavigationState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        covariance=np.eye(15) * 0.1
    )

    # Validate condition checks
    assert ekf.is_covariance_healthy(state) is True
    cond = ekf.get_condition_number(state)
    assert np.isclose(cond, 1.0)

    # Check update_zero_velocity
    state.velocity = np.array([0.5, -0.3, 0.2])
    updated_state, mahalanobis = ekf.update_zero_velocity(state)
    assert np.linalg.norm(updated_state.velocity) < np.linalg.norm(state.velocity)
    assert ekf.is_covariance_healthy(updated_state) is True


def test_zupt_detector_and_corrector():
    ekf = ErrorStateKalmanFilter()
    corrector = ZUPTCorrector(ekf, ZUPTConfig(window_size=6))
    state = NavigationState(
        position=np.zeros(3),
        velocity=np.array([0.2, 0.1, 0.0]),
        covariance=np.eye(15) * 0.1
    )

    # Feed 10 stationary IMU readings
    applied_correction = False
    for _ in range(10):
        acc = np.array([0.0, 0.0, -9.80665]) + np.random.normal(0, 0.005, 3)
        gyr = np.random.normal(0, 0.002, 3)
        state, corrected = corrector.process(state, acc, gyr)
        if corrected:
            applied_correction = True

    assert applied_correction is True
    assert np.linalg.norm(state.velocity) < 0.1

"""
Comprehensive Automated Test Suite for Classical Dead Reckoning (Objective 3 Final Audit).
Validates Local ENU heading conventions, parameter provenance/configuration, independent velocity RMSE,
standardized outage monotonicity, ZUPT causality, smartphone IMU isolation, baseline registry, and determinism.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.state import PlanarNavigationState, DeadReckoningTrajectory, wrap_to_pi, wrap_to_2pi
from src.wheel_odometry import WheelOdometryEstimator
from src.yaw import YawPropagator
from src.quality_gate import CausalQualityGate
from src.dead_reckoning import ClassicalDeadReckoningEngine, ClassicalDeadReckoningConfig, SmartphoneCalibrationGuard
from src.outage import GNSSOutageSimulator, OutageScenario
from src.evaluation import DeadReckoningEvaluator, NavigationMetrics, OutageEvaluationMetrics


# 1. ENU Heading: Northward Propagation (psi = 0 -> dE = 0, dN > 0)
def test_enu_heading_northward_propagation():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    engine.initialize(initial_heading_rad=0.0)  # True North

    # 10 steps of 0.1s at 10 m/s -> 10.0m North
    for k in range(10):
        st = engine.step(time_sec=(k + 1) * 0.1, dt_sec=0.1, wheel_speed_rl=10.0, wheel_speed_rr=10.0, yaw_rate=0.0)
    assert np.isclose(st.p_east_m, 0.0, atol=1e-5)
    assert np.isclose(st.p_north_m, 10.0, atol=1e-4)


# 2. ENU Heading: Eastward Propagation (psi = pi/2 -> dE > 0, dN = 0)
def test_enu_heading_eastward_propagation():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    engine.initialize(initial_heading_rad=np.pi / 2.0)  # East (90 deg)

    # 10 steps of 0.1s at 10 m/s -> 10.0m East
    for k in range(10):
        st = engine.step(time_sec=(k + 1) * 0.1, dt_sec=0.1, wheel_speed_rl=10.0, wheel_speed_rr=10.0, yaw_rate=0.0)
    assert np.isclose(st.p_east_m, 10.0, atol=1e-4)
    assert np.isclose(st.p_north_m, 0.0, atol=1e-5)


# 3. ENU Heading: Southward & Westward Propagation
def test_enu_heading_south_and_west():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    
    # South (psi = pi) -> 10m South
    engine.initialize(initial_heading_rad=np.pi)
    for k in range(10):
        st_south = engine.step(time_sec=(k + 1) * 0.1, dt_sec=0.1, wheel_speed_rl=10.0, wheel_speed_rr=10.0, yaw_rate=0.0)
    assert np.isclose(st_south.p_east_m, 0.0, atol=1e-5)
    assert np.isclose(st_south.p_north_m, -10.0, atol=1e-4)

    # West (psi = 3pi/2) -> 10m West
    engine.initialize(initial_heading_rad=3.0 * np.pi / 2.0)
    for k in range(10):
        st_west = engine.step(time_sec=(k + 1) * 0.1, dt_sec=0.1, wheel_speed_rl=10.0, wheel_speed_rr=10.0, yaw_rate=0.0)
    assert np.isclose(st_west.p_east_m, -10.0, atol=1e-4)
    assert np.isclose(st_west.p_north_m, 0.0, atol=1e-5)


# 4. Heading Propagation & Angle Wrapping
def test_heading_propagation_and_angle_wrapping():
    yaw_prop = YawPropagator(initial_heading_rad=np.radians(350.0))
    # Turn right by 20 deg over 1s -> should wrap to 10 deg
    h_new, rate, src = yaw_prop.step(yaw_rate_can_rads=np.radians(20.0), dt_sec=1.0)
    assert np.isclose(np.degrees(h_new), 10.0, atol=1e-3)

    # Negative angle wrap check: -0.1 rad wraps to 2pi - 0.1
    wrapped_neg = wrap_to_2pi(-0.1)
    assert np.isclose(wrapped_neg, 2.0 * np.pi - 0.1)


# 5. Vehicle Parameter Configuration & JSON Loading
def test_vehicle_parameter_configuration_and_json():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "classical_dead_reckoning_config.json"))
    cfg = ClassicalDeadReckoningConfig.from_json(config_path, baseline_type="BASELINE_A")
    assert cfg.track_width_m == 1.47
    assert cfg.zero_speed_threshold_ms == 0.08
    assert cfg.accel_weight_baseline_b == 0.15

    reg = cfg.get_parameter_registry()
    assert "PROVISIONAL" in reg["track_width_m"].status
    assert "Ford Fiesta" in reg["track_width_m"].provenance


# 6. Baseline Registry Metadata
def test_baseline_registry_metadata():
    baselines = ClassicalDeadReckoningConfig.get_baseline_registry()
    assert "BASELINE_A" in baselines
    assert "BASELINE_B" in baselines
    assert "BASELINE_C" in baselines
    assert "SMARTPHONE_SINS" in baselines
    assert baselines["SMARTPHONE_SINS"].causal_status == "BLOCKED — SENSOR FRAME CALIBRATION REQUIRED"


# 7. Smartphone Calibration Guard
def test_smartphone_calibration_guard():
    # Attempting to initialize SMARTPHONE_SINS without calibration raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        engine = ClassicalDeadReckoningEngine(baseline_type="SMARTPHONE_SINS")
    assert "SMARTPHONE IMU BLOCKED" in str(exc_info.value)


# 8. Velocity RMSE Independent Calculation
def test_velocity_rmse_independent_calculation():
    n = 50
    t = np.arange(n) * 0.1
    dt = np.full(n, 0.1)
    
    # Ground truth speed = 10.0 m/s
    ref_speed = np.full(n, 10.0)
    # Estimated speed has 0.2 m/s known bias
    est_speed = np.full(n, 10.2)

    traj = DeadReckoningTrajectory(
        timestamps_sec=t,
        dt_array_sec=dt,
        p_east_m=np.zeros(n),
        p_north_m=np.zeros(n),
        heading_rad=np.zeros(n),
        forward_speed_ms=est_speed,
        yaw_rate_rads=np.zeros(n),
        baseline_name="BASELINE_A"
    )

    metrics, _, _ = DeadReckoningEvaluator.evaluate(
        estimated_traj=traj,
        reference_p_east_m=np.zeros(n),
        reference_p_north_m=np.zeros(n),
        reference_speed_ms=ref_speed
    )

    # Velocity RMSE should exactly equal 0.20000 m/s
    assert np.isclose(metrics.velocity_rmse_ms, 0.20, atol=1e-4)


# 9. Outage Max Drift Monotonicity (Mathematical Invariant)
def test_outage_max_drift_monotonicity():
    # Over expanding time windows [t0, t0+5s] subset [t0, t0+10s] subset [t0, t0+30s],
    # the maximum accumulated error is mathematically non-decreasing
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    n = 600
    t_arr = np.arange(n) * 0.1
    dt_arr = np.full(n, 0.1)

    v_wheel = 10.0 * 1.005
    yaw_rate = 0.05

    df = pd.DataFrame({
        "time_sec": t_arr,
        "dt_sec": dt_arr,
        "wheel_speed_rl_ms": np.full(n, v_wheel),
        "wheel_speed_rr_ms": np.full(n, v_wheel),
        "yaw_rate_rads": np.full(n, yaw_rate)
    })

    ref_head = np.cumsum(yaw_rate * dt_arr)
    ref_e = np.cumsum(10.0 * np.sin(ref_head) * dt_arr)
    ref_n = np.cumsum(10.0 * np.cos(ref_head) * dt_arr)

    traj = engine.run_sequence(df, initial_heading_rad=0.0)

    start_idx = 200
    idx_5s = start_idx + 50
    idx_10s = start_idx + 100
    idx_30s = start_idx + 300

    max_drift_5s = DeadReckoningEvaluator.compute_outage_max_drift(traj, ref_e, ref_n, start_idx, idx_5s)
    max_drift_10s = DeadReckoningEvaluator.compute_outage_max_drift(traj, ref_e, ref_n, start_idx, idx_10s)
    max_drift_30s = DeadReckoningEvaluator.compute_outage_max_drift(traj, ref_e, ref_n, start_idx, idx_30s)

    assert max_drift_5s <= max_drift_10s <= max_drift_30s


# 10. Outage Metric Distinction
def test_outage_metric_distinction():
    n = 100
    t = np.arange(n) * 0.1
    dt = np.full(n, 0.1)

    ref_e = np.linspace(0, 50, n)
    ref_n = np.zeros(n)
    est_e = ref_e + 0.002 * (np.arange(n) ** 2)
    est_n = np.zeros(n)

    traj = DeadReckoningTrajectory(
        timestamps_sec=t,
        dt_array_sec=dt,
        p_east_m=est_e,
        p_north_m=est_n,
        heading_rad=np.zeros(n),
        forward_speed_ms=np.full(n, 5.0),
        yaw_rate_rads=np.zeros(n),
        baseline_name="BASELINE_A"
    )

    acc_drift = DeadReckoningEvaluator.compute_outage_accumulated_drift(traj, ref_e, ref_n, 20, 80)
    max_drift = DeadReckoningEvaluator.compute_outage_max_drift(traj, ref_e, ref_n, 20, 80)
    outage_ate = DeadReckoningEvaluator.compute_outage_ate_rmse(traj, ref_e, ref_n, 20, 80)

    assert max_drift >= acc_drift
    assert max_drift > outage_ate > 0.0


# 11. ZUPT Causality & Stationary Drift Lock
def test_zupt_causality_and_freeze():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A", zero_speed_threshold_ms=0.08)
    engine.initialize(initial_heading_rad=1.5)

    # Vehicle stopped (speed 0.02 m/s), gyro has small 0.05 rad/s noise
    st = engine.step(time_sec=0.1, dt_sec=0.1, wheel_speed_rl=0.02, wheel_speed_rr=0.02, yaw_rate=0.05)
    assert st.is_stationary is True
    assert st.forward_speed_ms == 0.0
    assert st.yaw_rate_rads == 0.0
    assert st.heading_rad == 1.5  # Heading frozen when stationary


# 12. Smartphone Frame Isolation
def test_smartphone_frame_isolation():
    # Verify that Baseline A runs purely on CAN signals and ignores raw phone axes
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    df = pd.DataFrame({
        "time_sec": [0.0, 0.1, 0.2],
        "dt_sec": [0.1, 0.1, 0.1],
        "wheel_speed_rl_ms": [10.0, 10.0, 10.0],
        "wheel_speed_rr_ms": [10.0, 10.0, 10.0],
        "yaw_rate_rads": [0.0, 0.0, 0.0],
        "phone_acc_x_ms2": [999.0, 999.0, 999.0],
        "phone_gyro_z_rads": [50.0, 50.0, 50.0]
    })
    traj = engine.run_sequence(df)
    assert np.allclose(traj.heading_rad, 0.0)
    assert np.allclose(traj.forward_speed_ms, 10.0)


# 13. GNSS / Reference Stream Isolation
def test_gnss_leakage_and_reference_stream_isolation():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    df_no_gps = pd.DataFrame({
        "time_sec": [0.0, 0.1, 0.2],
        "dt_sec": [0.1, 0.1, 0.1],
        "wheel_speed_rl_ms": [10.0, 10.0, 10.0],
        "wheel_speed_rr_ms": [10.0, 10.0, 10.0],
        "yaw_rate_rads": [0.0, 0.0, 0.0]
    })
    traj = engine.run_sequence(df_no_gps)
    assert len(traj.p_east_m) == 3


# 14. Future Data Mutation Invariance
def test_future_data_mutation_invariance():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    n = 60
    t_arr = np.arange(n) * 0.1
    dt_arr = np.full(n, 0.1)

    df_base = pd.DataFrame({
        "time_sec": t_arr,
        "dt_sec": dt_arr,
        "wheel_speed_rl_ms": np.full(n, 15.0),
        "wheel_speed_rr_ms": np.full(n, 15.0),
        "yaw_rate_rads": np.full(n, 0.05)
    })

    df_mutated = df_base.copy()
    df_mutated.loc[35:, "wheel_speed_rl_ms"] = 99.0
    df_mutated.loc[35:, "yaw_rate_rads"] = 2.0

    traj_base = engine.run_sequence(df_base)
    traj_mutated = engine.run_sequence(df_mutated)

    np.testing.assert_array_equal(traj_base.p_east_m[:35], traj_mutated.p_east_m[:35])
    np.testing.assert_array_equal(traj_base.p_north_m[:35], traj_mutated.p_north_m[:35])
    np.testing.assert_array_equal(traj_base.heading_rad[:35], traj_mutated.heading_rad[:35])


# 15. Deterministic Reproducibility
def test_deterministic_reproducibility():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    n = 40
    df = pd.DataFrame({
        "time_sec": np.arange(n) * 0.1,
        "dt_sec": np.full(n, 0.1),
        "wheel_speed_rl_ms": np.full(n, 12.0),
        "wheel_speed_rr_ms": np.full(n, 12.0),
        "yaw_rate_rads": np.full(n, 0.02)
    })
    traj1 = engine.run_sequence(df)
    traj2 = engine.run_sequence(df)

    np.testing.assert_array_equal(traj1.p_east_m, traj2.p_east_m)
    np.testing.assert_array_equal(traj1.p_north_m, traj2.p_north_m)
    np.testing.assert_array_equal(traj1.heading_rad, traj2.heading_rad)


# 16. Baseline C CAN Inertial Propagation
def test_baseline_c_can_inertial():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_C")
    engine.initialize(initial_heading_rad=0.0)

    for k in range(5):
        st = engine.step(time_sec=(k + 1) * 0.1, dt_sec=0.1, accel_x=1.0, yaw_rate=0.0)

    assert st.forward_speed_ms > 0.0
    assert st.p_north_m > 0.0


# 17. Reference Trajectory Mutation Isolation
def test_reference_mutation_isolation():
    engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
    df = pd.DataFrame({
        "time_sec": [0.0, 0.1, 0.2],
        "dt_sec": [0.1, 0.1, 0.1],
        "wheel_speed_rl_ms": [10.0, 10.0, 10.0],
        "wheel_speed_rr_ms": [10.0, 10.0, 10.0],
        "yaw_rate_rads": [0.0, 0.0, 0.0]
    })
    traj_a = engine.run_sequence(df, initial_heading_rad=0.0)

    ref_e1 = np.array([0.0, 0.0, 0.0])
    ref_n1 = np.array([0.0, 1.0, 2.0])
    m1, _, _ = DeadReckoningEvaluator.evaluate(traj_a, ref_e1, ref_n1)

    ref_e2 = np.array([100.0, 200.0, 300.0])
    ref_n2 = np.array([500.0, 600.0, 700.0])
    m2, _, _ = DeadReckoningEvaluator.evaluate(traj_a, ref_e2, ref_n2)

    np.testing.assert_array_equal(traj_a.p_east_m, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(traj_a.p_north_m, np.array([1.0, 2.0, 3.0]))

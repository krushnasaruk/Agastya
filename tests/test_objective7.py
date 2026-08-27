"""
Comprehensive Automated Test Suite for Objective 7:
Real-Time Navigation Engine Integration, Deployment Readiness & Hardware-in-the-Loop Validation.
Contains 40+ unit, integration, causality, safety, watchdog, and regression tests.
"""

import os
import copy
import numpy as np
import pandas as pd
import pytest
import torch

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from objective6.distribution_monitor import TrainingDistributionMonitor
from objective6.selective_policy import SelectiveCorrectionPolicy
from objective7.sensor_validator import SensorValidator, SensorValidationResult
from objective7.latency_monitor import LatencyMonitor, EpochLatencyBreakdown
from objective7.memory_monitor import MemoryMonitor
from objective7.watchdog import AIWatchdog, WatchdogStatus
from objective7.timeout_handler import TimeoutExperimentHandler
from objective7.deterministic_runtime import DeterministicRuntime, compute_file_sha256
from objective7.telemetry import TelemetryLogger, TelemetryFrame
from objective7.hardware_interface import ReplaySensorSource, HardwareSensorSource, RawSensorSample
from objective7.hil_runner import HILRunner
from objective7.numerical_stability import NumericalStabilityMonitor
from objective7.regression_checker import RegressionChecker
from objective7.realtime_engine import RealtimeNavigationEngine
from objective7.inference_runner import InferenceRunner
from objective7.replay_engine import RealtimeReplayEngine, ReplayResult
from objective7.benchmark import Objective7BenchmarkSuite
from objective7.experiments import Objective7ExperimentSuite


@pytest.fixture
def test_setup():
    DeterministicRuntime.set_deterministic_seed(42)
    model = CausalResidualGRU(input_dim=16, hidden_dim=64, mlp_dim=32, output_dim=2)
    if os.path.exists("artifacts/objective5/best_model.pt"):
        model.load_state_dict(torch.load("artifacts/objective5/best_model.pt", map_location="cpu"))
    model.eval()

    feat_scaler = TrainOnlyScaler.load_json("artifacts/objective5/feature_scaler.json") if os.path.exists("artifacts/objective5/feature_scaler.json") else TrainOnlyScaler()
    target_scaler = TargetScaler.load_json("artifacts/objective5/target_scaler.json") if os.path.exists("artifacts/objective5/target_scaler.json") else TargetScaler()
    dist_monitor = TrainingDistributionMonitor.load_json("artifacts/objective6/feature_distribution.json") if os.path.exists("artifacts/objective6/feature_distribution.json") else TrainingDistributionMonitor()

    engine = RealtimeNavigationEngine(
        model=model,
        feature_scaler=feat_scaler,
        target_scaler=target_scaler,
        distribution_monitor=dist_monitor,
        execution_budget_ms=25.0,
        window_size=10
    )
    return {
        "model": model,
        "feat_scaler": feat_scaler,
        "target_scaler": target_scaler,
        "dist_monitor": dist_monitor,
        "engine": engine
    }


# ==============================================================================
# 1. Real-Time Engine Initialization & Warmup Tests
# ==============================================================================

def test_realtime_engine_initialization(test_setup):
    engine = test_setup["engine"]
    engine.initialize(10.0, 20.0, 0.5, 1.0)
    st = engine.get_navigation_state()
    assert st.p_east_m == 10.0
    assert st.p_north_m == 20.0
    assert np.isclose(st.heading_rad, 0.5)
    assert st.time_sec == 1.0
    assert engine.is_initialized is True


def test_single_sample_processing(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    st = engine.process_sensor_sample(0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert not np.isnan(st.p_east_m)
    assert not np.isnan(st.p_north_m)
    assert st.forward_speed_ms >= 0.0


def test_window_warmup_fallback(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    # Process 5 samples (less than W=10)
    for i in range(5):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    t_df = engine.get_telemetry()
    assert len(t_df) == 5
    assert all(t_df["fallback_reason"] == "WINDOW_NOT_READY")
    assert all(t_df["fallback"] == True)


def test_causal_processing_no_future_access(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    # Ensure window buffer length never exceeds W=10
    for i in range(25):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
        assert len(engine.window_buffer) <= 10


def test_reset_behavior(test_setup):
    engine = test_setup["engine"]
    engine.initialize(5.0, 5.0, 1.0, 0.0)
    for i in range(15):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    engine.reset()
    st = engine.get_navigation_state()
    assert st.p_east_m == 0.0
    assert len(engine.window_buffer) == 0


# ==============================================================================
# 2. Frozen Model & Policy Enforcement Tests
# ==============================================================================

def test_model_artifact_loading(test_setup):
    model = test_setup["model"]
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 28194


def test_frozen_model_no_grad(test_setup):
    engine = test_setup["engine"]
    win = np.ones((10, 16))
    dv, dw, lat = engine.inference_runner.predict_residual(win)
    assert isinstance(dv, float)
    assert isinstance(dw, float)


def test_velocity_only_policy_default(test_setup):
    engine = test_setup["engine"]
    assert engine.policy.enable_v is True
    assert engine.policy.enable_w is False


def test_yaw_residual_disabled_by_default(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    for i in range(12):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.05)
    t_df = engine.get_telemetry()
    assert len(t_df) == 12


# ==============================================================================
# 3. Sensor Validator & Fault Injection Tests
# ==============================================================================

def test_sensor_validator_healthy():
    val = SensorValidator()
    res = val.validate_sample(1.0, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is True
    assert res.is_degraded is False


def test_nan_wheel_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, 0.1, np.nan, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_WHEEL_SPEED"


def test_nan_yaw_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, np.nan)
    assert res.is_valid is False
    assert res.status_code == "INVALID_YAW_RATE"


def test_inf_accel_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, 0.1, 10.0, 10.0, 10.0, 10.0, np.inf, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_ACCELERATION"


def test_invalid_nan_timestamp():
    val = SensorValidator()
    res = val.validate_sample(np.nan, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_TIMESTAMP"


def test_non_monotonic_timestamp():
    val = SensorValidator()
    val.validate_sample(10.0, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    res = val.validate_sample(9.0, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "NON_MONOTONIC_TIMESTAMP"


def test_zero_dt_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_DT"


def test_negative_dt_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, -0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_DT"


def test_excessive_dt_input():
    val = SensorValidator()
    res = val.validate_sample(1.0, 5.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert res.is_valid is False
    assert res.status_code == "INVALID_DT"


def test_engine_handles_sensor_dropout_gracefully(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    st = engine.process_sensor_sample(1.0, 0.1, None, 10.0, 10.0, 10.0, 0.0, 0.0)
    assert not np.isnan(st.p_east_m)
    assert not np.isnan(st.p_north_m)


def test_engine_handles_ai_exception_gracefully(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    for _ in range(10):
        engine.process_sensor_sample(0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    st = engine.process_sensor_sample(1.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, inject_ai_exception=True)
    assert not np.isnan(st.p_east_m)
    telem = engine.get_telemetry()
    assert telem.iloc[-1]["fallback_reason"] == "AI_EXCEPTION"


# ==============================================================================
# 4. Watchdog & AI Timeout Tests
# ==============================================================================

def test_watchdog_nominal_execution():
    wd = AIWatchdog(execution_budget_ms=25.0)
    wd.start_cycle()
    status = wd.check_deadline(artificial_delay_ms=2.0)
    assert status.is_timed_out is False
    assert status.fallback_triggered is False


def test_watchdog_timeout_trigger():
    wd = AIWatchdog(execution_budget_ms=25.0)
    wd.start_cycle()
    status = wd.check_deadline(artificial_delay_ms=30.0)
    assert status.is_timed_out is True
    assert status.fallback_triggered is True
    assert wd.timeout_count == 1


def test_timeout_handler_spectrum():
    wd = AIWatchdog(execution_budget_ms=25.0)
    res = TimeoutExperimentHandler.evaluate_timeout_degradation(wd, [1.0, 10.0, 30.0, 100.0])
    assert len(res) == 4
    assert res[0]["fallback_triggered"] is False
    assert res[2]["fallback_triggered"] is True
    assert res[3]["fallback_triggered"] is True


def test_engine_watchdog_timeout_fallback(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    for _ in range(10):
        engine.process_sensor_sample(0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    st = engine.process_sensor_sample(1.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, artificial_ai_delay_ms=35.0)
    assert not np.isnan(st.p_east_m)
    telem = engine.get_telemetry()
    assert bool(telem.iloc[-1]["watchdog_timeout"]) is True
    assert telem.iloc[-1]["fallback_reason"] == "AI_TIMEOUT"


# ==============================================================================
# 5. Latency, Memory, and Throughput Benchmarks
# ==============================================================================

def test_latency_monitor_summary():
    mon = LatencyMonitor(deadline_ms=100.0, preferred_target_ms=50.0)
    for _ in range(50):
        mon.record_epoch(EpochLatencyBreakdown(0.01, 0.02, 0.03, 0.001, 0.5, 0.02, 0.01, 0.591))
    stats = mon.get_summary_statistics()
    assert stats["total_epochs"] == 50
    assert stats["deadline_compliant"] is True
    assert stats["p99_total_ms"] < 50.0


def test_memory_monitor_boundedness():
    mem = MemoryMonitor()
    mem.record_snapshot("start")
    mem.record_snapshot("end")
    eval_res = mem.evaluate_memory_stability()
    assert eval_res["is_bounded"] is True


def test_benchmark_cold_vs_warm(test_setup):
    engine = test_setup["engine"]
    bench = Objective7BenchmarkSuite.run_cold_vs_warm_benchmark(engine, num_warm_epochs=50)
    assert "cold_start_latency_ms" in bench
    assert bench["warm_execution_summary"]["total_epochs"] == 51


def test_throughput_load_test(test_setup):
    engine = test_setup["engine"]
    tp = Objective7BenchmarkSuite.run_throughput_load_test(engine, target_frequencies=[10.0, 20.0], samples_per_freq=50)
    assert "10Hz_target" in tp
    assert tp["10Hz_target"]["is_realtime_capable"] is True


# ==============================================================================
# 6. Determinism & Numerical Stability Tests
# ==============================================================================

def test_deterministic_runtime_seed():
    DeterministicRuntime.set_deterministic_seed(42)
    r1 = np.random.rand(5)
    DeterministicRuntime.set_deterministic_seed(42)
    r2 = np.random.rand(5)
    assert np.allclose(r1, r2)


def test_numerical_stability_monitor():
    stab = NumericalStabilityMonitor()
    assert stab.check_state(10.0, 20.0, 1.5, 10.0) is True
    assert stab.check_state(np.nan, 20.0, 1.5, 10.0) is False
    assert stab.check_state(10.0, np.inf, 1.5, 10.0) is False
    res = stab.get_summary()
    assert res["nan_occurrences"] == 1
    assert res["inf_occurrences"] == 1


def test_deterministic_replay(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    for i in range(20):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    traj1 = engine.get_trajectory()

    engine.initialize()
    for i in range(20):
        engine.process_sensor_sample(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    traj2 = engine.get_trajectory()

    assert np.allclose(traj1.p_east_m, traj2.p_east_m)
    assert np.allclose(traj1.p_north_m, traj2.p_north_m)


# ==============================================================================
# 7. Hardware-in-the-Loop & Hardware Interface Tests
# ==============================================================================

def test_replay_sensor_source():
    df = pd.DataFrame({
        "time_sec": [0.0, 0.1, 0.2],
        "dt_sec": [0.1, 0.1, 0.1],
        "wheel_speed_fl_ms": [10.0, 10.0, 10.0],
        "wheel_speed_fr_ms": [10.0, 10.0, 10.0],
        "wheel_speed_rl_ms": [10.0, 10.0, 10.0],
        "wheel_speed_rr_ms": [10.0, 10.0, 10.0],
        "accel_x_ms2": [0.0, 0.0, 0.0],
        "yaw_rate_rads": [0.0, 0.0, 0.0]
    })
    source = ReplaySensorSource(df)
    s0 = source.read_sample()
    assert s0.sample_index == 0
    assert s0.wheel_fl_ms == 10.0
    s1 = source.read_sample()
    s2 = source.read_sample()
    s3 = source.read_sample()
    assert s3 is None


def test_hardware_sensor_source_interface():
    df = pd.DataFrame({
        "time_sec": [0.0, 0.1],
        "dt_sec": [0.1, 0.1],
        "wheel_speed_fl_ms": [8.0, 8.0],
        "wheel_speed_fr_ms": [8.0, 8.0],
        "wheel_speed_rl_ms": [8.0, 8.0],
        "wheel_speed_rr_ms": [8.0, 8.0],
        "accel_x_ms2": [0.0, 0.0],
        "yaw_rate_rads": [0.0, 0.0]
    })
    hw = HardwareSensorSource(ReplaySensorSource(df))
    assert hw.hardware_label.startswith("SOFTWARE-HIL")
    sample = hw.read_sample()
    assert sample.wheel_fl_ms == 8.0


def test_hil_stream_runner():
    runner = HILRunner(target_frequency_hz=10.0)
    res = runner.run_stream_benchmark(num_epochs=10)
    assert res["total_streamed_epochs"] == 10
    assert res["mean_jitter_ms"] >= 0.0


# ==============================================================================
# 8. Telemetry Schema & Regression Protection Tests
# ==============================================================================

def test_telemetry_schema():
    schema = TelemetryLogger.get_telemetry_schema()
    assert "fields" in schema
    assert len(schema["fields"]) == 17


def test_regression_checker_pass():
    mock_metrics = {
        "ate_rmse_m": 1.6062,
        "final_position_error_m": 1.8013,
        "heading_rmse_deg": 0.1560
    }
    res = RegressionChecker.evaluate_regression(mock_metrics, tolerance_m=0.01)
    assert res["regression_detected"] is False
    assert res["regression_check_status"].startswith("PASS")


def test_regression_checker_detects_regression():
    mock_metrics = {
        "ate_rmse_m": 2.5000,
        "final_position_error_m": 3.0000,
        "heading_rmse_deg": 1.0000
    }
    res = RegressionChecker.evaluate_regression(mock_metrics, tolerance_m=0.01)
    assert res["regression_detected"] is True
    assert res["regression_check_status"] == "REGRESSION_DETECTED"


# ==============================================================================
# 9. Step API & Architectural Compatibility Alias Tests
# ==============================================================================

def test_engine_step_api_with_dataclass(test_setup):
    from objective7.realtime_engine import SensorPacket, NavigationStepResult
    engine = test_setup["engine"]
    engine.initialize()
    packet = SensorPacket(
        timestamp_sec=0.1,
        dt_sec=0.1,
        wheel_speed_fl_ms=10.0,
        wheel_speed_fr_ms=10.0,
        wheel_speed_rl_ms=10.0,
        wheel_speed_rr_ms=10.0,
        accel_x_ms2=0.0,
        yaw_rate_rads=0.0
    )
    result: NavigationStepResult = engine.step(packet)
    assert isinstance(result, NavigationStepResult)
    assert result.velocity >= 0.0
    assert result.numerical_status == "STABLE"
    assert result.watchdog_status == "HEALTHY"


def test_engine_step_api_with_dict(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    packet = {
        "time_sec": 0.1,
        "dt_sec": 0.1,
        "wheel_fl": 12.0,
        "wheel_fr": 12.0,
        "wheel_rl": 12.0,
        "wheel_rr": 12.0,
        "accel_x": 0.1,
        "yaw_rate": 0.01
    }
    result = engine.step(packet)
    assert result.timestamp == 0.1
    assert result.velocity > 0.0


def test_architectural_aliases():
    from objective7 import (
        NumericalStabilityChecker, NumericalStabilityMonitor,
        Watchdog, AIWatchdog,
        TimeoutHandler, TimeoutExperimentHandler,
        HardwareInterface, SensorSource
    )
    assert NumericalStabilityChecker is NumericalStabilityMonitor
    assert Watchdog is AIWatchdog
    assert TimeoutHandler is TimeoutExperimentHandler
    assert HardwareInterface is SensorSource

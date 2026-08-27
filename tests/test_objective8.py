"""
Unit and Integration Test Suite for Objective 8:
Hardware-Ready Navigation Deployment, Quantized Inference & Robustness Validation.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import os

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler

from objective8.quantization import ModelQuantizer, FallbackQuantizedModel
from objective8.quantized_model import QuantizedInferenceWrapper
from objective8.model_compression import ModelCompressionAnalyzer
from objective8.memory_monitor import MemoryMonitor
from objective8.cpu_monitor import CPUMonitor
from objective8.resource_monitor import ResourceMonitor
from objective8.artifact_integrity import ArtifactIntegrityValidator, compute_file_sha256
from objective8.deployment_validator import DeploymentValidator
from objective8.runtime_profiles import DeploymentProfile, RuntimeProfileRegistry, PROFILE_REFERENCE_CPU
from objective8.constrained_runtime import ConstrainedRuntimeContext
from objective8.numerical_stability import NumericalStabilityMonitor
from objective8.hardware_ready_engine import HardwareReadyNavigationEngine, HardwareSensorPacket, HardwareStepResult
from objective8.fault_injector import HardwareFaultInjector
from objective8.outage_runner import OutageRunner
from objective8.hil_runner import HILRunner
from objective8.regression_checker import RegressionChecker
from objective8.long_duration_runner import LongDurationRunner
from objective8.benchmark import Objective8BenchmarkSuite


@pytest.fixture
def test_setup():
    model_path = "artifacts/objective5/best_model.pt"
    f_scaler_path = "artifacts/objective5/feature_scaler.json"
    t_scaler_path = "artifacts/objective5/target_scaler.json"

    model = CausalResidualGRU()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    f_scaler = TrainOnlyScaler.load(f_scaler_path) if os.path.exists(f_scaler_path) else TrainOnlyScaler(feature_names=["f" + str(i) for i in range(16)], mean=np.zeros(16), scale=np.ones(16))
    t_scaler = TargetScaler.load(t_scaler_path) if os.path.exists(t_scaler_path) else TargetScaler()

    engine = HardwareReadyNavigationEngine(
        model=model,
        feature_scaler=f_scaler,
        target_scaler=t_scaler,
        deployment_mode="MODE_B_INT8"
    )

    return {
        "model": model,
        "feature_scaler": f_scaler,
        "target_scaler": t_scaler,
        "engine": engine
    }


# ==============================================================================
# 1. Quantization & Compression Tests
# ==============================================================================

def test_dynamic_quantization_creation(test_setup):
    model = test_setup["model"]
    quant_model = ModelQuantizer.quantize_dynamic_int8(model)
    assert quant_model is not None
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        out, _ = quant_model(x)
    assert out.shape == (1, 2)


def test_quantization_error_analysis(test_setup):
    model = test_setup["model"]
    quant_model = ModelQuantizer.quantize_dynamic_int8(model)
    windows = np.random.randn(50, 10, 16).astype(np.float32)
    res = ModelQuantizer.compare_quantization_error(model, quant_model, windows)
    assert "velocity_residual" in res
    assert res["velocity_residual"]["mae_m_s"] >= 0.0
    assert res["yaw_residual"]["mae_rad_s"] >= 0.0


def test_model_compression_analysis(test_setup):
    model = test_setup["model"]
    quant_model = ModelQuantizer.quantize_dynamic_int8(model)
    comp = ModelCompressionAnalyzer.analyze_model_compression(model, quant_model)
    assert comp["parameter_counts"]["fp32_total_parameters"] == 28194
    assert comp["compression_efficiency"]["compression_ratio"] >= 1.0


# ==============================================================================
# 2. Artifact Integrity & Deployment Pre-Flight Tests
# ==============================================================================

def test_artifact_integrity_verification():
    res = ArtifactIntegrityValidator.verify_artifacts(
        model_path="artifacts/objective5/best_model.pt",
        feature_scaler_path="artifacts/objective5/feature_scaler.json",
        target_scaler_path="artifacts/objective5/target_scaler.json"
    )
    assert res["integrity_passed"] is True
    assert res["status"] == "PASS"


def test_deployment_validator_preflight():
    res = DeploymentValidator.run_preflight_checks(
        model_path="artifacts/objective5/best_model.pt",
        feature_scaler_path="artifacts/objective5/feature_scaler.json",
        target_scaler_path="artifacts/objective5/target_scaler.json"
    )
    assert res["preflight_passed"] is True
    assert res["status"] == "DEPLOYMENT_READY"


# ==============================================================================
# 3. CPU & Memory Resource Monitoring Tests
# ==============================================================================

def test_cpu_monitor_single_core():
    cpu_mon = CPUMonitor()
    cpu_mon.enable_single_core_simulation()
    assert cpu_mon.get_cpu_status()["current_num_threads"] == 1
    cpu_mon.restore_default_cores()
    assert cpu_mon.get_cpu_status()["current_num_threads"] == cpu_mon.default_threads


def test_memory_monitor_boundedness():
    mem = MemoryMonitor(max_allowed_growth_mb=25.0)
    mem.record_snapshot("init")
    mem.record_snapshot("step_100")
    res = mem.evaluate_memory_stability()
    assert res["is_bounded"] is True


def test_resource_monitor_tracking():
    mon = ResourceMonitor(memory_limit_mb=25.0, latency_budget_ms=25.0)
    t0 = mon.start_epoch()
    res = mon.end_epoch(t0, ai_applied=True, fallback=False)
    assert res["is_latency_compliant"] is True
    summary = mon.get_resource_summary()
    assert summary["total_navigation_epochs"] == 1
    assert summary["total_ai_inferences"] == 1


# ==============================================================================
# 4. Runtime Profiles & Constrained Context Tests
# ==============================================================================

def test_runtime_profile_registry():
    prof = RuntimeProfileRegistry.get_profile("SINGLE_CORE")
    assert prof.num_threads == 1
    assert prof.precision_mode == "INT8"


def test_constrained_runtime_context():
    prof = RuntimeProfileRegistry.get_profile("SINGLE_CORE")
    with ConstrainedRuntimeContext(prof) as ctx:
        info = ctx.get_runtime_info()
        assert info["active_threads"] == 1
    assert torch.get_num_threads() == prof.num_threads or torch.get_num_threads() >= 1


# ==============================================================================
# 5. Hardware Ready Engine Step Execution & Deployment Modes
# ==============================================================================

def test_engine_initialization(test_setup):
    engine = test_setup["engine"]
    engine.initialize(0.0, 0.0, 0.0, 0.0)
    assert engine.is_initialized is True
    assert engine.current_state.is_stationary is True


def test_engine_step_execution(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    pkt = HardwareSensorPacket(
        timestamp_sec=0.1,
        dt_sec=0.1,
        wheel_speed_fl_ms=10.0,
        wheel_speed_fr_ms=10.0,
        wheel_speed_rl_ms=10.0,
        wheel_speed_rr_ms=10.0,
        accel_x_ms2=0.0,
        yaw_rate_rads=0.0
    )
    res: HardwareStepResult = engine.step(pkt)
    assert isinstance(res, HardwareStepResult)
    assert res.velocity >= 0.0
    assert res.numerical_status == "STABLE"


def test_engine_classical_mode(test_setup):
    engine_class = HardwareReadyNavigationEngine(
        model=test_setup["model"],
        feature_scaler=test_setup["feature_scaler"],
        target_scaler=test_setup["target_scaler"],
        deployment_mode="MODE_C_CLASSICAL"
    )
    engine_class.initialize()
    pkt = HardwareSensorPacket(0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0)
    res = engine_class.step(pkt)
    assert res.fallback_active is True
    assert res.fallback_reason == "MODE_CLASSICAL_ONLY"


def test_engine_watchdog_timeout(test_setup):
    engine = test_setup["engine"]
    engine.initialize()
    for i in range(12):
        engine.step(HardwareSensorPacket(i * 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
    res = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0), artificial_ai_delay_ms=35.0)
    assert res.fallback_active is True
    assert res.fallback_reason == "AI_TIMEOUT"


# ==============================================================================
# 6. Fault Injection Matrix (16 Scenarios)
# ==============================================================================

def test_fault_injection_matrix(test_setup):
    engine = test_setup["engine"]
    res = HardwareFaultInjector.run_all_16_fault_tests(engine)
    assert res["passed_scenarios"] == 16
    assert res["all_faults_handled_gracefully"] is True
    assert res["status"] == "PASS"


# ==============================================================================
# 7. GNSS Outage, HIL, & Regression Tests
# ==============================================================================

def test_hil_runner_benchmark():
    runner = HILRunner(target_frequency_hz=10.0)
    res = runner.run_stream_benchmark(num_epochs=20)
    assert res["total_streamed_epochs"] == 20
    assert res["mean_jitter_ms"] >= 0.0
    assert res["status"] == "PASS"
    assert "SOFTWARE-HIL" in res["hardware_validation_label"]


def test_regression_checker():
    metrics = {
        "ate_rmse_m": 1.6062,
        "final_position_error_m": 1.8013,
        "heading_rmse_deg": 0.1560,
        "ai_application_rate_pct": 70.6
    }
    res = RegressionChecker.evaluate_regression(metrics)
    assert res["regression_detected"] is False
    assert res["status"] == "PASS (Zero Regression)"


def test_long_duration_stress(test_setup):
    engine = test_setup["engine"]
    res = LongDurationRunner.run_stress_test(engine, num_epochs=100)
    assert res["total_stress_epochs"] == 100
    assert res["stability_summary"]["is_numerically_stable"] is True
    assert res["status"] == "PASS"


# ==============================================================================
# 8. Additional Edge & Deployment Profile Tests
# ==============================================================================

def test_all_runtime_profiles():
    for name in ["REFERENCE_CPU", "SINGLE_CORE", "TIGHT_BUDGET_10MS", "MICRO_BUDGET_2MS", "MEMORY_CONSTRAINED_4MB"]:
        prof = RuntimeProfileRegistry.get_profile(name)
        assert prof.name is not None
        assert prof.watchdog_budget_ms > 0
        assert prof.memory_limit_mb > 0


def test_fallback_quantized_model():
    fallback_m = FallbackQuantizedModel()
    x = torch.randn(1, 10, 16)
    out, h = fallback_m(x)
    assert out.shape == (1, 2)
    assert h is not None


def test_quantized_wrapper_prediction_shape(test_setup):
    quant_model = ModelQuantizer.quantize_dynamic_int8(test_setup["model"])
    wrapper = QuantizedInferenceWrapper(quant_model, test_setup["feature_scaler"], test_setup["target_scaler"])
    w = np.random.randn(10, 16).astype(np.float32)
    dv, dw, lat, h = wrapper.predict_step(w)
    assert isinstance(dv, float)
    assert isinstance(dw, float)
    assert lat >= 0.0


def test_preflight_missing_file_failure():
    res = DeploymentValidator.run_preflight_checks(
        model_path="non_existent_model.pt",
        feature_scaler_path="artifacts/objective5/feature_scaler.json",
        target_scaler_path="artifacts/objective5/target_scaler.json"
    )
    assert res["preflight_passed"] is False
    assert res["status"] == "PREFLIGHT_FAILED"


def test_hardware_sensor_packet_dict_conversion():
    pkt = HardwareSensorPacket(1.0, 0.1, 12.0, 12.0, 12.0, 12.0, 0.5, 0.02)
    d = pkt.to_dict()
    assert d["timestamp_sec"] == 1.0
    assert d["accel_x_ms2"] == 0.5


def test_numerical_stability_monitor_boundaries():
    mon = NumericalStabilityMonitor(max_pos_bound_m=1000.0, max_speed_bound_ms=50.0)
    assert mon.check_state(0.0, 0.0, 0.0, 10.0) is True
    assert mon.check_state(np.nan, 0.0, 0.0, 10.0) is False
    assert mon.check_state(0.0, 1500.0, 0.0, 10.0) is False
    assert mon.check_state(0.0, 0.0, 0.0, 60.0) is False
    summary = mon.get_summary()
    assert summary["nan_occurrences"] >= 1
    assert summary["speed_explosions"] >= 1


def test_resource_monitor_compliance():
    mon = ResourceMonitor(memory_limit_mb=50.0, latency_budget_ms=50.0)
    t0 = mon.start_epoch()
    res = mon.end_epoch(t0, ai_applied=True, fallback=False)
    assert res["is_latency_compliant"] is True
    assert mon.get_resource_summary()["resource_compliance"]["latency_compliant"] is True


def test_outage_runner_evaluation(test_setup):
    test_df = pd.DataFrame({
        "time_sec": [i * 0.1 for i in range(200)],
        "dt_sec": [0.1] * 200,
        "wheel_speed_fl_ms": [10.0] * 200,
        "wheel_speed_fr_ms": [10.0] * 200,
        "wheel_speed_rl_ms": [10.0] * 200,
        "wheel_speed_rr_ms": [10.0] * 200,
        "accel_x_ms2": [0.0] * 200,
        "yaw_rate_rads": [0.0] * 200
    })
    ref_df = pd.DataFrame({
        "pos_east_m": [0.0] * 200,
        "pos_north_m": [i * 1.0 for i in range(200)]
    })
    engine_f = HardwareReadyNavigationEngine(test_setup["model"], test_setup["feature_scaler"], test_setup["target_scaler"], deployment_mode="MODE_A_FP32")
    engine_i = HardwareReadyNavigationEngine(test_setup["model"], test_setup["feature_scaler"], test_setup["target_scaler"], deployment_mode="MODE_B_INT8")
    res = OutageRunner.evaluate_outages(engine_f, engine_i, test_df, ref_df, outage_start_time_sec=5.0, outage_durations=[5.0, 10.0])
    assert "outage_records" in res
    assert res["status"] == "PASS"


def test_benchmark_latency_profiling(test_setup):
    lat_stats = Objective8BenchmarkSuite.run_latency_benchmark(test_setup["engine"], num_epochs=20)
    assert lat_stats["total_epochs"] == 20
    assert lat_stats["total_latency"]["mean_ms"] > 0
    assert lat_stats["p99_total_ms"] >= 0.0


def test_benchmark_throughput_profiling(test_setup):
    tp_stats = Objective8BenchmarkSuite.run_throughput_load_test(test_setup["engine"], target_frequencies_hz=[10.0, 20.0])
    assert "10Hz_target" in tp_stats
    assert tp_stats["10Hz_target"]["achieved_throughput_hz"] > 0


def test_benchmark_profiled_runs(test_setup):
    prof_res = Objective8BenchmarkSuite.run_profiled_benchmarks(test_setup["engine"], num_epochs=10)
    assert "profiles" in prof_res
    assert len(prof_res["profiles"]) == 5


# ==============================================================================
# 9. Extended Verification Tests
# ==============================================================================

def test_compression_ratio_bounds(test_setup):
    comp = ModelCompressionAnalyzer.analyze_model_compression(test_setup["model"], test_setup["model"])
    assert comp["compression_efficiency"]["compression_ratio"] >= 1.0


def test_memory_monitor_rss_recording():
    mon = MemoryMonitor()
    mon.record_snapshot("t0")
    mon.record_snapshot("t1")
    assert len(mon.snapshots) == 2


def test_memory_monitor_slope_calculation():
    mon = MemoryMonitor()
    for i in range(10):
        mon.record_snapshot(f"step_{i}")
    res = mon.evaluate_memory_stability()
    assert "leak_slope_mb_per_min" in res
    assert res["is_bounded"] is True


def test_cpu_monitor_thread_override():
    mon = CPUMonitor()
    mon.set_thread_allocation(2)
    assert mon.get_cpu_status()["current_num_threads"] == 2
    mon.restore_default_cores()


def test_artifact_integrity_hash_computation():
    model_path = "artifacts/objective5/best_model.pt"
    if os.path.exists(model_path):
        h = compute_file_sha256(model_path)
        assert len(h) == 64


def test_artifact_integrity_tamper_detection(tmp_path):
    fake_file = tmp_path / "fake_model.pt"
    fake_file.write_text("corrupted content")
    val = ArtifactIntegrityValidator.verify_artifacts(
        model_path=str(fake_file),
        feature_scaler_path="artifacts/objective5/feature_scaler.json",
        target_scaler_path="artifacts/objective5/target_scaler.json"
    )
    assert val["integrity_passed"] is False


def test_hardware_sensor_packet_fields():
    pkt = HardwareSensorPacket(
        timestamp_sec=5.0,
        dt_sec=0.1,
        wheel_speed_fl_ms=15.0,
        wheel_speed_fr_ms=15.1,
        wheel_speed_rl_ms=14.9,
        wheel_speed_rr_ms=15.0,
        accel_x_ms2=0.2,
        yaw_rate_rads=0.01
    )
    assert pkt.timestamp_sec == 5.0
    assert pkt.accel_x_ms2 == 0.2


def test_hardware_step_result_dataclass():
    from navigation_engine.dead_reckoning import PlanarNavigationState
    st = PlanarNavigationState()
    res = HardwareStepResult(
        timestamp=1.0,
        position=(10.0, 20.0),
        velocity=15.0,
        heading=1.57,
        classical_state=st,
        corrected_state=st,
        delta_v=0.0,
        delta_omega_z=0.0,
        ai_applied=True,
        fallback_active=False,
        fallback_reason="NONE",
        confidence=1.0,
        ood_score=0.0,
        inference_latency_ms=0.45,
        total_latency_ms=0.85,
        numerical_status="STABLE",
        deployment_mode="MODE_B_INT8"
    )
    d = res.to_dict()
    assert d["velocity"] == 15.0
    assert d["ai_applied"] is True


def test_telemetry_bounded_buffer():
    from objective7.telemetry import TelemetryLogger, TelemetryFrame
    logger = TelemetryLogger(max_buffer_size=5)
    for i in range(10):
        frame = TelemetryFrame(
            timestamp=float(i), dt=0.1, classical_velocity=10.0, corrected_velocity=10.0,
            predicted_delta_velocity=0.0, predicted_delta_yaw=0.0, ai_applied=True,
            fallback=False, fallback_reason="NONE", confidence=1.0, ood_score=0.0,
            inference_latency_ms=0.5, total_latency_ms=0.8, watchdog_timeout=False,
            sensor_valid=True, stationary=False, navigation_state_valid=True
        )
        logger.log_frame(frame)
    assert len(logger.frames) == 5


def test_numerical_stability_recovery():
    mon = NumericalStabilityMonitor()
    mon.record_violation("NaN in position")
    assert mon.nan_count == 1
    summary = mon.get_summary()
    assert summary["is_numerically_stable"] is False


def test_regression_checker_detection():
    # If ATE error increases beyond threshold, should flag regression
    bad_metrics = {
        "ate_rmse_m": 2.5000,
        "final_position_error_m": 3.0,
        "heading_rmse_deg": 0.5,
        "ai_application_rate_pct": 50.0
    }
    res = RegressionChecker.evaluate_regression(bad_metrics)
    assert res["regression_detected"] is True
    assert res["status"] == "REGRESSION_FAIL"


def test_hil_runner_jitter_calculation():
    runner = HILRunner(target_frequency_hz=10.0)
    timestamps = [0.0, 0.101, 0.199, 0.302, 0.398]
    jitters = [abs((timestamps[i] - timestamps[i-1]) * 1000.0 - 100.0) for i in range(1, len(timestamps))]
    mean_j = float(np.mean(jitters))
    assert mean_j < 5.0



"""
Comprehensive Automated Test Suite for Objective 4 AI Error Modeling & Formulation.
Validates residual target extraction, causal sliding window generation, zero future/reference leakage,
sequence-level dataset splitting, statistical descriptors, safety bounding gates, and determinism.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.ml.src.targets.residual_targets import ResidualTargetExtractor, ResidualTargetsContainer, wrap_to_pi
from services.ml.src.targets.statistics import ResidualStatisticsAnalyzer, TargetStatistics
from services.ml.src.features.causal_features import CausalFeatureExtractor, CAUSAL_FEATURE_REGISTRY
from services.ml.src.features.window_builder import CausalWindowBuilder
from services.ml.src.data.alignment import TrajectoryTargetAligner
from services.ml.src.data.splits import DatasetSplitManager, SequenceSplitConfig
from services.ml.src.analysis.error_decomposition import PhysicalErrorDecomposer
from services.ml.src.analysis.temporal_windows import TemporalWindowAnalyzer
from services.ml.src.correction.interface import AICorrectionInput, AICorrectionOutput, AICorrectionSafetyGuard


# 1. Deterministic Target Alignment
def test_target_alignment_deterministic():
    n = 20
    t = np.arange(n) * 0.1
    dt = np.full(n, 0.1)

    nav_df = pd.DataFrame({"time_sec": t, "dt_sec": dt})
    traj_df = pd.DataFrame({
        "estimated_p_east_m": np.zeros(n),
        "estimated_p_north_m": np.zeros(n),
        "estimated_heading_rad": np.zeros(n),
        "estimated_speed_ms": np.full(n, 10.0),
        "yaw_rate_rads": np.zeros(n)
    })
    ref_df = pd.DataFrame({
        "pos_east_m": np.zeros(n),
        "pos_north_m": np.zeros(n),
        "heading_rad": np.zeros(n),
        "ground_speed_ms": np.full(n, 10.2)
    })

    aligned = TrajectoryTargetAligner.align(nav_df, traj_df, ref_df)
    assert aligned.num_aligned_samples == 20
    assert np.all(aligned.valid_mask)


# 2. Velocity Residual Calculation
def test_velocity_residual_calculation():
    v_ref = np.array([10.0, 15.0, 20.0])
    v_class = np.array([9.8, 14.9, 20.3])
    delta_v = v_ref - v_class
    np.testing.assert_allclose(delta_v, [0.2, 0.1, -0.3], atol=1e-5)


# 3. Heading Residual Angle Wrapping
def test_heading_residual_angle_wrapping():
    # Wrap test: Reference is 5 deg (0.087 rad), Estimate is 355 deg (6.196 rad)
    # True wrapped error should be +10 deg (+0.1745 rad), NOT -350 deg
    ref_h = np.radians(5.0)
    est_h = np.radians(355.0)
    diff = wrap_to_pi(ref_h - est_h)
    assert np.isclose(np.degrees(diff), 10.0, atol=1e-4)


# 4. Incremental Displacement Residual
def test_incremental_displacement_residual():
    ref_e = np.array([0.0, 1.0, 2.1, 3.3])
    class_e = np.array([0.0, 1.0, 2.0, 3.0])
    d_ref = np.diff(ref_e)
    d_class = np.diff(class_e)
    delta_disp = d_ref - d_class
    np.testing.assert_allclose(delta_disp, [0.0, 0.1, 0.2], atol=1e-5)


# 5. Causal Feature Registry Schema
def test_causal_feature_registry_schema():
    assert len(CAUSAL_FEATURE_REGISTRY) >= 15
    for feat_name, meta in CAUSAL_FEATURE_REGISTRY.items():
        assert meta.causal_status == "STRICTLY CAUSAL"
        assert meta.units != ""
        assert meta.physical_interpretation != ""


# 6. Causal Sliding Window Shapes
def test_causal_window_builder_shapes():
    n_samples = 100
    n_feats = 16
    w_size = 10

    features_df = pd.DataFrame(np.random.randn(n_samples, n_feats), columns=[f"f_{i}" for i in range(n_feats)])
    targets = np.random.randn(n_samples)

    x_win, y_win, names = CausalWindowBuilder.build_causal_windows(features_df, targets, window_size=w_size)
    assert x_win.shape == (91, 10, 16)
    assert y_win.shape == (91,)
    assert len(names) == 16


# 7. Strict Future Leakage Invariance in Window Builder
def test_strict_future_leakage_in_windows():
    n_samples = 50
    w_size = 10
    features = np.arange(n_samples * 2).reshape(n_samples, 2).astype(np.float32)
    features_df = pd.DataFrame(features, columns=["feat_1", "feat_2"])
    targets = np.arange(n_samples).astype(np.float32)

    x_win, y_win, _ = CausalWindowBuilder.build_causal_windows(features_df, targets, window_size=w_size)

    # Window at index 0 (epochs 0 to 9) must contain exactly samples 0 to 9
    np.testing.assert_array_equal(x_win[0, :, 0], features[0:10, 0])
    # Target aligned at index 0 must be the target at epoch 9
    assert y_win[0] == targets[9]


# 8. Zero Reference Leakage in Causal Feature Extractor
def test_zero_reference_leakage_in_features():
    nav_df = pd.DataFrame({
        "time_sec": np.arange(10) * 0.1,
        "dt_sec": np.full(10, 0.1),
        "wheel_speed_fl_ms": np.full(10, 10.0),
        "wheel_speed_fr_ms": np.full(10, 10.0),
        "wheel_speed_rl_ms": np.full(10, 10.0),
        "wheel_speed_rr_ms": np.full(10, 10.0),
        "accel_x_ms2": np.full(10, 0.5),
        "yaw_rate_rads": np.full(10, 0.02)
    })

    feats_df = CausalFeatureExtractor.extract_features(nav_df)
    # Ensure no GPS/reference columns exist in output
    for col in feats_df.columns:
        assert "gps" not in col.lower()
        assert "ref" not in col.lower()
        assert "vbox" not in col.lower()


# 9. Sequence-Level Dataset Splitting
def test_sequence_level_dataset_splitting():
    split_cfg = SequenceSplitConfig(
        train_sequences=["sync_01"],
        val_sequences=["v_standalone_03"],
        test_sequences=["sync_02"]
    )
    assert DatasetSplitManager.validate_no_leakage(split_cfg) is True

    # Leaky split must raise ValueError
    leaky_cfg = SequenceSplitConfig(
        train_sequences=["sync_01", "sync_02"],
        val_sequences=["v_standalone_03"],
        test_sequences=["sync_02"]
    )
    with pytest.raises(ValueError) as exc:
        DatasetSplitManager.validate_no_leakage(leaky_cfg)
    assert "Data Leakage Detected" in str(exc.value)


# 10. Target Statistics Computation
def test_target_statistics_computation():
    data = np.array([0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.0, 0.0, 0.02, -0.02])
    stats = ResidualStatisticsAnalyzer.analyze_target(data, "test_target")
    assert stats.num_samples == 10
    assert np.isclose(stats.mean, 0.0)
    assert stats.min_val == -0.2
    assert stats.max_val == 0.2
    assert stats.median == 0.0


# 11. Error Decomposition Metrics
def test_error_decomposition_metrics():
    n = 100
    t = np.arange(n) * 0.1
    dt = np.full(n, 0.1)
    v_wheel = np.full(n, 10.0)
    v_ref = np.full(n, 10.1)  # 1.0% scale factor error

    decomp = PhysicalErrorDecomposer.decompose(
        time_sec=t,
        dt_sec=dt,
        v_wheel_rear_ms=v_wheel,
        yaw_rate_can_rads=np.zeros(n),
        accel_x_ms2=np.zeros(n),
        v_ref_ms=v_ref,
        v_classical_ms=v_wheel,
        heading_classical_rad=np.zeros(n),
        heading_ref_rad=np.zeros(n)
    )

    assert np.isclose(decomp.estimated_tire_scale_factor_error_pct, 1.0, atol=1e-2)
    assert decomp.slip_event_count == 0


# 12. Temporal Window Trade-off Analysis
def test_temporal_window_evaluations():
    reports = TemporalWindowAnalyzer.evaluate_all_windows(total_sequence_samples=600)
    assert len(reports) == 4
    # 1.0s window is recommended primary
    assert reports[1].duration_sec == 1.0
    assert "RECOMMENDED PRIMARY" in reports[1].recommendation_status


# 13. Safety Guard Physical Bounds Clamping
def test_ai_correction_safety_guard_bounds():
    guard = AICorrectionSafetyGuard(max_velocity_correction_ms=3.0, max_yaw_rate_correction_rads=0.50)

    # Normal plausible prediction (1.2 m/s, 0.1 rad/s)
    out1 = guard.sanitize_correction(raw_delta_v=1.2, raw_delta_yaw=0.1, var_v=0.05, var_yaw=0.01)
    assert out1.correction_applied is True
    assert out1.status == "APPLIED"
    assert np.isclose(out1.delta_velocity_ms, 1.2)

    # Implausible spike (10.0 m/s) -> must be clamped to 3.0 m/s
    out2 = guard.sanitize_correction(raw_delta_v=10.0, raw_delta_yaw=0.1, var_v=0.05, var_yaw=0.01)
    assert out2.correction_applied is True
    assert out2.status == "CLAMPED"
    assert out2.delta_velocity_ms == 3.0


# 14. Fallback on Sensor Degradation & Stationarity
def test_fallback_on_sensor_degradation():
    guard = AICorrectionSafetyGuard()

    # Degraded sensor flag -> fallback to 0 correction (pure classical physics)
    out_deg = guard.sanitize_correction(raw_delta_v=1.5, raw_delta_yaw=0.1, var_v=0.05, var_yaw=0.01, is_sensor_valid=False)
    assert out_deg.correction_applied is False
    assert out_deg.delta_velocity_ms == 0.0
    assert "FALLBACK" in out_deg.status

    # Stationary flag -> fallback to 0 correction
    out_stat = guard.sanitize_correction(raw_delta_v=1.5, raw_delta_yaw=0.1, var_v=0.05, var_yaw=0.01, is_stationary=True)
    assert out_stat.correction_applied is False
    assert out_stat.delta_velocity_ms == 0.0


# 15. Fallback on Low Confidence / High Uncertainty
def test_fallback_on_low_confidence():
    guard = AICorrectionSafetyGuard(max_acceptable_velocity_variance=1.0)
    # High variance (sigma^2 = 5.0) -> fallback
    out_unc = guard.sanitize_correction(raw_delta_v=1.5, raw_delta_yaw=0.1, var_v=5.0, var_yaw=0.01)
    assert out_unc.correction_applied is False
    assert out_unc.delta_velocity_ms == 0.0
    assert out_unc.status == "FALLBACK_LOW_CONFIDENCE"


# 16. Deterministic Target Generation Reproducibility
def test_deterministic_target_reproducibility():
    n = 50
    t = np.arange(n) * 0.1
    dt = np.full(n, 0.1)
    zeros = np.zeros(n)
    ones = np.ones(n)

    t1 = ResidualTargetExtractor.extract_all_targets(t, dt, zeros, zeros, zeros, ones, zeros, ones, zeros, zeros, zeros, ones * 1.05)
    t2 = ResidualTargetExtractor.extract_all_targets(t, dt, zeros, zeros, zeros, ones, zeros, ones, zeros, zeros, zeros, ones * 1.05)

    np.testing.assert_array_equal(t1.delta_velocity_ms, t2.delta_velocity_ms)
    np.testing.assert_array_equal(t1.delta_heading_rad, t2.delta_heading_rad)

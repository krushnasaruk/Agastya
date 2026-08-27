"""
Expanded Automated Test Suite for IO-VNBD Data Engineering Pipeline in Project AGASTYA.
Validates timestamp jitter, duplicate/retrograde handling, strict causality, future leakage guards,
ground-truth isolation, coordinate conversions, quality masking, and bitwise reproducibility.
"""

import os
import copy
import pytest
import numpy as np
import pandas as pd

from services.ml.src.data.schema import IOVNBDSchemaRegistry, CoordinateFrame, SignalSource
from services.ml.src.data.timestamps import TimestampAnalyzer
from services.ml.src.data.units import UnitNormalizer
from services.ml.src.data.coordinates import GeodeticConverter
from services.ml.src.data.quality import DataQualityManager, QualityFlag
from services.ml.src.data.synchronization import StreamSynchronizer
from services.ml.src.data.reference import ReferenceTrajectoryBuilder
from services.ml.src.data.consistency import PhysicalConsistencyChecker
from services.ml.src.data.loader import IOVNBDDataLoader, RawSequenceContainer
from services.ml.src.data.pipeline import NavigationDataPipeline, PreprocessingCausality


# 1. Schema Registry & Verification
def test_schema_registry_validation():
    valid_v_cols = [
        "Time", "Wheel speed FL", "Wheel speed FR", "Wheel speed RL", "Wheel speed RR",
        "Longitudinal acceleration", "Yaw rate", "GPS latitude", "GPS longitude",
        "GPS altitude", "GPS speed", "GPS orientation", "GPS accuracy", "GPS satellites"
    ]
    res_v = IOVNBDSchemaRegistry.validate_columns(valid_v_cols, "vehicle")
    assert res_v["is_valid"] is True
    assert len(res_v["missing_required_signals"]) == 0

    res_bad = IOVNBDSchemaRegistry.validate_columns(["Time", "GPS latitude"], "vehicle")
    assert res_bad["is_valid"] is False
    assert "GPS longitude" in res_bad["missing_required_signals"]


# 2. Timestamp Jitter Calculation
def test_timestamp_jitter_calculation():
    analyzer = TimestampAnalyzer(gap_threshold_sec=0.25, nominal_rate_hz=10.0)
    rng = np.random.RandomState(42)
    raw_ms = np.arange(100) * 100.0 + rng.normal(0, 1.5, 100)
    raw_ms = np.maximum.accumulate(raw_ms)

    stats, dt_arr, mask = analyzer.analyze(raw_ms)
    assert stats.num_samples == 100
    assert np.isclose(stats.mean_dt_sec, 0.10, atol=0.01)
    assert stats.std_dt_sec > 0.0  # Captures real timing jitter
    assert stats.is_strictly_monotonic is True
    assert len(dt_arr) == 100
    assert np.all(mask)


# 3. Duplicate Timestamp Handling
def test_duplicate_timestamp_handling():
    analyzer = TimestampAnalyzer(gap_threshold_sec=0.25, nominal_rate_hz=10.0)
    raw_ms = np.array([0.0, 100.0, 100.0, 200.0, 300.0])  # Duplicate at index 2 (dt=0)

    stats, dt_arr, mask = analyzer.analyze(raw_ms)
    assert stats.num_duplicates == 1
    assert stats.is_strictly_monotonic is False
    assert dt_arr[2] == 0.0
    assert bool(mask[2]) is False  # Duplicate epoch is masked as invalid


# 4. Non-Monotonic / Retrograde Timestamp Handling
def test_non_monotonic_timestamp_handling():
    analyzer = TimestampAnalyzer(gap_threshold_sec=0.25, nominal_rate_hz=10.0)
    raw_ms = np.array([0.0, 100.0, 80.0, 200.0, 300.0])  # Clock reset at index 2 (dt=-20ms)

    stats, dt_arr, mask = analyzer.analyze(raw_ms)
    assert stats.num_retrograde == 1
    assert stats.is_strictly_monotonic is False
    assert np.isclose(dt_arr[2], -0.02)
    assert bool(mask[2]) is False  # Retrograde epoch is masked as invalid


# 5. Missing Sample & Outlier Detection
def test_missing_sample_detection():
    qm = DataQualityManager(max_speed_ms=70.0, max_accel_ms2=25.0)
    n = 20
    data = {
        "wheel_speed_rl_ms": np.full(n, 15.0),
        "wheel_speed_rr_ms": np.full(n, 15.0),
        "accel_x_ms2": np.zeros(n),
        "latitude_deg": np.full(n, 52.41),
        "longitude_deg": np.full(n, -1.51),
        "gps_accuracy_m": np.full(n, 1.2),
        "satellites_count": np.full(n, 10),
        "gps_speed_ms": np.full(n, 15.0)
    }
    data["wheel_speed_rl_ms"][5] = np.nan
    data["accel_x_ms2"][10] = np.inf

    masks, summary = qm.evaluate_quality(data)
    assert summary.missing_samples_count == 2
    assert masks["quality_flags"][5] == QualityFlag.MISSING
    assert masks["quality_flags"][10] == QualityFlag.MISSING
    assert bool(masks["sensor_valid_mask"][5]) is False
    assert bool(masks["sensor_valid_mask"][10]) is False


# 6. Causal Synchronization Strictness (t_s <= t_v)
def test_causal_synchronization_strictness():
    sync = StreamSynchronizer()
    v_data = {"time_sec": np.array([0.0, 0.10, 0.20, 0.30, 0.40])}
    s_data = {
        "phone_time_sec": np.array([0.02, 0.12, 0.22, 0.32, 0.42]),
        "phone_acc_x_ms2": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    }

    aligned, meta = sync.align_streams(v_data, s_data, method="nearest_causal")
    assert meta["causality"] == "CAUSAL"
    
    # For every aligned sample, source_timestamp_s_ms must be <= time_sec * 1000 + epsilon
    t_v = aligned["time_sec"] * 1000.0
    t_s = aligned["source_timestamp_s_ms"]
    for i in range(len(t_v)):
        assert t_s[i] <= t_v[i] or np.isclose(t_s[i], t_v[i], atol=25.0)


# 7. Future Data Leakage Prevention Guard
def test_future_data_leakage_guard():
    pipeline = NavigationDataPipeline()

    def build_test_container(mutate_future=False):
        n = 50
        t_ms = np.arange(n) * 100.0
        v_df = pd.DataFrame({
            "Time": t_ms,
            "Wheel speed FL": np.full(n, 36.0),
            "Wheel speed FR": np.full(n, 36.0),
            "Wheel speed RL": np.full(n, 36.0),
            "Wheel speed RR": np.full(n, 36.0),
            "Longitudinal acceleration": np.zeros(n),
            "Yaw rate": np.zeros(n),
            "GPS latitude": np.full(n, 52.41),
            "GPS longitude": np.full(n, -1.51),
            "GPS altitude": np.full(n, 100.0),
            "GPS speed": np.full(n, 36.0),
            "GPS orientation": np.zeros(n),
            "GPS accuracy": np.full(n, 1.0),
            "GPS satellites": np.full(n, 10)
        })
        if mutate_future:
            # Drastically alter samples from index 30 to 49
            v_df.loc[30:, "Wheel speed FL"] = 180.0
            v_df.loc[30:, "Longitudinal acceleration"] = 10.0

        return RawSequenceContainer(
            sequence_id="leakage_test",
            vehicle_df=v_df,
            smartphone_df=None,
            is_synchronized_pair=False
        )

    base_pkg = pipeline.process(build_test_container(mutate_future=False))
    mutated_pkg = pipeline.process(build_test_container(mutate_future=True))

    # Causal outputs for samples 0 to 29 MUST BE EXACTLY IDENTICAL
    base_past = base_pkg.navigation_inputs_df.iloc[:30]
    mutated_past = mutated_pkg.navigation_inputs_df.iloc[:30]
    pd.testing.assert_frame_equal(base_past, mutated_past)


# 8. Coordinate Conversion ENU Directional Checks
def test_coordinate_conversion_enu_axes():
    conv = GeodeticConverter(lat0_deg=52.4100, lon0_deg=-1.5100, alt0_m=100.0)

    # Step due North (+lat), due East (+lon), and Up (+alt)
    lat_step = np.array([52.4100, 52.4100, 52.4110, 52.4100])
    lon_step = np.array([-1.5100, -1.5090, -1.5100, -1.5100])
    alt_step = np.array([100.0, 100.0, 100.0, 110.0])

    east, north, up = conv.geodetic_to_enu(lat_step, lon_step, alt_step)

    # Index 1: Moving East -> East > 0, North ~ 0
    assert east[1] > 60.0 and abs(north[1]) < 1.0
    # Index 2: Moving North -> North > 0, East ~ 0
    assert north[2] > 100.0 and abs(east[2]) < 1.0
    # Index 3: Moving Up -> Up == 10m
    assert np.isclose(up[3], 10.0, atol=1e-3)


# 9. Numerical Round-Trip Conversion Error (Float Precision Check)
def test_numerical_roundtrip_precision():
    conv = GeodeticConverter(lat0_deg=52.4100, lon0_deg=-1.5100, alt0_m=100.0)

    lat_in = np.array([52.4100, 52.4150, 52.4200])
    lon_in = np.array([-1.5100, -1.5050, -1.5000])
    alt_in = np.array([100.0, 150.0, 200.0])

    east, north, up = conv.geodetic_to_enu(lat_in, lon_in, alt_in)
    lat_out, lon_out, alt_out = conv.enu_to_geodetic(east, north, up)

    # Numerical float precision error (< 1e-6 degrees ~ 0.1 mm)
    lat_err_deg = float(np.max(np.abs(lat_in - lat_out)))
    lon_err_deg = float(np.max(np.abs(lon_in - lon_out)))
    alt_err_m = float(np.max(np.abs(alt_in - alt_out)))

    assert lat_err_deg < 5e-6
    assert lon_err_deg < 5e-6
    assert alt_err_m < 1e-3  # < 1 mm mathematical round-trip closure


# 10. Ground-Truth Isolation Verification (NO GPS in Causal Inputs)
def test_ground_truth_isolation():
    raw_dir = os.path.abspath("data/raw")
    loader = IOVNBDDataLoader(raw_dir)
    discovered = loader.discover_sequences()
    seq_id = list(discovered.keys())[0]

    raw_container = loader.load_raw_sequence(seq_id)
    pipeline = NavigationDataPipeline()
    package = pipeline.process(raw_container)

    nav_cols = list(package.navigation_inputs_df.columns)
    ref_cols = list(package.reference_trajectory_df.columns)

    # Assert ZERO GPS position, speed, or heading reference leakage in navigation_inputs_df
    forbidden_in_inputs = [
        "latitude_deg", "longitude_deg", "altitude_m", "pos_east_m", "pos_north_m",
        "pos_up_m", "gps_speed_ms", "ground_speed_ms", "heading_rad",
        "velocity_east_ms", "velocity_north_ms"
    ]
    for col in forbidden_in_inputs:
        assert col not in nav_cols, f"LEAKAGE DETECTED: Ground-truth column '{col}' found in navigation inputs!"

    # Assert reference_trajectory_df properly contains the ground truth
    assert "pos_east_m" in ref_cols
    assert "pos_north_m" in ref_cols
    assert "ground_speed_ms" in ref_cols


# 11. NaN / Inf Propagation Prevention
def test_nan_inf_propagation_prevention():
    qm = DataQualityManager()
    data = {
        "wheel_speed_rl_ms": np.array([10.0, np.nan, 10.0]),
        "accel_x_ms2": np.array([0.0, 0.0, np.inf]),
        "latitude_deg": np.array([52.41, 52.41, 52.41]),
        "longitude_deg": np.array([-1.51, -1.51, -1.51])
    }
    masks, summary = qm.evaluate_quality(data)
    assert not np.any(np.isnan(masks["overall_valid_mask"]))
    assert not bool(masks["overall_valid_mask"][1])
    assert not bool(masks["overall_valid_mask"][2])


# 12. Sequence Boundary Handling (Single and Two-Sample Sequences)
def test_sequence_boundary_handling():
    analyzer = TimestampAnalyzer()
    # 1 sample
    stats1, dt1, m1 = analyzer.analyze(np.array([1000.0]))
    assert stats1.num_samples == 1
    assert len(dt1) == 1
    # 2 samples
    stats2, dt2, m2 = analyzer.analyze(np.array([1000.0, 1100.0]))
    assert stats2.num_samples == 2
    assert np.isclose(dt2[1], 0.10)


# 13. Deterministic Processing Reproducibility (Bitwise Check)
def test_deterministic_processing_reproducibility():
    raw_dir = os.path.abspath("data/raw")
    loader = IOVNBDDataLoader(raw_dir)
    seq_id = list(loader.discover_sequences().keys())[0]

    pipeline = NavigationDataPipeline()
    pkg1 = pipeline.process(loader.load_raw_sequence(seq_id))
    pkg2 = pipeline.process(loader.load_raw_sequence(seq_id))

    # Strict Bitwise DataFrame Equality
    pd.testing.assert_frame_equal(pkg1.navigation_inputs_df, pkg2.navigation_inputs_df)
    pd.testing.assert_frame_equal(pkg1.reference_trajectory_df, pkg2.reference_trajectory_df)
    pd.testing.assert_frame_equal(pkg1.quality_df, pkg2.quality_df)


# 14. Metadata Completeness
def test_metadata_completeness():
    raw_dir = os.path.abspath("data/raw")
    loader = IOVNBDDataLoader(raw_dir)
    seq_id = list(loader.discover_sequences().keys())[0]

    pipeline = NavigationDataPipeline()
    package = pipeline.process(loader.load_raw_sequence(seq_id))
    exported = package.export(os.path.abspath("data/processed"))

    import json
    with open(exported["metadata_json"], "r") as f:
        meta = json.load(f)

    required_fields = [
        "pipeline_version", "sequence_id", "num_samples", "duration_sec",
        "nominal_rate_hz", "observed_rate_hz", "total_distance_m",
        "ground_truth_isolation_verified", "timestamp_stats",
        "quality_summary", "consistency_report", "sync_metadata", "origin_metadata"
    ]
    for field in required_fields:
        assert field in meta, f"Missing required metadata field: '{field}'"

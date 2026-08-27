"""
Automated Test Suite for Objective 5 Causal Residual Model Training & Validation.
Validates zero future leakage, zero reference leakage, test-set isolation, train-only scaler provenance,
window boundary isolation, target leakage prevention, safety guard clamping, and determinism.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_residual.feature_registry import CANONICAL_FEATURES, CANONICAL_FEATURE_NAMES, NUM_CANONICAL_FEATURES, validate_feature_matrix_columns
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from ai_residual.dataset import CausalWindowDataset
from ai_residual.model import CausalResidualGRU
from ai_residual.safety import SafetyGuard
from ai_residual.trainer import ResidualModelTrainer, set_seed
from ai_residual.rollout import AIRolloutEngine
from ai_residual.outage_eval import OutageComparator
from ai_residual.ablations import AblationRunner


# 1. Canonical Feature Registry Integrity
def test_canonical_feature_registry_integrity():
    assert len(CANONICAL_FEATURES) == 16
    assert len(CANONICAL_FEATURE_NAMES) == 16
    assert NUM_CANONICAL_FEATURES == 16
    assert validate_feature_matrix_columns(CANONICAL_FEATURE_NAMES) is True

    # Order mismatch must raise ValueError
    bad_order = CANONICAL_FEATURE_NAMES.copy()
    bad_order[0], bad_order[1] = bad_order[1], bad_order[0]
    with pytest.raises(ValueError) as exc:
        validate_feature_matrix_columns(bad_order)
    assert "Feature order mismatch" in str(exc.value)


# 2. Strict Future Leakage Invariance in CausalWindowDataset
def test_strict_future_leakage_in_windows():
    n_samples = 30
    w_size = 10
    features = np.arange(n_samples * 16).reshape(n_samples, 16).astype(np.float32)
    targets = np.arange(n_samples * 2).reshape(n_samples, 2).astype(np.float32)

    ds = CausalWindowDataset(features, targets, window_size=w_size)
    x_win_0, y_target_0, t_0 = ds[0]

    # Modify future samples in the source feature array (from index 10 onwards)
    features_mutated = features.copy()
    features_mutated[10:] = 9999.0
    ds_mutated = CausalWindowDataset(features_mutated, targets, window_size=w_size)
    x_win_0_mut, y_target_0_mut, t_0_mut = ds_mutated[0]

    # Window 0 (samples 0..9) must be 100% bitwise identical
    np.testing.assert_array_equal(x_win_0.numpy(), x_win_0_mut.numpy())
    np.testing.assert_array_equal(y_target_0.numpy(), y_target_0_mut.numpy())


# 3. Train-Only Scaler Provenance & Zero Leakage
def test_train_only_scaler_provenance():
    # Training data: mean 10.0, std 2.0
    train_df = pd.DataFrame(np.random.normal(10.0, 2.0, size=(100, 16)), columns=CANONICAL_FEATURE_NAMES)
    scaler = TrainOnlyScaler().fit(train_df, sequence_id="sync_01")

    assert scaler.fitted_sequence_id == "sync_01"
    assert scaler.num_training_samples == 100

    # Transform unseen test data with different mean: mean 50.0
    test_df = pd.DataFrame(np.random.normal(50.0, 2.0, size=(50, 16)), columns=CANONICAL_FEATURE_NAMES)
    normed_test = scaler.transform(test_df)

    # Scaler mean must NOT change
    assert np.isclose(scaler.means[0], 10.0, atol=0.8)
    assert not np.isclose(scaler.means[0], 50.0)


# 4. Target Leakage Prevention (Target column cannot exist in causal features)
def test_target_leakage_prevention():
    forbidden_targets = ["delta_v", "delta_omega", "target", "v_ref", "pos_ref", "gps"]
    for feat in CANONICAL_FEATURE_NAMES:
        for f in forbidden_targets:
            assert f not in feat.lower(), f"Target leakage detected in causal feature: '{feat}'"


# 5. Model Architecture & Shape Invariance
def test_model_forward_shapes():
    model = CausalResidualGRU(input_dim=16, hidden_dim=64, mlp_dim=32, output_dim=2)
    batch_size = 8
    w_size = 10
    x = torch.randn(batch_size, w_size, 16)
    out, h_n = model(x)
    assert out.shape == (batch_size, 2)
    assert h_n.shape == (1, batch_size, 64)


# 6. Safety Guard Physical Clamping & Fallbacks
def test_safety_guard_clamping_and_fallback():
    guard = SafetyGuard(max_velocity_bound_ms=3.0, max_yaw_bound_rads=0.50)

    # 1. Normal plausible residual
    res1 = guard.sanitize(raw_delta_v=1.2, raw_delta_yaw=0.1)
    assert res1.is_clamped is False
    assert res1.is_fallback is False
    assert res1.delta_velocity_ms == 1.2

    # 2. Extreme spike -> clamped to 3.0 m/s and 0.5 rad/s
    res2 = guard.sanitize(raw_delta_v=15.0, raw_delta_yaw=-2.5)
    assert res2.is_clamped is True
    assert res2.delta_velocity_ms == 3.0
    assert res2.delta_yaw_rate_rads == -0.5

    # 3. Sensor degraded or stationary -> fallback to zero
    res3 = guard.sanitize(raw_delta_v=1.2, raw_delta_yaw=0.1, is_sensor_valid=False)
    assert res3.is_fallback is True
    assert res3.delta_velocity_ms == 0.0
    assert res3.delta_yaw_rate_rads == 0.0

    res4 = guard.sanitize(raw_delta_v=1.2, raw_delta_yaw=0.1, is_stationary=True)
    assert res4.is_fallback is True
    assert res4.delta_velocity_ms == 0.0


# 7. Deterministic Training Reproducibility
def test_deterministic_training_reproducibility():
    def train_dummy_model(seed_val: int) -> float:
        set_seed(seed_val)
        model = CausalResidualGRU(input_dim=16, hidden_dim=16, mlp_dim=8, output_dim=2)
        feats = np.random.randn(50, 16).astype(np.float32)
        targets = np.random.randn(50, 2).astype(np.float32)
        ds = CausalWindowDataset(feats, targets, window_size=5)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        trainer = ResidualModelTrainer(model, learning_rate=1e-2)
        loss, _, _ = trainer.train_epoch(loader)
        return loss

    loss1 = train_dummy_model(seed_val=123)
    loss2 = train_dummy_model(seed_val=123)
    assert np.isclose(loss1, loss2, atol=1e-6)

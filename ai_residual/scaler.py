"""
Train-Only Scaler Module for Project AGASTYA (Objective 5).
Fits Z-score normalization statistics exclusively on the training sequence (sync_01)
to strictly prevent data leakage into validation or test sets.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .feature_registry import CANONICAL_FEATURE_NAMES, CANONICAL_FEATURES


@dataclass
class ScalerParams:
    feature_names: List[str]
    means: List[float]
    stds: List[float]
    fitted_sequence_id: str
    num_training_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainOnlyScaler:
    """
    Standardizes features and targets using parameters fitted strictly on training data.
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or CANONICAL_FEATURE_NAMES
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.fitted_sequence_id: Optional[str] = None
        self.num_training_samples: int = 0
        self.is_fitted: bool = False

    def fit(self, data_df: pd.DataFrame, sequence_id: str = "sync_01") -> "TrainOnlyScaler":
        """
        Fit mean and standard deviation strictly on the training sequence.
        """
        self.fitted_sequence_id = sequence_id
        self.num_training_samples = len(data_df)

        matrix = data_df[self.feature_names].to_numpy(dtype=np.float64)
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)

        # For pass-through features (e.g. binary flags), avoid zero std division
        for i, feat_spec in enumerate(CANONICAL_FEATURES):
            if i < len(stds) and (feat_spec.normalization_policy == "PASS_THROUGH" or stds[i] < 1e-8):
                means[i] = 0.0
                stds[i] = 1.0

        # Replace any residual zero stds with 1.0
        stds[stds < 1e-8] = 1.0

        self.means = means
        self.stds = stds
        self.is_fitted = True
        return self

    def transform(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Apply training-set normalization to a DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() on training data first.")
        matrix = data_df[self.feature_names].to_numpy(dtype=np.float64)
        normed = (matrix - self.means) / self.stds
        return normed.astype(np.float32)

    def transform_array(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply training-set normalization to a numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        normed = (matrix - self.means) / self.stds
        return normed.astype(np.float32)

    def inverse_transform(self, normed_matrix: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to physical units.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return (normed_matrix * self.stds + self.means).astype(np.float32)

    def save_json(self, output_path: str) -> None:
        """
        Export scaler parameters to JSON file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        params = ScalerParams(
            feature_names=self.feature_names,
            means=[float(m) for m in self.means],
            stds=[float(s) for s in self.stds],
            fitted_sequence_id=self.fitted_sequence_id or "UNKNOWN",
            num_training_samples=self.num_training_samples
        )
        with open(output_path, "w") as f:
            json.dump(params.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, input_path: str) -> "TrainOnlyScaler":
        """
        Load scaler parameters from JSON file.
        """
        with open(input_path, "r") as f:
            data = json.load(f)
        scaler = cls(feature_names=data["feature_names"])
        scaler.means = np.array(data["means"], dtype=np.float64)
        scaler.stds = np.array(data["stds"], dtype=np.float64)
        scaler.fitted_sequence_id = data.get("fitted_sequence_id", "UNKNOWN")
        scaler.num_training_samples = data.get("num_training_samples", 0)
        scaler.is_fitted = True
        return scaler

    @classmethod
    def load(cls, input_path: str) -> "TrainOnlyScaler":
        return cls.load_json(input_path)


class TargetScaler:
    """
    Standardizes multi-task targets [delta_v, delta_omega] using training-set statistics.
    """
    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names or ["delta_velocity_ms", "delta_yaw_rate_rads"]
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.fitted_sequence_id: Optional[str] = None
        self.is_fitted: bool = False

    def fit(self, targets_matrix: np.ndarray, sequence_id: str = "sync_01") -> "TargetScaler":
        self.fitted_sequence_id = sequence_id
        means = np.mean(targets_matrix, axis=0)
        stds = np.std(targets_matrix, axis=0)
        stds[stds < 1e-8] = 1.0
        self.means = means
        self.stds = stds
        self.is_fitted = True
        return self

    def transform(self, targets_matrix: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TargetScaler is not fitted.")
        return ((targets_matrix - self.means) / self.stds).astype(np.float32)

    def inverse_transform(self, normed_targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TargetScaler is not fitted.")
        return (normed_targets * self.stds + self.means).astype(np.float32)

    def save_json(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "target_names": self.target_names,
            "means": [float(m) for m in self.means],
            "stds": [float(s) for s in self.stds],
            "fitted_sequence_id": self.fitted_sequence_id or "UNKNOWN"
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, input_path: str) -> "TargetScaler":
        with open(input_path, "r") as f:
            data = json.load(f)
        scaler = cls(target_names=data["target_names"])
        scaler.means = np.array(data["means"], dtype=np.float64)
        scaler.stds = np.array(data["stds"], dtype=np.float64)
        scaler.fitted_sequence_id = data.get("fitted_sequence_id", "UNKNOWN")
        scaler.is_fitted = True
        return scaler

    @classmethod
    def load(cls, input_path: str) -> "TargetScaler":
        return cls.load_json(input_path)

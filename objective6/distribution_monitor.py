"""
Training Distribution Monitor and Out-of-Distribution (OOD) Detector for Objective 6.
Calculates normalized statistical distance of causal input features strictly against the training distribution (sync_01).
"""

import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TrainingDistributionMonitor:
    """
    Monitors feature distribution shift and computes an ensemble-free OOD score.
    Fitted strictly on training data (sync_01).
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        ood_threshold: float = 3.5,
        sequence_id: str = "sync_01"
    ):
        self.feature_names = feature_names or []
        self.ood_threshold = ood_threshold
        self.sequence_id = sequence_id
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.p95_distance: float = 0.0
        self.p99_distance: float = 0.0
        self.max_training_distance: float = 0.0
        self.is_fitted: bool = False

    def fit(self, features_df: pd.DataFrame, sequence_id: str = "sync_01") -> "TrainingDistributionMonitor":
        """
        Fit distribution baseline strictly on training sequence.
        """
        self.sequence_id = sequence_id
        self.feature_names = list(features_df.columns)
        mat = features_df.to_numpy(dtype=np.float64)

        self.means = np.nanmean(mat, axis=0)
        self.stds = np.nanstd(mat, axis=0)
        self.stds[self.stds < 1e-6] = 1.0  # Avoid zero division

        # Compute in-sample normalized distances
        z = (mat - self.means) / self.stds
        dists = np.mean(z**2, axis=1)

        self.p95_distance = float(np.percentile(dists, 95))
        self.p99_distance = float(np.percentile(dists, 99))
        self.max_training_distance = float(np.max(dists))
        
        # Set conservative OOD threshold at 3.0 * p95 or 1.5 * p99
        self.ood_threshold = max(3.5, float(self.p99_distance * 1.5))
        self.is_fitted = True
        return self

    def compute_ood_score(self, feature_vector_or_window: np.ndarray) -> float:
        """
        Compute normalized squared Z-score distance from training distribution.
        Handles 1D vector [D] or 2D window [W, D].
        """
        if not self.is_fitted or self.means is None or self.stds is None:
            raise RuntimeError("TrainingDistributionMonitor must be fitted before computing OOD scores.")

        arr = np.asarray(feature_vector_or_window, dtype=np.float64)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return 999.0  # Anomalous / Degraded

        if arr.ndim == 1:
            z = (arr - self.means) / self.stds
            return float(np.mean(z**2))
        elif arr.ndim == 2:
            # For a window [W, D], compute distance on the latest sample (arr[-1])
            # and average across window for stability
            z = (arr - self.means) / self.stds
            curr_dist = float(np.mean(z[-1]**2))
            win_dist = float(np.mean(z**2))
            return 0.7 * curr_dist + 0.3 * win_dist
        else:
            raise ValueError(f"Unsupported array dimension for OOD score: {arr.ndim}")

    def is_in_distribution(self, feature_vector_or_window: np.ndarray) -> bool:
        """
        Check if feature vector/window is within the acceptable distribution boundary.
        """
        score = self.compute_ood_score(feature_vector_or_window)
        return score <= self.ood_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "feature_names": self.feature_names,
            "ood_threshold": round(self.ood_threshold, 4),
            "p95_distance": round(self.p95_distance, 4),
            "p99_distance": round(self.p99_distance, 4),
            "max_training_distance": round(self.max_training_distance, 4),
            "means": [round(float(m), 6) for m in (self.means if self.means is not None else [])],
            "stds": [round(float(s), 6) for s in (self.stds if self.stds is not None else [])],
            "is_fitted": self.is_fitted
        }

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDistributionMonitor":
        mon = cls(
            feature_names=data.get("feature_names", []),
            ood_threshold=data.get("ood_threshold", 3.5),
            sequence_id=data.get("sequence_id", "sync_01")
        )
        mon.means = np.array(data.get("means", []), dtype=np.float64)
        mon.stds = np.array(data.get("stds", []), dtype=np.float64)
        mon.p95_distance = data.get("p95_distance", 0.0)
        mon.p99_distance = data.get("p99_distance", 0.0)
        mon.max_training_distance = data.get("max_training_distance", 0.0)
        mon.is_fitted = data.get("is_fitted", True)
        return mon

    @classmethod
    def load_json(cls, filepath: str) -> "TrainingDistributionMonitor":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

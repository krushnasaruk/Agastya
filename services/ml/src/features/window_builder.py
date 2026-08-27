"""
Causal Sliding Window Generator for Project AGASTYA (Objective 4).
Constructs historical temporal input tensors [N - W + 1, W, D] strictly causally
without future lookahead and with aligned residual targets.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd


class CausalWindowBuilder:
    """
    Constructs causal historical sliding window matrices for sequence modeling.
    """
    @classmethod
    def build_causal_windows(
        cls,
        features_df: pd.DataFrame,
        targets_array: np.ndarray,
        window_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build causal sliding windows of size W.
        
        Parameters:
            features_df: DataFrame of shape [N, D] containing causal features
            targets_array: Array of shape [N] containing aligned residual targets
            window_size: Integer number of past epochs to include (W >= 1)

        Returns:
            X_windows: 3D array of shape [N - W + 1, W, D]
            y_aligned: 1D array of shape [N - W + 1] (target at the current epoch k)
            feature_names: List of column names
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        feature_matrix = features_df.to_numpy(dtype=np.float32)
        n_samples, n_features = feature_matrix.shape

        if n_samples < window_size:
            raise ValueError(f"Sample count ({n_samples}) is less than window_size ({window_size})")

        num_windows = n_samples - window_size + 1
        x_windows = np.empty((num_windows, window_size, n_features), dtype=np.float32)
        y_aligned = np.empty(num_windows, dtype=np.float32)

        for i in range(num_windows):
            # Window index i corresponds to sequence slice [i : i + window_size]
            # Target is at the final epoch of the window (epoch i + window_size - 1)
            x_windows[i] = feature_matrix[i : i + window_size]
            y_aligned[i] = float(targets_array[i + window_size - 1])

        return x_windows, y_aligned, list(features_df.columns)

    @classmethod
    def build_flattened_causal_windows(
        cls,
        features_df: pd.DataFrame,
        targets_array: np.ndarray,
        window_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build 2D flattened causal window matrix of shape [N - W + 1, W * D] for tabular models.
        """
        x_3d, y_aligned, feat_names = cls.build_causal_windows(features_df, targets_array, window_size)
        n_win, w, d = x_3d.shape
        x_flat = x_3d.reshape(n_win, w * d)

        flat_names = []
        for step in range(w):
            offset = step - w + 1  # e.g. -9, -8, ..., 0
            for name in feat_names:
                flat_names.append(f"{name}_lag_{abs(offset)}" if offset < 0 else f"{name}_current")

        return x_flat, y_aligned, flat_names

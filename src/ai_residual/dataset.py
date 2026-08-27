"""
Causal Window PyTorch Dataset for Project AGASTYA (Objective 5).
Generates causal historical window tensors [B, W, 16] and target tensors [B, 2]
without future lookahead or cross-trajectory boundary mixing.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset


class CausalWindowDataset(Dataset):
    """
    PyTorch Dataset yielding strictly causal temporal windows of shape [W, 16]
    and aligned multi-task targets [delta_v, delta_omega] at the current timestep k.
    """
    def __init__(
        self,
        normed_features: np.ndarray,      # Shape [N, 16]
        normed_targets: np.ndarray,       # Shape [N, 2]
        window_size: int = 10,
        timestamps_sec: Optional[np.ndarray] = None,
        sequence_id: str = "unknown"
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        n_samples, n_features = normed_features.shape
        if n_samples < window_size:
            raise ValueError(f"Sample count ({n_samples}) is less than window_size ({window_size})")

        self.window_size = window_size
        self.sequence_id = sequence_id
        self.num_windows = n_samples - window_size + 1

        self.features = normed_features.astype(np.float32)
        self.targets = normed_targets.astype(np.float32)
        self.timestamps = timestamps_sec.astype(np.float64) if timestamps_sec is not None else np.arange(n_samples, dtype=np.float64) * 0.1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            x_window: Tensor of shape [W, 16] (samples from idx to idx + window_size - 1)
            y_target: Tensor of shape [2] (target at current epoch idx + window_size - 1)
            current_time_sec: float timestamp at current epoch
        """
        # Causal window slice: [idx : idx + window_size]
        current_idx = idx + self.window_size - 1
        x_window = self.features[idx : idx + self.window_size]
        y_target = self.targets[current_idx]
        t_curr = float(self.timestamps[current_idx])

        return (
            torch.from_numpy(x_window),
            torch.from_numpy(y_target),
            t_curr
        )

"""
Inference Runner and Execution Engine for Objective 7.
Provides optimized, CPU-first forward passes with watchdog timing and fault injection hooks.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler


class InferenceRunner:
    """
    Executes neural model forward pass with microsecond timing and failure containment.
    """
    def __init__(
        self,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def predict_residual(
        self,
        raw_feature_window: np.ndarray,
        artificial_delay_ms: float = 0.0,
        inject_model_exception: bool = False
    ) -> Tuple[float, float, float]:
        """
        Execute forward pass and return (delta_v_ms, delta_omega_rads, inference_latency_ms).
        """
        if inject_model_exception:
            raise RuntimeError("Injected AI Model Exception for fault testing")

        t_start = time.perf_counter()

        # Handle artificial timeout delays
        if artificial_delay_ms > 0:
            time.sleep(artificial_delay_ms / 1000.0)

        # Normalize window [W, 16]
        norm_win = self.feature_scaler.transform_array(raw_feature_window)
        win_tensor = torch.from_numpy(norm_win).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_norm, _ = self.model(win_tensor)
            out_phys = self.target_scaler.inverse_transform(out_norm.cpu().numpy())[0]

        delta_v = float(out_phys[0])
        delta_w = float(out_phys[1])
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return delta_v, delta_w, latency_ms

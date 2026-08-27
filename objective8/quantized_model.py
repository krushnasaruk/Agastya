"""
Quantized Model Wrapper for Objective 8.
Provides unified, decoupled inference execution across FP32, INT8, and fallback models.
"""

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from ai_residual.scaler import TrainOnlyScaler, TargetScaler


class QuantizedInferenceWrapper:
    """
    Standardized inference wrapper for hardware deployment.
    Accepts 10x16 numpy array, normalizes, executes CPU inference,
    inverses target scaling, and records microsecond inference latency.
    """
    def __init__(
        self,
        model: nn.Module,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        precision_mode: str = "INT8"
    ):
        self.model = model
        self.model.eval()
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.precision_mode = precision_mode.upper()
        self.total_inferences = 0
        self.last_inference_latency_ms = 0.0

    def predict_step(
        self,
        feature_window_10x16: np.ndarray,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, float, Optional[torch.Tensor]]:
        """
        Executes single-step causal inference.
        Returns:
            delta_v_ms: float (unnormalized velocity residual in m/s)
            delta_omega_rads: float (unnormalized yaw residual in rad/s)
            latency_ms: float (measured forward pass latency in ms)
            next_hidden: Optional[torch.Tensor]
        """
        start_time = time.perf_counter()

        # Normalize features using frozen scaler
        norm_window = self.feature_scaler.transform_array(feature_window_10x16)
        x_tensor = torch.from_numpy(norm_window.astype(np.float32)).unsqueeze(0)  # Shape [1, 10, 16]

        with torch.no_grad():
            output, next_hidden = self.model(x_tensor, hidden_state)
            norm_residuals = output[0].cpu().numpy()

        # Denormalize residuals: [delta_v, delta_omega]
        # TargetScaler uses standard mu/sigma inverse
        v_mu = self.target_scaler.mean_[0] if hasattr(self.target_scaler, "mean_") else 0.0
        v_std = self.target_scaler.scale_[0] if hasattr(self.target_scaler, "scale_") else 1.0
        w_mu = self.target_scaler.mean_[1] if hasattr(self.target_scaler, "mean_") else 0.0
        w_std = self.target_scaler.scale_[1] if hasattr(self.target_scaler, "scale_") else 1.0

        delta_v_ms = float(norm_residuals[0] * v_std + v_mu)
        delta_omega_rads = float(norm_residuals[1] * w_std + w_mu)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_inference_latency_ms = latency_ms
        self.total_inferences += 1

        return delta_v_ms, delta_omega_rads, latency_ms, next_hidden

    def get_summary(self) -> Dict[str, Any]:
        return {
            "precision_mode": self.precision_mode,
            "total_inferences": self.total_inferences,
            "last_latency_ms": self.last_inference_latency_ms
        }

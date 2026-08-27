"""
Causal Residual GRU Model for Project AGASTYA (Objective 5).
Small, interpretable temporal neural architecture predicting multi-task residual errors
[delta_v, delta_omega] from strictly causal sensor windows.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class CausalResidualGRU(nn.Module):
    """
    Causal Temporal Residual Estimator:
      Input: [B, W=10, D=16]
      -> Linear Projection (16 -> 64) -> ReLU
      -> GRU (input_size=64, hidden_size=64, num_layers=1, batch_first=True)
      -> Final Hidden State [B, 64]
      -> MLP Head: Linear(64 -> 32) -> ReLU -> Linear(32 -> 2)
      -> Output: [B, 2] (Normalized delta_v, delta_omega)
    """
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        mlp_dim: int = 32,
        output_dim: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.num_gru_layers = num_gru_layers

        # 1. Input Linear Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Causal Recurrent Encoder (GRU)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )

        # 3. Multi-Task Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Parameters:
            x: Tensor of shape [B, W, input_dim]
            h_0: Optional initial hidden state of shape [num_layers, B, hidden_dim]
        Returns:
            out: Tensor of shape [B, output_dim] ([delta_v_norm, delta_omega_norm])
            h_n: Final hidden state [num_layers, B, hidden_dim]
        """
        # Linear projection over sequence
        proj = self.input_proj(x)  # [B, W, hidden_dim]

        # Temporal GRU encoding
        gru_out, h_n = self.gru(proj, h_0)  # gru_out: [B, W, hidden_dim]

        # Use representation at the current (last) causal timestep W-1
        last_hidden = gru_out[:, -1, :]  # [B, hidden_dim]

        # Output predictions
        predictions = self.head(last_hidden)  # [B, 2]

        return predictions, h_n

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_type": "CausalResidualGRU",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "mlp_dim": self.mlp_dim,
            "output_dim": self.output_dim,
            "num_gru_layers": self.num_gru_layers,
            "total_parameters": sum(p.numel() for p in self.parameters())
        }

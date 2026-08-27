"""
Temporal Inertial Transformer for Kinematic Sequence Modeling.
Applies Multi-Head Self-Attention across temporal IMU sequences for dead reckoning regression.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class InertialTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 250
    ):
        super().__init__()

        # Linear input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        # Regression Heads
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        self.bias_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Input x: (Batch, Window, 6)
        Output: {"velocity": (Batch, 3), "bias": (Batch, 3)}
        """
        # Embed
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        # Encode
        encoded = self.transformer_encoder(h)
        encoded = self.norm(encoded)

        # Global average pooling across time
        context = torch.mean(encoded, dim=1)  # (Batch, d_model)

        pred_vel = self.vel_head(context)
        pred_bias = self.bias_head(context)

        return {
            "velocity": pred_vel,
            "bias": pred_bias
        }

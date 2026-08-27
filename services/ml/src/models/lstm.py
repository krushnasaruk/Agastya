"""
Bidirectional LSTM Neural Dead Reckoning Model with Temporal Attention.
Predicts 3D body-frame velocity vectors and sensor bias residuals from sequential IMU windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    Computes scalar attention weights over the sequence dimension
    and returns a context vector as a weighted sum of hidden states.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Hidden_Dim)
        energy = self.projection(x)  # (Batch, Seq_Len, 1)
        weights = F.softmax(energy, dim=1)  # (Batch, Seq_Len, 1)
        context = torch.sum(x * weights, dim=1)  # (Batch, Hidden_Dim)
        return context


class BiLSTMDeadReckoning(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # 1D Convolutional Front-end (Extract local temporal dynamics)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.1),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Self-Attention Pooling
        total_hidden = lstm_hidden * 2  # Bidirectional
        self.attention = TemporalSelfAttention(total_hidden)

        # Velocity Regression Head (3D: vx, vy, vz)
        self.vel_head = nn.Sequential(
            nn.Linear(total_hidden, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3)
        )

        # Bias Regression Head (3D: ba_x, ba_y, ba_z)
        self.bias_head = nn.Sequential(
            nn.Linear(total_hidden, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Input x: (Batch, Window, 6)
        Output dict: {"velocity": (Batch, 3), "bias": (Batch, 3)}
        """
        # Permute for 1D Conv: (Batch, 6, Window)
        x_conv = self.conv(x.transpose(1, 2))
        x_lstm_in = x_conv.transpose(1, 2)  # (Batch, Window, conv_channels)

        lstm_out, _ = self.lstm(x_lstm_in)  # (Batch, Window, total_hidden)
        context = self.attention(lstm_out)   # (Batch, total_hidden)

        pred_vel = self.vel_head(context)
        pred_bias = self.bias_head(context)

        return {
            "velocity": pred_vel,
            "bias": pred_bias
        }

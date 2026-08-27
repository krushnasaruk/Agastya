"""
Neural Inertial Dead Reckoning Model Training Script.
Trains BiLSTM or Inertial Transformer with Huber loss and Directional Cosine Loss.
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.loader import generate_synthetic_flight_dataset
from src.models.lstm import BiLSTMDeadReckoning
from src.models.transformer import InertialTransformer


class DirectionalCosineLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1, eps=self.eps)
        return torch.mean(1.0 - cos_sim)


def train_model(
    model_type: str = "bilstm",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "models/bilstm_dead_reckoning.pt",
    device: str = "cpu"
):
    print("=" * 70)
    print(f"AGASTYA Neural Dead Reckoning Training Pipeline [{model_type.upper()}]")
    print("=" * 70)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # 1. Prepare Datasets
    print("Generating synthetic multi-maneuver inertial dataset...")
    train_ds, val_ds = generate_synthetic_flight_dataset(
        num_trajectories=15,
        trajectory_duration_sec=30.0,
        dt=0.01,
        window_size=100,
        step_size=5
    )
    print(f"Dataset ready: {len(train_ds)} train samples, {len(val_ds)} val samples.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    if model_type == "bilstm":
        model = BiLSTMDeadReckoning(input_dim=6, lstm_hidden=128, num_lstm_layers=2)
    elif model_type == "transformer":
        model = InertialTransformer(input_dim=6, d_model=128, nhead=8, num_layers=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 3. Loss Functions & Optimizer
    huber_loss = nn.HuberLoss(delta=1.0)
    cosine_loss = DirectionalCosineLoss()
    bias_mse_loss = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_vel_loss = 0.0

        for batch in train_loader:
            x_imu = batch["imu"].to(device)
            y_vel = batch["velocity"].to(device)

            optimizer.zero_grad()
            outputs = model(x_imu)
            pred_vel = outputs["velocity"]

            l_huber = huber_loss(pred_vel, y_vel)
            l_cos = cosine_loss(pred_vel, y_vel)
            loss = l_huber + 0.5 * l_cos

            if "bias" in batch and "bias" in outputs:
                y_bias = batch["bias"].to(device)
                loss += 0.2 * bias_mse_loss(outputs["bias"], y_bias)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_vel_loss += l_huber.item()

        scheduler.step()
        train_loss /= len(train_loader)
        train_vel_loss /= len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_rmse_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_imu = batch["imu"].to(device)
                y_vel = batch["velocity"].to(device)

                outputs = model(x_imu)
                pred_vel = outputs["velocity"]

                l_huber = huber_loss(pred_vel, y_vel)
                l_cos = cosine_loss(pred_vel, y_vel)
                v_loss = l_huber + 0.5 * l_cos
                val_loss += v_loss.item()

                rmse = torch.sqrt(torch.mean((pred_vel - y_vel) ** 2))
                val_rmse_sum += rmse.item()

        val_loss /= len(val_loader)
        val_rmse = val_rmse_sum / len(val_loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Vel RMSE: {val_rmse:.3f} m/s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_type": model_type
            }, save_path)

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.2f}s.")
    print(f"Saved best model checkpoint to: {save_path} (Best Val Loss: {best_val_loss:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Dead Reckoning Model")
    parser.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="services/ml/models/bilstm_dead_reckoning.pt")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )

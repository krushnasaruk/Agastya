"""
Deterministic Multi-Task Trainer for Project AGASTYA (Objective 5).
Executes reproducible PyTorch training with early stopping strictly monitored on validation loss.
"""

import os
import json
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CausalResidualGRU
from .dataset import CausalWindowDataset


def set_seed(seed: int = 42) -> None:
    """Set global seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResidualModelTrainer:
    """
    Multi-task trainer for CausalResidualGRU with validation early stopping.
    """
    def __init__(
        self,
        model: CausalResidualGRU,
        learning_rate: float = 1e-3,
        lambda_v: float = 1.0,
        lambda_omega: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.lambda_v = lambda_v
        self.lambda_omega = lambda_omega
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        loss_v_sum = 0.0
        loss_w_sum = 0.0
        num_batches = len(dataloader)

        for x_win, y_true, _ in dataloader:
            x_win = x_win.to(self.device)
            y_true = y_true.to(self.device)

            self.optimizer.zero_grad()
            y_pred, _ = self.model(x_win)

            l_v = self.criterion(y_pred[:, 0], y_true[:, 0])
            l_w = self.criterion(y_pred[:, 1], y_true[:, 1])
            loss = self.lambda_v * l_v + self.lambda_omega * l_w

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loss_v_sum += l_v.item()
            loss_w_sum += l_w.item()

        return (
            total_loss / max(num_batches, 1),
            loss_v_sum / max(num_batches, 1),
            loss_w_sum / max(num_batches, 1)
        )

    def evaluate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        loss_v_sum = 0.0
        loss_w_sum = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for x_win, y_true, _ in dataloader:
                x_win = x_win.to(self.device)
                y_true = y_true.to(self.device)

                y_pred, _ = self.model(x_win)
                l_v = self.criterion(y_pred[:, 0], y_true[:, 0])
                l_w = self.criterion(y_pred[:, 1], y_true[:, 1])
                loss = self.lambda_v * l_v + self.lambda_omega * l_w

                total_loss += loss.item()
                loss_v_sum += l_v.item()
                loss_w_sum += l_w.item()

        return (
            total_loss / max(num_batches, 1),
            loss_v_sum / max(num_batches, 1),
            loss_w_sum / max(num_batches, 1)
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 15,
        checkpoint_dir: str = "artifacts/objective5",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping on validation loss.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_loss_v": [],
            "val_loss_v": [],
            "train_loss_omega": [],
            "val_loss_omega": []
        }

        for epoch in range(1, max_epochs + 1):
            t_loss, t_v, t_w = self.train_epoch(train_loader)
            v_loss, v_v, v_w = self.evaluate_epoch(val_loader)

            history["epochs"].append(epoch)
            history["train_loss"].append(round(t_loss, 6))
            history["val_loss"].append(round(v_loss, 6))
            history["train_loss_v"].append(round(t_v, 6))
            history["val_loss_v"].append(round(v_v, 6))
            history["train_loss_omega"].append(round(t_w, 6))
            history["val_loss_omega"].append(round(v_w, 6))

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d}/{max_epochs:03d} | Train Loss: {t_loss:.5f} (v: {t_v:.5f}, w: {t_w:.5f}) | Val Loss: {v_loss:.5f} (v: {v_v:.5f}, w: {v_w:.5f})")

            # Checkpoint best model on validation loss
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[Early Stopping] Triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss: {best_val_loss:.5f}")
                    break

        # Save training history JSON
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 6),
                "history": history
            }, f, indent=2)

        # Load best weights back into model
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_model_path": best_model_path,
            "history": history
        }

"""
Residual Metrics Evaluation Module for Project AGASTYA (Objective 5).
Calculates MAE, RMSE, Bias, R², and Pearson correlation for velocity and yaw residuals.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import CausalResidualGRU
from .scaler import TargetScaler


@dataclass
class ResidualMetrics:
    target_name: str
    num_samples: int
    mae: float
    rmse: float
    bias: float
    r2_score: float
    pearson_correlation: float
    trivial_zero_rmse: float          # Baseline error if model predicted delta = 0 everywhere
    improvement_over_zero_pct: float  # Percentage error reduction over zero predictor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResidualEvaluator:
    """
    Evaluates PyTorch model residual predictions in true physical units.
    """
    @classmethod
    def evaluate_dataset(
        cls,
        model: CausalResidualGRU,
        dataloader: DataLoader,
        target_scaler: TargetScaler,
        device: Optional[torch.device] = None
    ) -> Tuple[Dict[str, ResidualMetrics], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model predictions and return physical metrics, true targets, predicted targets, and timestamps.
        """
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.eval()
        model.to(dev)

        all_preds = []
        all_trues = []
        all_times = []

        with torch.no_grad():
            for x_win, y_true, t_curr in dataloader:
                x_win = x_win.to(dev)
                preds, _ = model(x_win)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(y_true.numpy())
                all_times.append(t_curr.numpy())

        y_pred_norm = np.vstack(all_preds)
        y_true_norm = np.vstack(all_trues)
        timestamps = np.concatenate(all_times)

        # Unscale back to physical SI units
        y_pred_phys = target_scaler.inverse_transform(y_pred_norm)
        y_true_phys = target_scaler.inverse_transform(y_true_norm)

        # 1. Velocity Residual Metrics (Target A)
        v_true = y_true_phys[:, 0]
        v_pred = y_pred_phys[:, 0]
        v_mae = float(np.mean(np.abs(v_true - v_pred)))
        v_rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))
        v_bias = float(np.mean(v_pred - v_true))
        v_zero_rmse = float(np.sqrt(np.mean(v_true ** 2)))
        v_ss_tot = np.sum((v_true - np.mean(v_true)) ** 2)
        v_ss_res = np.sum((v_true - v_pred) ** 2)
        v_r2 = float(1.0 - (v_ss_res / max(v_ss_tot, 1e-9)))
        if np.std(v_true) > 1e-9 and np.std(v_pred) > 1e-9:
            v_r = float(np.corrcoef(v_true, v_pred)[0, 1])
        else:
            v_r = 0.0
        v_imp = float(((v_zero_rmse - v_rmse) / max(v_zero_rmse, 1e-9)) * 100.0)

        v_metrics = ResidualMetrics(
            target_name="delta_velocity_ms",
            num_samples=len(v_true),
            mae=round(v_mae, 5),
            rmse=round(v_rmse, 5),
            bias=round(v_bias, 5),
            r2_score=round(v_r2, 4),
            pearson_correlation=round(v_r, 4),
            trivial_zero_rmse=round(v_zero_rmse, 5),
            improvement_over_zero_pct=round(v_imp, 2)
        )

        # 2. Yaw Rate Residual Metrics (Target B1)
        w_true = y_true_phys[:, 1]
        w_pred = y_pred_phys[:, 1]
        w_mae = float(np.mean(np.abs(w_true - w_pred)))
        w_rmse = float(np.sqrt(np.mean((w_true - w_pred) ** 2)))
        w_bias = float(np.mean(w_pred - w_true))
        w_zero_rmse = float(np.sqrt(np.mean(w_true ** 2)))
        w_ss_tot = np.sum((w_true - np.mean(w_true)) ** 2)
        w_ss_res = np.sum((w_true - w_pred) ** 2)
        w_r2 = float(1.0 - (w_ss_res / max(w_ss_tot, 1e-9)))
        if np.std(w_true) > 1e-9 and np.std(w_pred) > 1e-9:
            w_r = float(np.corrcoef(w_true, w_pred)[0, 1])
        else:
            w_r = 0.0
        w_imp = float(((w_zero_rmse - w_rmse) / max(w_zero_rmse, 1e-9)) * 100.0)

        w_metrics = ResidualMetrics(
            target_name="delta_yaw_rate_rads",
            num_samples=len(w_true),
            mae=round(w_mae, 5),
            rmse=round(w_rmse, 5),
            bias=round(w_bias, 5),
            r2_score=round(w_r2, 4),
            pearson_correlation=round(w_r, 4),
            trivial_zero_rmse=round(w_zero_rmse, 5),
            improvement_over_zero_pct=round(w_imp, 2)
        )

        results = {
            "delta_velocity_ms": v_metrics,
            "delta_yaw_rate_rads": w_metrics
        }
        return results, y_true_phys, y_pred_phys, timestamps

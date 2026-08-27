"""
Ensemble-Free Predictive Uncertainty and Confidence Estimation for Objective 6.
Combines feature-space distance, temporal fluctuation, and normalized residual scale
into a deterministic confidence score in [0.0, 1.0].
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class PredictiveConfidenceEstimator:
    """
    Deterministic confidence estimator and uncertainty proxy.
    Requires no ensemble, no second network, and executes with sub-millisecond latency.
    """
    def __init__(
        self,
        min_confidence_threshold: float = 0.45,
        target_v_scale: float = 0.05,
        target_w_scale: float = 0.05
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.target_v_scale = max(target_v_scale, 1e-4)
        self.target_w_scale = max(target_w_scale, 1e-4)

    def estimate_confidence(
        self,
        raw_delta_v: float,
        raw_delta_w: float,
        ood_score: float,
        ood_threshold: float,
        v_jump: float,
        max_v_jump: float,
        is_stationary: bool = False,
        is_sensor_valid: bool = True
    ) -> Dict[str, Any]:
        """
        Compute uncertainty components and unified confidence score in [0.0, 1.0].
        """
        if not is_sensor_valid or np.isnan(raw_delta_v) or np.isinf(raw_delta_v):
            return {
                "confidence": 0.0,
                "uncertainty": 1.0,
                "is_confident": False,
                "confidence_tier": "LOW",
                "u_ood": 1.0,
                "u_temp": 1.0,
                "u_mag": 1.0
            }

        if is_stationary:
            return {
                "confidence": 0.0,
                "uncertainty": 1.0,
                "is_confident": False,
                "confidence_tier": "LOW",
                "u_ood": 0.0,
                "u_temp": 0.0,
                "u_mag": 0.0
            }

        # 1. Distribution distance uncertainty [0..1]
        u_ood = float(np.clip(ood_score / max(ood_threshold, 1e-3), 0.0, 1.0))

        # 2. Temporal jump uncertainty [0..1]
        u_temp = float(np.clip(v_jump / max(max_v_jump, 1e-3), 0.0, 1.0))

        # 3. Residual scale magnitude uncertainty [0..1]
        norm_v_mag = abs(raw_delta_v) / (self.target_v_scale * 3.0)
        u_mag = float(np.clip(norm_v_mag, 0.0, 1.0))

        # Unified uncertainty proxy
        uncertainty = 0.40 * u_ood + 0.35 * u_temp + 0.25 * u_mag
        confidence = float(np.clip(1.0 - uncertainty, 0.0, 1.0))

        if confidence >= 0.75:
            tier = "HIGH"
        elif confidence >= 0.50:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        is_confident = (confidence >= self.min_confidence_threshold)

        return {
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "is_confident": is_confident,
            "confidence_tier": tier,
            "u_ood": round(u_ood, 4),
            "u_temp": round(u_temp, 4),
            "u_mag": round(u_mag, 4)
        }

    @classmethod
    def evaluate_calibration(
        cls,
        confidences: np.ndarray,
        absolute_errors: np.ndarray,
        num_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate empirical relationship between confidence and prediction error.
        """
        valid_mask = ~(np.isnan(confidences) | np.isnan(absolute_errors))
        c_clean = confidences[valid_mask]
        e_clean = absolute_errors[valid_mask]

        if len(c_clean) < 4:
            return {
                "correlation_pearson": 0.0,
                "correlation_spearman": 0.0,
                "status": "INSUFFICIENT_DATA",
                "bins": []
            }

        # Pearson correlation (ideal: negative, higher confidence -> lower error)
        r_corr = float(np.corrcoef(c_clean, e_clean)[0, 1]) if np.std(c_clean) > 1e-6 and np.std(e_clean) > 1e-6 else 0.0

        # Bin evaluation
        bins = []
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        for i in range(num_bins):
            b_low, b_high = bin_edges[i], bin_edges[i+1]
            in_bin = (c_clean >= b_low) & (c_clean <= b_high if i == num_bins - 1 else c_clean < b_high)
            count = int(np.sum(in_bin))
            if count > 0:
                mean_conf = float(np.mean(c_clean[in_bin]))
                mean_err = float(np.mean(e_clean[in_bin]))
                rmse_err = float(np.sqrt(np.mean(e_clean[in_bin]**2)))
            else:
                mean_conf = float((b_low + b_high) / 2.0)
                mean_err = 0.0
                rmse_err = 0.0

            bins.append({
                "bin_index": i,
                "range": [round(float(b_low), 2), round(float(b_high), 2)],
                "sample_count": count,
                "mean_confidence": round(mean_conf, 4),
                "mean_absolute_error": round(mean_err, 5),
                "rmse_error": round(rmse_err, 5)
            })

        # Monotonicity check
        valid_errs = [b["mean_absolute_error"] for b in bins if b["sample_count"] > 5]
        is_monotonic_decreasing = all(valid_errs[j] >= valid_errs[j+1] for j in range(len(valid_errs)-1)) if len(valid_errs) >= 2 else False

        if r_corr < -0.20 or is_monotonic_decreasing:
            calib_status = "PARTIALLY CALIBRATED"
        elif r_corr < -0.50:
            calib_status = "CALIBRATED"
        else:
            calib_status = "WEAKLY CALIBRATED / UNCORRELATED"

        return {
            "correlation_pearson": round(r_corr, 4),
            "calibration_status": calib_status,
            "bins": bins
        }

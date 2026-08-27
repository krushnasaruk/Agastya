"""
Comprehensive Statistical and Navigation Metrics Engine for Objective 6.
Calculates statistical moments, MAD, lag-1 autocorrelation, outlier rates, and navigation drift.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class Objective6MetricsCalculator:
    """
    Computes rigorous statistical distributions and navigation accuracy metrics.
    """
    @classmethod
    def compute_distribution_statistics(cls, values: np.ndarray, name: str = "signal") -> Dict[str, Any]:
        """
        Compute full statistical profile: mean, std, median, MAD, min, max, p95, outlier rate, lag-1 autocorrelation.
        """
        clean = values[~np.isnan(values)]
        if len(clean) == 0:
            return {
                "name": name,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "mad": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "outlier_pct": 0.0,
                "lag1_autocorrelation": 0.0
            }

        mean_val = float(np.mean(clean))
        std_val = float(np.std(clean))
        med_val = float(np.median(clean))
        mad_val = float(np.median(np.abs(clean - med_val)))
        min_val = float(np.min(clean))
        max_val = float(np.max(clean))
        p95_val = float(np.percentile(np.abs(clean), 95))

        # Robust outlier detection (> 3.5 * MAD from median, or > 3 * std if MAD is 0)
        threshold = 3.5 * mad_val if mad_val > 1e-6 else max(3.0 * std_val, 1e-6)
        outliers = np.abs(clean - med_val) > threshold
        outlier_pct = float((np.sum(outliers) / len(clean)) * 100.0)

        # Lag-1 Autocorrelation
        if len(clean) > 2 and std_val > 1e-6:
            c_centered = clean - mean_val
            lag1_num = np.sum(c_centered[1:] * c_centered[:-1])
            lag1_den = np.sum(c_centered**2)
            lag1_corr = float(lag1_num / max(lag1_den, 1e-9))
        else:
            lag1_corr = 0.0

        return {
            "name": name,
            "count": len(clean),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "median": round(med_val, 6),
            "mad": round(mad_val, 6),
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "p95": round(p95_val, 6),
            "outlier_pct": round(outlier_pct, 2),
            "lag1_autocorrelation": round(lag1_corr, 4)
        }

    @classmethod
    def compute_maneuver_stratified_metrics(
        cls,
        maneuver_labels: np.ndarray,
        pos_errors_m: np.ndarray,
        head_errors_rad: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stratify position and heading errors by physical driving regime.
        """
        unique_labels = np.unique(maneuver_labels)
        stratified = {}

        for m_name in unique_labels:
            mask = (maneuver_labels == m_name)
            count = int(np.sum(mask))
            if count == 0:
                continue

            m_pos_err = pos_errors_m[mask]
            ate_rmse = float(np.sqrt(np.mean(m_pos_err**2)))
            max_err = float(np.max(m_pos_err))
            mean_err = float(np.mean(m_pos_err))

            h_rmse_deg = 0.0
            if head_errors_rad is not None:
                m_h_err = head_errors_rad[mask]
                h_rmse_deg = float(np.degrees(np.sqrt(np.mean(m_h_err**2))))

            stratified[str(m_name)] = {
                "sample_count": count,
                "ate_rmse_m": round(ate_rmse, 4),
                "mean_error_m": round(mean_err, 4),
                "max_error_m": round(max_err, 4),
                "heading_rmse_deg": round(h_rmse_deg, 4)
            }

        return stratified

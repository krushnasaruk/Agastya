"""
Statistical Diagnostics and Autocorrelation Suite for Residual Targets (Objective 4).
Computes robust statistical moments, percentiles, outlier ratios, and autocorrelation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class TargetStatistics:
    target_name: str
    num_samples: int
    mean: float
    std: float
    median: float
    mad: float                       # Median Absolute Deviation
    min_val: float
    max_val: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float
    outlier_ratio_pct: float         # Fraction of samples > 3 * MAD from median
    autocorr_lag1: float             # Lag-1 temporal autocorrelation
    autocorr_lag5: float             # Lag-5 temporal autocorrelation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResidualStatisticsAnalyzer:
    """
    Computes rigorous statistical descriptors and correlation profiles for candidate residual targets.
    """
    @classmethod
    def analyze_target(cls, target_array: np.ndarray, target_name: str) -> TargetStatistics:
        """
        Compute comprehensive statistics for a single residual target array.
        """
        valid_data = target_array[~np.isnan(target_array)]
        n = len(valid_data)
        if n == 0:
            raise ValueError(f"Target '{target_name}' contains zero valid numeric samples.")

        mean_val = float(np.mean(valid_data))
        std_val = float(np.std(valid_data))
        med_val = float(np.median(valid_data))
        mad_val = float(np.median(np.abs(valid_data - med_val)))
        min_v = float(np.min(valid_data))
        max_v = float(np.max(valid_data))

        p10 = float(np.percentile(valid_data, 10))
        p25 = float(np.percentile(valid_data, 25))
        p75 = float(np.percentile(valid_data, 75))
        p90 = float(np.percentile(valid_data, 90))
        p95 = float(np.percentile(valid_data, 95))
        p99 = float(np.percentile(valid_data, 99))

        # Outlier ratio based on 3 * MAD
        threshold = max(3.0 * mad_val, 1e-6)
        outliers = np.sum(np.abs(valid_data - med_val) > threshold)
        outlier_pct = float((outliers / n) * 100.0)

        # Autocorrelation (Lag-1 and Lag-5)
        if std_val > 1e-12 and n > 5:
            centered = valid_data - mean_val
            var = np.sum(centered ** 2)
            acf1 = float(np.sum(centered[:-1] * centered[1:]) / var)
            acf5 = float(np.sum(centered[:-5] * centered[5:]) / var)
        else:
            acf1 = 0.0
            acf5 = 0.0

        return TargetStatistics(
            target_name=target_name,
            num_samples=n,
            mean=round(mean_val, 6),
            std=round(std_val, 6),
            median=round(med_val, 6),
            mad=round(mad_val, 6),
            min_val=round(min_v, 6),
            max_val=round(max_v, 6),
            p10=round(p10, 6),
            p25=round(p25, 6),
            p75=round(p75, 6),
            p90=round(p90, 6),
            p95=round(p95, 6),
            p99=round(p99, 6),
            outlier_ratio_pct=round(outlier_pct, 2),
            autocorr_lag1=round(acf1, 4),
            autocorr_lag5=round(acf5, 4)
        )

    @classmethod
    def compute_feature_correlations(
        cls,
        features_df: pd.DataFrame,
        target_series: np.ndarray,
        target_name: str
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation coefficients between causal features and a target residual stream.
        """
        correlations: Dict[str, float] = {}
        t_clean = np.asarray(target_series)
        
        for col in features_df.columns:
            f_col = features_df[col].to_numpy()
            valid_mask = (~np.isnan(f_col)) & (~np.isnan(t_clean))
            if np.sum(valid_mask) > 10 and np.std(f_col[valid_mask]) > 1e-9 and np.std(t_clean[valid_mask]) > 1e-9:
                r = float(np.corrcoef(f_col[valid_mask], t_clean[valid_mask])[0, 1])
                correlations[col] = round(r, 4)
            else:
                correlations[col] = 0.0

        return correlations

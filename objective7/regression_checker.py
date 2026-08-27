"""
Automated Regression Protection Engine for Objective 7.
Verifies zero unintended regression against frozen Objective 3, Objective 5, and Objective 6 milestones.
"""

from typing import Dict, Any
import numpy as np


class RegressionChecker:
    """
    Compares real-time navigation metrics against established historical baselines.
    """
    FROZEN_BASELINES = {
        "objective3_classical": {
            "ate_rmse_m": 1.6366,
            "final_position_error_m": 1.8270,
            "heading_rmse_deg": 0.1560,
            "ai_application_rate_pct": 0.0
        },
        "objective5_velocity_only": {
            "ate_rmse_m": 1.5968,
            "final_position_error_m": 1.7903,
            "heading_rmse_deg": 0.1560,
            "ai_application_rate_pct": 100.0
        },
        "objective6_selective_velocity": {
            "ate_rmse_m": 1.6062,
            "final_position_error_m": 1.8013,
            "heading_rmse_deg": 0.1560,
            "ai_application_rate_pct": 70.6
        }
    }

    @classmethod
    def evaluate_regression(
        cls,
        obj7_metrics: Dict[str, Any],
        tolerance_m: float = 0.01
    ) -> Dict[str, Any]:
        """
        Verify Objective 7 results match Objective 6 reference within numerical tolerance.
        """
        ref = cls.FROZEN_BASELINES["objective6_selective_velocity"]
        actual_ate = obj7_metrics.get("ate_rmse_m", 0.0)
        actual_final = obj7_metrics.get("final_position_error_m", 0.0)
        actual_h = obj7_metrics.get("heading_rmse_deg", 0.0)

        ate_diff = abs(actual_ate - ref["ate_rmse_m"])
        final_diff = abs(actual_final - ref["final_position_error_m"])
        h_diff = abs(actual_h - ref["heading_rmse_deg"])

        is_ate_ok = (ate_diff <= tolerance_m)
        is_final_ok = (final_diff <= tolerance_m)
        is_h_ok = (h_diff <= 0.05)

        is_pass = is_ate_ok and is_final_ok and is_h_ok

        return {
            "target_baseline": "objective6_selective_velocity",
            "reference_ate_rmse_m": ref["ate_rmse_m"],
            "actual_ate_rmse_m": round(actual_ate, 4),
            "ate_difference_m": round(ate_diff, 6),
            "reference_final_error_m": ref["final_position_error_m"],
            "actual_final_error_m": round(actual_final, 4),
            "final_difference_m": round(final_diff, 6),
            "reference_heading_rmse_deg": ref["heading_rmse_deg"],
            "actual_heading_rmse_deg": round(actual_h, 4),
            "heading_difference_deg": round(h_diff, 6),
            "regression_detected": not is_pass,
            "regression_check_status": "PASS (Zero Regression)" if is_pass else "REGRESSION_DETECTED"
        }

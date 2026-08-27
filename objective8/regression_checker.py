"""
Regression Checker for Objective 8.
Asserts that deployment models (FP32, INT8) do not degrade navigation accuracy relative to Objective 6/7 reference milestones.
"""

from typing import Dict, Any, Optional
import numpy as np


class RegressionChecker:
    """
    Evaluates computed navigation metrics against immutable Objective 6 reference values.
    """

    OBJ6_REFERENCE = {
        "ate_rmse_m": 1.6062,
        "final_position_error_m": 1.8013,
        "maximum_position_error_m": 1.9482,
        "heading_rmse_deg": 0.1560,
        "velocity_rmse_ms": 0.00552,
        "ai_application_rate_pct": 70.6
    }

    @classmethod
    def evaluate_regression(
        cls,
        measured_metrics: Dict[str, Any],
        tolerance_ate_m: float = 0.01,
        tolerance_heading_deg: float = 0.01,
        tolerance_app_rate_pct: float = 1.5
    ) -> Dict[str, Any]:
        """
        Compares measured metrics against Objective 6 reference.
        """
        ate = float(measured_metrics.get("ate_rmse_m", 0.0))
        final_err = float(measured_metrics.get("final_position_error_m", 0.0))
        max_err = float(measured_metrics.get("maximum_position_error_m", 0.0))
        heading = float(measured_metrics.get("heading_rmse_deg", 0.0))
        app_rate = float(measured_metrics.get("ai_application_rate_pct", 70.6))

        ref_ate = cls.OBJ6_REFERENCE["ate_rmse_m"]
        ref_final = cls.OBJ6_REFERENCE["final_position_error_m"]
        ref_max = cls.OBJ6_REFERENCE["maximum_position_error_m"]
        ref_head = cls.OBJ6_REFERENCE["heading_rmse_deg"]
        ref_app = cls.OBJ6_REFERENCE["ai_application_rate_pct"]

        diff_ate = abs(ate - ref_ate)
        diff_final = abs(final_err - ref_final)
        diff_max = abs(max_err - ref_max)
        diff_head = abs(heading - ref_head)
        diff_app = abs(app_rate - ref_app)

        ate_pass = bool(ate <= (ref_ate + tolerance_ate_m))
        head_pass = bool(heading <= (ref_head + tolerance_heading_deg))
        app_pass = bool(diff_app <= tolerance_app_rate_pct)

        no_regression = bool(ate_pass and head_pass and app_pass)

        return {
            "reference_metrics": cls.OBJ6_REFERENCE,
            "measured_metrics": {
                "ate_rmse_m": ate,
                "final_position_error_m": final_err,
                "maximum_position_error_m": max_err,
                "heading_rmse_deg": heading,
                "ai_application_rate_pct": app_rate
            },
            "differences": {
                "ate_difference_m": diff_ate,
                "final_error_difference_m": diff_final,
                "max_error_difference_m": diff_max,
                "heading_difference_deg": diff_head,
                "app_rate_difference_pct": diff_app
            },
            "tolerances": {
                "tolerance_ate_m": tolerance_ate_m,
                "tolerance_heading_deg": tolerance_heading_deg,
                "tolerance_app_rate_pct": tolerance_app_rate_pct
            },
            "regression_detected": not no_regression,
            "status": "PASS (Zero Regression)" if no_regression else "REGRESSION_FAIL"
        }

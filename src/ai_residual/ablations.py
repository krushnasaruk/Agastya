"""
Scientific Ablation Study Module for Project AGASTYA (Objective 5).
Evaluates 4 controlled configurations:
  - Ablation A: Classical Only (No AI)
  - Ablation B: Velocity Correction Only (delta_v enabled, delta_omega = 0)
  - Ablation C: Yaw Correction Only (delta_v = 0, delta_omega enabled)
  - Ablation D: Full Correction (delta_v + delta_omega)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from navigation_engine.evaluation import DeadReckoningEvaluator, NavigationMetrics
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from .rollout import AIRolloutEngine
from .model import CausalResidualGRU
from .scaler import TrainOnlyScaler, TargetScaler
from .safety import SafetyGuard


class AblationRunner:
    """
    Executes controlled scientific ablations on a test sequence.
    """
    @classmethod
    def run_ablations(
        cls,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        nav_inputs_df: pd.DataFrame,
        causal_features_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        sequence_id: str = "sync_02",
        device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all 4 ablations and return metrics dictionary.
        """
        init_heading = float(ref_df["heading_rad"].iloc[0]) if "heading_rad" in ref_df else 0.0
        init_east = float(ref_df["pos_east_m"].iloc[0]) if "pos_east_m" in ref_df else 0.0
        init_north = float(ref_df["pos_north_m"].iloc[0]) if "pos_north_m" in ref_df else 0.0

        ref_east = ref_df["pos_east_m"].to_numpy()
        ref_north = ref_df["pos_north_m"].to_numpy()
        ref_h = ref_df.get("heading_rad", None)
        ref_v = ref_df.get("ground_speed_ms", None)

        configurations = [
            ("A_classical_only", False, False),
            ("B_velocity_only", True, False),
            ("C_yaw_only", False, True),
            ("D_full_correction", True, True)
        ]

        results = {}

        for name, en_v, en_w in configurations:
            engine = AIRolloutEngine(
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                safety_guard=SafetyGuard(),
                window_size=10,
                enable_velocity_correction=en_v,
                enable_yaw_correction=en_w,
                device=device
            )
            traj = engine.run_rollout(
                nav_inputs_df,
                causal_features_df,
                initial_p_east_m=init_east,
                initial_p_north_m=init_north,
                initial_heading_rad=init_heading
            )
            metrics, pos_err, head_err = DeadReckoningEvaluator.evaluate(
                traj, ref_east, ref_north, ref_h, ref_v
            )
            results[name] = {
                "metrics": metrics.to_dict(),
                "trajectory": traj,
                "pos_errors": pos_err,
                "head_errors": head_err,
                "velocity_enabled": en_v,
                "yaw_enabled": en_w
            }

        return results

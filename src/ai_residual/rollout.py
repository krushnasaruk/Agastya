"""
Closed-Loop Navigation Rollout Engine for Project AGASTYA (Objective 5).
Fuses Baseline A deterministic physics state with guarded AI residual corrections
to produce corrected navigation trajectories.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch

from navigation_engine.state import DeadReckoningTrajectory, wrap_to_2pi
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from .model import CausalResidualGRU
from .scaler import TrainOnlyScaler, TargetScaler
from .safety import SafetyGuard
from .feature_registry import CANONICAL_FEATURE_NAMES


class AIRolloutEngine:
    """
    Executes dead-reckoning trajectory propagation augmented by guarded AI residual corrections.
    """
    def __init__(
        self,
        model: Optional[CausalResidualGRU] = None,
        feature_scaler: Optional[TrainOnlyScaler] = None,
        target_scaler: Optional[TargetScaler] = None,
        safety_guard: Optional[SafetyGuard] = None,
        window_size: int = 10,
        enable_velocity_correction: bool = True,
        enable_yaw_correction: bool = True,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.safety_guard = safety_guard or SafetyGuard()
        self.window_size = window_size
        self.enable_v = enable_velocity_correction
        self.enable_w = enable_yaw_correction

    def run_rollout(
        self,
        navigation_inputs_df: pd.DataFrame,
        causal_features_df: pd.DataFrame,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0
    ) -> DeadReckoningTrajectory:
        """
        Execute full causal rollout over sequence.
        """
        n = len(navigation_inputs_df)
        t_arr = navigation_inputs_df["time_sec"].to_numpy()
        dt_arr = navigation_inputs_df["dt_sec"].to_numpy()

        p_east = np.zeros(n, dtype=np.float64)
        p_north = np.zeros(n, dtype=np.float64)
        headings = np.zeros(n, dtype=np.float64)
        speeds = np.zeros(n, dtype=np.float64)
        yaw_rates = np.zeros(n, dtype=np.float64)

        # Baseline A engine to supply underlying classical physics baseline
        classical_engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
        classical_engine.initialize(
            initial_p_east_m=initial_p_east_m,
            initial_p_north_m=initial_p_north_m,
            initial_heading_rad=initial_heading_rad,
            initial_time_sec=t_arr[0] if n > 0 else 0.0
        )

        p_east[0] = initial_p_east_m
        p_north[0] = initial_p_north_m
        headings[0] = wrap_to_2pi(initial_heading_rad)

        # Precompute normalized features if model and scaler exist
        if self.model is not None and self.feature_scaler is not None:
            normed_feats = self.feature_scaler.transform(causal_features_df)
        else:
            normed_feats = None

        v_fl = navigation_inputs_df.get("wheel_speed_fl_ms", pd.Series([None] * n)).to_numpy()
        v_fr = navigation_inputs_df.get("wheel_speed_fr_ms", pd.Series([None] * n)).to_numpy()
        v_rl = navigation_inputs_df.get("wheel_speed_rl_ms", pd.Series([None] * n)).to_numpy()
        v_rr = navigation_inputs_df.get("wheel_speed_rr_ms", pd.Series([None] * n)).to_numpy()
        ax = navigation_inputs_df.get("accel_x_ms2", pd.Series([None] * n)).to_numpy()
        yr = navigation_inputs_df.get("yaw_rate_rads", pd.Series([None] * n)).to_numpy()

        curr_east = initial_p_east_m
        curr_north = initial_p_north_m
        curr_heading = wrap_to_2pi(initial_heading_rad)

        for k in range(n):
            dt_k = dt_arr[k]
            t_k = t_arr[k]

            # 1. Classical Step
            st_class = classical_engine.step(
                time_sec=t_k,
                dt_sec=dt_k,
                wheel_speed_fl=v_fl[k],
                wheel_speed_fr=v_fr[k],
                wheel_speed_rl=v_rl[k],
                wheel_speed_rr=v_rr[k],
                accel_x=ax[k],
                yaw_rate=yr[k]
            )

            raw_v_class = st_class.forward_speed_ms
            raw_w_class = st_class.yaw_rate_rads
            is_stat = st_class.is_stationary

            # 2. AI Residual Prediction (if window available)
            raw_dv = 0.0
            raw_dw = 0.0

            if self.model is not None and normed_feats is not None and k >= self.window_size - 1:
                win_slice = normed_feats[k - self.window_size + 1 : k + 1]  # [W, 16]
                with torch.no_grad():
                    win_tensor = torch.from_numpy(win_slice).unsqueeze(0).to(self.device)  # [1, W, 16]
                    pred_norm, _ = self.model(win_tensor)
                    pred_phys = self.target_scaler.inverse_transform(pred_norm.cpu().numpy())[0]
                    raw_dv = float(pred_phys[0]) if self.enable_v else 0.0
                    raw_dw = float(pred_phys[1]) if self.enable_w else 0.0

            # 3. Safety Guard Sanitization
            guarded = self.safety_guard.sanitize(
                raw_delta_v=raw_dv,
                raw_delta_yaw=raw_dw,
                is_sensor_valid=True,
                is_stationary=is_stat
            )

            # 4. Corrected Kinematics
            v_corrected = max(0.0, raw_v_class + guarded.delta_velocity_ms) if not is_stat else 0.0
            w_corrected = raw_w_class + guarded.delta_yaw_rate_rads if not is_stat else 0.0

            # 5. Midpoint ENU Integration
            delta_psi = w_corrected * dt_k
            psi_mid = curr_heading + 0.5 * delta_psi
            d_east = v_corrected * np.sin(psi_mid) * dt_k
            d_north = v_corrected * np.cos(psi_mid) * dt_k

            curr_east += d_east
            curr_north += d_north
            curr_heading = wrap_to_2pi(curr_heading + delta_psi)

            p_east[k] = curr_east
            p_north[k] = curr_north
            headings[k] = curr_heading
            speeds[k] = v_corrected
            yaw_rates[k] = w_corrected

        step_dists = np.sqrt(np.diff(p_east)**2 + np.diff(p_north)**2)
        total_dist = float(np.sum(step_dists)) if n > 1 else 0.0

        return DeadReckoningTrajectory(
            timestamps_sec=t_arr,
            dt_array_sec=dt_arr,
            p_east_m=p_east,
            p_north_m=p_north,
            heading_rad=headings,
            forward_speed_ms=speeds,
            yaw_rate_rads=yaw_rates,
            baseline_name="AI_CORRECTED_BASELINE",
            total_distance_m=total_dist
        )

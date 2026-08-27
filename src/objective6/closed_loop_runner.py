"""
Closed-Loop Navigation Rollout Engine for Objective 6.
Executes dead reckoning augmented by the SelectiveCorrectionPolicy and logs full telemetry.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch

from navigation_engine.state import DeadReckoningTrajectory, wrap_to_2pi
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from .selective_policy import SelectiveCorrectionPolicy, PolicyDecision


@dataclass
class Objective6RolloutResult:
    trajectory: DeadReckoningTrajectory
    decisions_df: pd.DataFrame
    total_timesteps: int
    applied_timesteps: int
    fallback_timesteps: int
    application_rate_pct: float
    fallback_rate_pct: float
    fallback_reason_counts: Dict[str, int]


class Objective6RolloutRunner:
    """
    Executes dead-reckoning trajectory propagation governed by Objective 6 SelectiveCorrectionPolicy.
    """
    def __init__(
        self,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        policy: SelectiveCorrectionPolicy,
        window_size: int = 10,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.policy = policy
        self.window_size = window_size

    def run_rollout(
        self,
        navigation_inputs_df: pd.DataFrame,
        causal_features_df: pd.DataFrame,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0
    ) -> Objective6RolloutResult:
        """
        Execute full causal rollout over sequence with decision telemetry.
        """
        self.policy.reset()
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

        normed_feats = self.feature_scaler.transform(causal_features_df)
        raw_feats = causal_features_df.to_numpy()

        v_fl = navigation_inputs_df.get("wheel_speed_fl_ms", pd.Series([None] * n)).to_numpy()
        v_fr = navigation_inputs_df.get("wheel_speed_fr_ms", pd.Series([None] * n)).to_numpy()
        v_rl = navigation_inputs_df.get("wheel_speed_rl_ms", pd.Series([None] * n)).to_numpy()
        v_rr = navigation_inputs_df.get("wheel_speed_rr_ms", pd.Series([None] * n)).to_numpy()
        ax = navigation_inputs_df.get("accel_x_ms2", pd.Series([None] * n)).to_numpy()
        yr = navigation_inputs_df.get("yaw_rate_rads", pd.Series([None] * n)).to_numpy()

        curr_east = initial_p_east_m
        curr_north = initial_p_north_m
        curr_heading = wrap_to_2pi(initial_heading_rad)

        decision_records = []
        applied_count = 0
        fallback_count = 0
        reason_counts: Dict[str, int] = {}

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

            if k >= self.window_size - 1:
                win_slice = normed_feats[k - self.window_size + 1 : k + 1]  # [W, 16]
                with torch.no_grad():
                    win_tensor = torch.from_numpy(win_slice).unsqueeze(0).to(self.device)  # [1, W, 16]
                    pred_norm, _ = self.model(win_tensor)
                    pred_phys = self.target_scaler.inverse_transform(pred_norm.cpu().numpy())[0]
                    raw_dv = float(pred_phys[0])
                    raw_dw = float(pred_phys[1])

            # 3. Evaluate Policy Decision
            feat_window_raw = raw_feats[max(0, k - self.window_size + 1) : k + 1]
            decision = self.policy.evaluate(
                raw_delta_v=raw_dv,
                raw_delta_w=raw_dw,
                feature_vector_or_window=feat_window_raw,
                classical_speed_ms=raw_v_class,
                is_stationary=is_stat,
                is_sensor_valid=True
            )

            rec = decision.to_dict()
            rec["time_sec"] = t_k
            rec["classical_speed_ms"] = raw_v_class
            decision_records.append(rec)

            if decision.is_applied:
                applied_count += 1
            else:
                fallback_count += 1
                r = decision.fallback_reason
                reason_counts[r] = reason_counts.get(r, 0) + 1

            # 4. Apply Corrected Kinematics
            delta_v_to_use = decision.applied_delta_v if decision.is_applied else 0.0
            delta_w_to_use = decision.applied_delta_w if decision.is_applied else 0.0

            v_corrected = max(0.0, raw_v_class + delta_v_to_use) if not is_stat else 0.0
            w_corrected = raw_w_class + delta_w_to_use if not is_stat else 0.0

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

        traj = DeadReckoningTrajectory(
            timestamps_sec=t_arr,
            dt_array_sec=dt_arr,
            p_east_m=p_east,
            p_north_m=p_north,
            heading_rad=headings,
            forward_speed_ms=speeds,
            yaw_rate_rads=yaw_rates,
            baseline_name="OBJ6_SELECTIVE_RESIDUAL",
            total_distance_m=total_dist
        )

        dec_df = pd.DataFrame(decision_records)
        app_rate = (applied_count / max(n, 1)) * 100.0
        fb_rate = (fallback_count / max(n, 1)) * 100.0

        return Objective6RolloutResult(
            trajectory=traj,
            decisions_df=dec_df,
            total_timesteps=n,
            applied_timesteps=applied_count,
            fallback_timesteps=fallback_count,
            application_rate_pct=round(app_rate, 2),
            fallback_rate_pct=round(fb_rate, 2),
            fallback_reason_counts=reason_counts
        )

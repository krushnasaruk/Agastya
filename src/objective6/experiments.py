"""
Master Experiment Suite for Objective 6 (Experiments A through J).
Executes all required benchmarking protocols deterministically on the held-out test sequence.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from navigation_engine.evaluation import DeadReckoningEvaluator
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from ai_residual.rollout import AIRolloutEngine
from ai_residual.safety import SafetyGuard
from .distribution_monitor import TrainingDistributionMonitor
from .temporal_consistency import TemporalConsistencyMonitor
from .confidence import PredictiveConfidenceEstimator
from .selective_policy import SelectiveCorrectionPolicy
from .closed_loop_runner import Objective6RolloutRunner, Objective6RolloutResult
from .maneuver_classifier import CausalManeuverClassifier
from .outage_simulator import StandardizedOutageSimulator
from .metrics import Objective6MetricsCalculator


class Objective6ExperimentSuite:
    """
    Executes Experiments A through J for Objective 6.
    """
    @classmethod
    def run_all_experiments(
        cls,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        distribution_monitor: TrainingDistributionMonitor,
        test_nav_df: pd.DataFrame,
        test_causal_feats_df: pd.DataFrame,
        test_ref_df: pd.DataFrame,
        test_sequence_id: str = "sync_02",
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.to(dev)
        model.eval()

        init_h = float(test_ref_df["heading_rad"].iloc[0]) if "heading_rad" in test_ref_df else 0.0
        init_e = float(test_ref_df["pos_east_m"].iloc[0]) if "pos_east_m" in test_ref_df else 0.0
        init_n = float(test_ref_df["pos_north_m"].iloc[0]) if "pos_north_m" in test_ref_df else 0.0

        ref_e = test_ref_df["pos_east_m"].to_numpy()
        ref_n = test_ref_df["pos_north_m"].to_numpy()
        ref_h = test_ref_df.get("heading_rad", None)
        ref_v = test_ref_df.get("ground_speed_ms", None)

        # ----------------------------------------------------------------------
        # EXPERIMENT A: Classical Baseline (No AI)
        # ----------------------------------------------------------------------
        classical_engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
        classical_traj = classical_engine.run_sequence(
            test_nav_df,
            initial_heading_rad=init_h,
            initial_p_east_m=init_e,
            initial_p_north_m=init_n
        )
        exp_a_metrics, exp_a_pos_err, exp_a_head_err = DeadReckoningEvaluator.evaluate(
            classical_traj, ref_e, ref_n, ref_h, ref_v
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT B: Objective 5 Velocity-Only (Unconditional)
        # ----------------------------------------------------------------------
        obj5_v_engine = AIRolloutEngine(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            safety_guard=SafetyGuard(),
            window_size=10,
            enable_velocity_correction=True,
            enable_yaw_correction=False,
            device=dev
        )
        obj5_v_traj = obj5_v_engine.run_rollout(
            test_nav_df,
            test_causal_feats_df,
            initial_p_east_m=init_e,
            initial_p_north_m=init_n,
            initial_heading_rad=init_h
        )
        exp_b_metrics, exp_b_pos_err, exp_b_head_err = DeadReckoningEvaluator.evaluate(
            obj5_v_traj, ref_e, ref_n, ref_h, ref_v
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT C: Objective 6 Selective Velocity (All Gates Active)
        # ----------------------------------------------------------------------
        sel_policy_full = SelectiveCorrectionPolicy(
            distribution_monitor=distribution_monitor,
            temporal_monitor=TemporalConsistencyMonitor(),
            confidence_estimator=PredictiveConfidenceEstimator(),
            enable_velocity_correction=True,
            enable_yaw_correction=False,
            enable_sensor_gate=True,
            enable_stationary_gate=True,
            enable_ood_gate=True,
            enable_temporal_gate=True,
            enable_confidence_gate=True
        )
        obj6_runner = Objective6RolloutRunner(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            policy=sel_policy_full,
            window_size=10,
            device=dev
        )
        obj6_res = obj6_runner.run_rollout(
            test_nav_df,
            test_causal_feats_df,
            initial_p_east_m=init_e,
            initial_p_north_m=init_n,
            initial_heading_rad=init_h
        )
        obj6_traj = obj6_res.trajectory
        exp_c_metrics, exp_c_pos_err, exp_c_head_err = DeadReckoningEvaluator.evaluate(
            obj6_traj, ref_e, ref_n, ref_h, ref_v
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT D: Selective Correction Gate Ablation Study (D1 - D6)
        # ----------------------------------------------------------------------
        gate_configs = [
            ("D1_sensor_only", True, False, False, False, False),
            ("D2_stationary_only", False, True, False, False, False),
            ("D3_ood_only", False, False, True, False, False),
            ("D4_temporal_only", False, False, False, True, False),
            ("D5_confidence_only", False, False, False, False, True),
            ("D6_all_gates", True, True, True, True, True)
        ]

        exp_d_results = {}
        for g_name, en_sens, en_stat, en_ood, en_temp, en_conf in gate_configs:
            p = SelectiveCorrectionPolicy(
                distribution_monitor=distribution_monitor,
                temporal_monitor=TemporalConsistencyMonitor(),
                confidence_estimator=PredictiveConfidenceEstimator(),
                enable_velocity_correction=True,
                enable_yaw_correction=False,
                enable_sensor_gate=en_sens,
                enable_stationary_gate=en_stat,
                enable_ood_gate=en_ood,
                enable_temporal_gate=en_temp,
                enable_confidence_gate=en_conf
            )
            r = Objective6RolloutRunner(model, feature_scaler, target_scaler, p, window_size=10, device=dev).run_rollout(
                test_nav_df, test_causal_feats_df, init_e, init_n, init_h
            )
            m, _, _ = DeadReckoningEvaluator.evaluate(r.trajectory, ref_e, ref_n, ref_h, ref_v)
            exp_d_results[g_name] = {
                "metrics": m.to_dict(),
                "application_rate_pct": r.application_rate_pct,
                "fallback_rate_pct": r.fallback_rate_pct
            }

        # ----------------------------------------------------------------------
        # EXPERIMENT E: Yaw-Only Correction
        # ----------------------------------------------------------------------
        yaw_engine = AIRolloutEngine(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            safety_guard=SafetyGuard(),
            window_size=10,
            enable_velocity_correction=False,
            enable_yaw_correction=True,
            device=dev
        )
        yaw_traj = yaw_engine.run_rollout(test_nav_df, test_causal_feats_df, init_e, init_n, init_h)
        exp_e_metrics, exp_e_pos_err, exp_e_head_err = DeadReckoningEvaluator.evaluate(
            yaw_traj, ref_e, ref_n, ref_h, ref_v
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT F: Full Velocity + Yaw Correction
        # ----------------------------------------------------------------------
        full_engine = AIRolloutEngine(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            safety_guard=SafetyGuard(),
            window_size=10,
            enable_velocity_correction=True,
            enable_yaw_correction=True,
            device=dev
        )
        full_traj = full_engine.run_rollout(test_nav_df, test_causal_feats_df, init_e, init_n, init_h)
        exp_f_metrics, exp_f_pos_err, exp_f_head_err = DeadReckoningEvaluator.evaluate(
            full_traj, ref_e, ref_n, ref_h, ref_v
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT G: Extended GNSS Outages (5s, 10s, 15s, 20s, 30s, 45s)
        # ----------------------------------------------------------------------
        outage_records = StandardizedOutageSimulator.evaluate_multi_duration_outages(
            classical_traj=classical_traj,
            obj5_traj=obj5_v_traj,
            obj6_traj=obj6_traj,
            ref_east_m=ref_e,
            ref_north_m=ref_n,
            entry_time_sec=20.0,
            durations=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
        )

        # ----------------------------------------------------------------------
        # EXPERIMENT H: Maneuver-Stratified Breakdown
        # ----------------------------------------------------------------------
        v_can = test_nav_df["wheel_speed_rear_mean_ms"].to_numpy() if "wheel_speed_rear_mean_ms" in test_nav_df else (test_nav_df["wheel_speed_rl_ms"] + test_nav_df["wheel_speed_rr_ms"]) * 0.5
        ax_can = test_nav_df["accel_x_ms2"].to_numpy()
        yr_can = test_nav_df["yaw_rate_rads"].to_numpy()
        maneuver_labels = CausalManeuverClassifier.classify_sequence(v_can, ax_can, yr_can)

        maneuver_classical = Objective6MetricsCalculator.compute_maneuver_stratified_metrics(maneuver_labels, exp_a_pos_err, exp_a_head_err)
        maneuver_obj5 = Objective6MetricsCalculator.compute_maneuver_stratified_metrics(maneuver_labels, exp_b_pos_err, exp_b_head_err)
        maneuver_obj6 = Objective6MetricsCalculator.compute_maneuver_stratified_metrics(maneuver_labels, exp_c_pos_err, exp_c_head_err)

        # ----------------------------------------------------------------------
        # EXPERIMENT I: AI Application Rate & Fallback Telemetry
        # ----------------------------------------------------------------------
        app_metrics = {
            "total_timesteps": obj6_res.total_timesteps,
            "applied_timesteps": obj6_res.applied_timesteps,
            "fallback_timesteps": obj6_res.fallback_timesteps,
            "application_rate_pct": obj6_res.application_rate_pct,
            "fallback_rate_pct": obj6_res.fallback_rate_pct,
            "fallback_reason_breakdown": obj6_res.fallback_reason_counts
        }

        # ----------------------------------------------------------------------
        # EXPERIMENT J: Residual Quality on Accepted Timesteps
        # ----------------------------------------------------------------------
        dec_df = obj6_res.decisions_df
        raw_dv = dec_df["raw_delta_v"].to_numpy()
        app_mask = dec_df["is_applied"].to_numpy()

        # Calibration reliability analysis
        conf_scores = dec_df["confidence_score"].to_numpy()
        # Compute absolute true error on raw predictions if reference is available
        ref_v_arr = ref_v.to_numpy() if ref_v is not None else np.zeros_like(raw_dv)
        true_dv = ref_v_arr - dec_df["classical_speed_ms"].to_numpy()
        res_errors = np.abs(raw_dv - true_dv)

        calibration_analysis = PredictiveConfidenceEstimator.evaluate_calibration(conf_scores, res_errors)

        return {
            "sequence_id": test_sequence_id,
            "classical_traj": classical_traj,
            "obj5_v_traj": obj5_v_traj,
            "obj6_traj": obj6_traj,
            "yaw_traj": yaw_traj,
            "full_traj": full_traj,
            "decisions_df": dec_df,
            "maneuver_labels": maneuver_labels,
            "experiment_a_classical": exp_a_metrics.to_dict(),
            "experiment_b_obj5_velocity": exp_b_metrics.to_dict(),
            "experiment_c_obj6_selective": exp_c_metrics.to_dict(),
            "experiment_d_ablations": exp_d_results,
            "experiment_e_yaw_only": exp_e_metrics.to_dict(),
            "experiment_f_full": exp_f_metrics.to_dict(),
            "experiment_g_outages": outage_records,
            "experiment_h_maneuvers": {
                "classical": maneuver_classical,
                "objective5_velocity": maneuver_obj5,
                "objective6_selective": maneuver_obj6
            },
            "experiment_i_ai_usage": app_metrics,
            "experiment_j_calibration": calibration_analysis,
            "pos_errors": {
                "classical": exp_a_pos_err,
                "obj5_velocity": exp_b_pos_err,
                "obj6_selective": exp_c_pos_err,
                "yaw_only": exp_e_pos_err,
                "full": exp_f_pos_err
            },
            "heading_errors": {
                "classical": exp_a_head_err,
                "obj5_velocity": exp_b_head_err,
                "obj6_selective": exp_c_head_err,
                "yaw_only": exp_e_head_err,
                "full": exp_f_head_err
            }
        }

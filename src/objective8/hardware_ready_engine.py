"""
Hardware-Ready Navigation Engine for Objective 8.
Integrates classical physics, quantized INT8 / FP32 inference, watchdog supervision,
Objective 6 multi-gate safety policy, and resource profiling into a unified edge runtime.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from navigation_engine.state import PlanarNavigationState, DeadReckoningTrajectory, wrap_to_2pi
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from services.ml.src.features.causal_features import CausalFeatureExtractor
from objective6.selective_policy import SelectiveCorrectionPolicy, PolicyDecision
from objective6.distribution_monitor import TrainingDistributionMonitor
from objective6.temporal_consistency import TemporalConsistencyMonitor
from objective6.confidence import PredictiveConfidenceEstimator

from objective7.sensor_validator import SensorValidator, SensorValidationResult
from objective7.latency_monitor import LatencyMonitor, EpochLatencyBreakdown
from objective7.watchdog import AIWatchdog
from objective7.telemetry import TelemetryLogger, TelemetryFrame

from .quantization import ModelQuantizer
from .quantized_model import QuantizedInferenceWrapper
from .resource_monitor import ResourceMonitor
from .numerical_stability import NumericalStabilityMonitor


@dataclass
class HardwareSensorPacket:
    timestamp_sec: float
    dt_sec: float
    wheel_speed_fl_ms: Optional[float] = None
    wheel_speed_fr_ms: Optional[float] = None
    wheel_speed_rl_ms: Optional[float] = None
    wheel_speed_rr_ms: Optional[float] = None
    accel_x_ms2: Optional[float] = None
    yaw_rate_rads: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareStepResult:
    timestamp: float
    position: Tuple[float, float]
    velocity: float
    heading: float
    classical_state: PlanarNavigationState
    corrected_state: PlanarNavigationState
    delta_v: float
    delta_omega_z: float
    ai_applied: bool
    fallback_active: bool
    fallback_reason: str
    confidence: float
    ood_score: float
    inference_latency_ms: float
    total_latency_ms: float
    numerical_status: str
    deployment_mode: str
    dt: float = 0.1
    watchdog_status: str = "HEALTHY"

    @property
    def ai_residual(self) -> Tuple[float, float]:
        return (self.delta_v, self.delta_omega_z)

    @property
    def fallback(self) -> bool:
        return self.fallback_active

    @property
    def latency_ms(self) -> float:
        return self.total_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareReadyNavigationEngine:
    """
    Automotive-grade edge navigation engine supporting FP32, INT8 Quantized,
    Classical-Only, and Auto-Safe deployment modes.
    """

    DEPLOYMENT_MODES = {"MODE_A_FP32", "MODE_B_INT8", "MODE_C_CLASSICAL", "MODE_D_AUTO"}

    def __init__(
        self,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        deployment_mode: str = "MODE_B_INT8",
        distribution_monitor: Optional[TrainingDistributionMonitor] = None,
        selective_policy: Optional[SelectiveCorrectionPolicy] = None,
        watchdog_budget_ms: float = 25.0,
        memory_limit_mb: float = 25.0,
        window_size: int = 10
    ):
        self.deployment_mode = deployment_mode.upper()
        if self.deployment_mode not in self.DEPLOYMENT_MODES:
            self.deployment_mode = "MODE_B_INT8"

        self.window_size = window_size
        self.sensor_validator = SensorValidator()
        self.latency_monitor = LatencyMonitor(deadline_ms=100.0, preferred_target_ms=50.0)
        self.watchdog = AIWatchdog(execution_budget_ms=watchdog_budget_ms)
        self.telemetry_logger = TelemetryLogger()
        self.resource_monitor = ResourceMonitor(memory_limit_mb=memory_limit_mb, latency_budget_ms=watchdog_budget_ms)
        self.stability_monitor = NumericalStabilityMonitor()

        # Build models
        self.fp32_model = model
        self.fp32_model.eval()

        if self.deployment_mode in {"MODE_B_INT8", "MODE_D_AUTO"}:
            self.int8_model = ModelQuantizer.quantize_dynamic_int8(self.fp32_model)
            self.inference_runner = QuantizedInferenceWrapper(self.int8_model, feature_scaler, target_scaler, precision_mode="INT8")
        else:
            self.int8_model = None
            self.inference_runner = QuantizedInferenceWrapper(self.fp32_model, feature_scaler, target_scaler, precision_mode="FP32")

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        self.dist_monitor = distribution_monitor or TrainingDistributionMonitor()
        self.policy = selective_policy or SelectiveCorrectionPolicy(
            distribution_monitor=self.dist_monitor,
            temporal_monitor=TemporalConsistencyMonitor(),
            confidence_estimator=PredictiveConfidenceEstimator(),
            enable_velocity_correction=True,
            enable_yaw_correction=False
        )

        self.classical_engine = ClassicalDeadReckoningEngine(baseline_type="BASELINE_A")
        self.current_state: Optional[PlanarNavigationState] = None
        self.window_buffer: List[np.ndarray] = []
        self.nav_history_records: List[Dict[str, Any]] = []
        self.hidden_state: Optional[torch.Tensor] = None
        self.is_initialized = False

    def initialize(
        self,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0,
        initial_speed_ms: float = 0.0
    ) -> None:
        """Initializes navigation state and clears runtime history."""
        self.classical_engine.initialize(
            initial_p_east_m=initial_p_east_m,
            initial_p_north_m=initial_p_north_m,
            initial_heading_rad=initial_heading_rad,
            initial_time_sec=0.0
        )
        self.current_state = PlanarNavigationState(
            time_sec=0.0,
            p_east_m=initial_p_east_m,
            p_north_m=initial_p_north_m,
            heading_rad=initial_heading_rad,
            forward_speed_ms=initial_speed_ms,
            yaw_rate_rads=0.0,
            accel_longitudinal_ms2=0.0,
            is_stationary=bool(initial_speed_ms < 0.08),
            quality_status="INITIALIZED"
        )
        self.window_buffer.clear()
        self.nav_history_records.clear()
        self.telemetry_logger.reset()
        self.sensor_validator.reset()
        self.policy.reset()
        self.hidden_state = None
        self.is_initialized = True

    def step(self, packet: Union[HardwareSensorPacket, Dict[str, Any]], artificial_ai_delay_ms: float = 0.0) -> HardwareStepResult:
        """
        Step API processing one sensor packet.
        """
        epoch_start = time.perf_counter()

        if isinstance(packet, dict):
            t_raw = packet.get("timestamp_sec", packet.get("time_sec", 0.0))
            dt_raw = packet.get("dt_sec", 0.1)
            w_fl = packet.get("wheel_speed_fl_ms", packet.get("wheel_fl", None))
            w_fr = packet.get("wheel_speed_fr_ms", packet.get("wheel_fr", None))
            w_rl = packet.get("wheel_speed_rl_ms", packet.get("wheel_rl", None))
            w_rr = packet.get("wheel_speed_rr_ms", packet.get("wheel_rr", None))
            ax = packet.get("accel_x_ms2", packet.get("accel_x", None))
            yr = packet.get("yaw_rate_rads", packet.get("yaw_rate", None))
        else:
            t_raw = packet.timestamp_sec
            dt_raw = packet.dt_sec
            w_fl = packet.wheel_speed_fl_ms
            w_fr = packet.wheel_speed_fr_ms
            w_rl = packet.wheel_speed_rl_ms
            w_rr = packet.wheel_speed_rr_ms
            ax = packet.accel_x_ms2
            yr = packet.yaw_rate_rads

        # 1. Validation stage
        t0 = time.perf_counter()
        v_res: SensorValidationResult = self.sensor_validator.validate_sample(
            timestamp_sec=t_raw,
            dt_sec=dt_raw,
            wheel_fl=w_fl,
            wheel_fr=w_fr,
            wheel_rl=w_rl,
            wheel_rr=w_rr,
            accel_x=ax,
            yaw_rate=yr
        )
        val_lat = (time.perf_counter() - t0) * 1000.0

        t_k = t_raw
        dt_k = v_res.cleaned_dt
        clean_fl = v_res.cleaned_wheel_fl
        clean_fr = v_res.cleaned_wheel_fr
        clean_rl = v_res.cleaned_wheel_rl
        clean_rr = v_res.cleaned_wheel_rr
        clean_ax = v_res.cleaned_accel_x
        clean_yr = v_res.cleaned_yaw_rate

        # 2. Classical physics update
        t0 = time.perf_counter()
        class_st = self.classical_engine.step(
            time_sec=t_k,
            dt_sec=dt_k,
            wheel_speed_fl=clean_fl,
            wheel_speed_fr=clean_fr,
            wheel_speed_rl=clean_rl,
            wheel_speed_rr=clean_rr,
            accel_x=clean_ax,
            yaw_rate=clean_yr
        )
        class_lat = (time.perf_counter() - t0) * 1000.0

        # 3. Causal Feature Extraction
        t0 = time.perf_counter()
        v_fl_f = clean_fl if clean_fl is not None else class_st.forward_speed_ms
        v_fr_f = clean_fr if clean_fr is not None else class_st.forward_speed_ms
        v_rl_f = clean_rl if clean_rl is not None else class_st.forward_speed_ms
        v_rr_f = clean_rr if clean_rr is not None else class_st.forward_speed_ms

        v_rear_mean = 0.5 * (v_rl_f + v_rr_f)
        v_rear_diff = v_rr_f - v_rl_f
        v_front_mean = 0.5 * (v_fl_f + v_fr_f)
        v_axle_diff = v_front_mean - v_rear_mean

        prev_ax = self.current_state.accel_longitudinal_ms2 if self.current_state else 0.0
        prev_yr = self.current_state.yaw_rate_rads if self.current_state else 0.0
        safe_dt = max(dt_k, 0.005)
        jerk = (clean_ax - prev_ax) / safe_dt
        yaw_acc = (clean_yr - prev_yr) / safe_dt
        v_class = class_st.forward_speed_ms
        curv = clean_yr / max(v_class, 0.1)
        is_stat_flag = float(class_st.is_stationary)
        slip_flag = float((abs(v_rear_diff) > 2.5) and (v_class > 2.0))

        feat_vector = np.array([
            v_fl_f, v_fr_f, v_rl_f, v_rr_f,
            v_rear_mean, v_rear_diff, v_axle_diff,
            clean_ax, jerk, clean_yr, yaw_acc,
            dt_k, v_class, curv, is_stat_flag, slip_flag
        ], dtype=np.float64)
        feat_lat = (time.perf_counter() - t0) * 1000.0

        # 4. Window Buffer Update
        t0 = time.perf_counter()
        self.window_buffer.append(feat_vector)
        if len(self.window_buffer) > self.window_size:
            self.window_buffer.pop(0)
        window_lat = (time.perf_counter() - t0) * 1000.0

        # 5. Inference & Safety Gate Execution
        pred_dv = 0.0
        pred_dw = 0.0
        ai_applied = False
        fallback = True
        fallback_reason = "NONE"
        conf_score = 0.0
        ood_dist = 0.0
        inf_lat = 0.0
        pol_lat = 0.0
        watchdog_timeout = False

        if self.deployment_mode == "MODE_C_CLASSICAL":
            fallback = True
            fallback_reason = "MODE_CLASSICAL_ONLY"
        elif not v_res.is_valid:
            fallback = True
            fallback_reason = v_res.status_code or "SENSOR_INVALID"
        elif len(self.window_buffer) < self.window_size:
            fallback = True
            fallback_reason = "WARMUP_INCOMPLETE"
        else:
            # Run supervised inference
            window_10x16 = np.array(self.window_buffer)  # Shape [10, 16]
            self.watchdog.start_cycle()

            try:
                if artificial_ai_delay_ms > 0:
                    time.sleep(artificial_ai_delay_ms / 1000.0)

                pred_dv_raw, pred_dw_raw, inf_lat, self.hidden_state = self.inference_runner.predict_step(
                    window_10x16, self.hidden_state
                )

                if self.watchdog.is_timeout():
                    watchdog_timeout = True
                    fallback = True
                    fallback_reason = "AI_TIMEOUT"
                else:
                    pred_dv = pred_dv_raw
                    pred_dw = pred_dw_raw

                    # Objective 6 Multi-Gate Safety Policy
                    t0 = time.perf_counter()
                    decision: PolicyDecision = self.policy.evaluate(
                        raw_delta_v=pred_dv,
                        raw_delta_w=pred_dw,
                        feature_vector_or_window=feat_vector,
                        classical_speed_ms=class_st.forward_speed_ms,
                        is_stationary=class_st.is_stationary,
                        is_sensor_valid=bool(clean_fl is not None and clean_ax is not None)
                    )
                    pol_lat = (time.perf_counter() - t0) * 1000.0

                    ai_applied = decision.is_applied
                    fallback = not decision.is_applied
                    fallback_reason = decision.fallback_reason
                    conf_score = decision.confidence_score
                    ood_dist = decision.ood_score

                    if not ai_applied:
                        pred_dv = 0.0
                        pred_dw = 0.0

            except Exception as ex:
                fallback = True
                fallback_reason = "AI_EXCEPTION"

        # 6. Kinematic State Integration
        enable_yaw = getattr(self.policy, 'enable_yaw_correction', getattr(self.policy, 'enable_w', False))
        v_corr = class_st.forward_speed_ms + (pred_dv if ai_applied else 0.0)
        w_corr = class_st.yaw_rate_rads + (pred_dw if (ai_applied and enable_yaw) else 0.0)

        prev_east = self.current_state.p_east_m if self.current_state else 0.0
        prev_north = self.current_state.p_north_m if self.current_state else 0.0
        prev_head = self.current_state.heading_rad if self.current_state else 0.0

        new_heading = wrap_to_2pi(prev_head + w_corr * dt_k)
        mid_heading = prev_head + 0.5 * w_corr * dt_k
        new_east = prev_east + v_corr * np.sin(mid_heading) * dt_k
        new_north = prev_north + v_corr * np.cos(mid_heading) * dt_k

        # 7. Numerical stability check
        is_stable = self.stability_monitor.check_state(new_east, new_north, new_heading, v_corr)
        if not is_stable:
            # Preserve last known state if unstable
            new_east, new_north = prev_east, prev_north
            new_heading = prev_head
            v_corr = class_st.forward_speed_ms
            fallback = True
            fallback_reason = "NUMERICAL_INSTABILITY"

        self.current_state = PlanarNavigationState(
            time_sec=t_k,
            p_east_m=new_east,
            p_north_m=new_north,
            heading_rad=new_heading,
            forward_speed_ms=v_corr,
            yaw_rate_rads=w_corr,
            accel_longitudinal_ms2=clean_ax,
            is_stationary=bool(v_corr < 0.08),
            quality_status="OK" if is_stable else "FALLBACK"
        )

        tot_lat = (time.perf_counter() - epoch_start) * 1000.0
        telem_lat = 0.01

        # Record metrics
        self.latency_monitor.record_epoch(
            EpochLatencyBreakdown(val_lat, class_lat, feat_lat, window_lat, inf_lat, pol_lat, telem_lat, tot_lat)
        )
        self.resource_monitor.end_epoch(epoch_start, ai_applied, fallback)

        telem_frame = TelemetryFrame(
            timestamp=t_k,
            dt=dt_k,
            classical_velocity=class_st.forward_speed_ms,
            corrected_velocity=v_corr,
            predicted_delta_velocity=pred_dv,
            predicted_delta_yaw=pred_dw,
            ai_applied=ai_applied,
            fallback=fallback,
            fallback_reason=fallback_reason,
            confidence=conf_score,
            ood_score=ood_dist,
            inference_latency_ms=inf_lat,
            total_latency_ms=tot_lat,
            watchdog_timeout=watchdog_timeout,
            sensor_valid=v_res.is_valid,
            stationary=bool(v_corr < 0.08),
            navigation_state_valid=is_stable
        )
        self.telemetry_logger.log_frame(telem_frame)

        self.nav_history_records.append({
            "time_sec": t_k,
            "dt_sec": dt_k,
            "pos_east_m": new_east,
            "pos_north_m": new_north,
            "heading_rad": new_heading,
            "forward_speed_ms": v_corr,
            "yaw_rate_rads": w_corr
        })

        return HardwareStepResult(
            timestamp=t_k,
            position=(new_east, new_north),
            velocity=v_corr,
            heading=new_heading,
            classical_state=class_st,
            corrected_state=self.current_state,
            delta_v=pred_dv,
            delta_omega_z=pred_dw,
            ai_applied=ai_applied,
            fallback_active=fallback,
            fallback_reason=fallback_reason,
            confidence=conf_score,
            ood_score=ood_dist,
            inference_latency_ms=inf_lat,
            total_latency_ms=tot_lat,
            numerical_status="STABLE" if is_stable else "UNSTABLE",
            deployment_mode=self.deployment_mode,
            dt=dt_k,
            watchdog_status="TIMEOUT" if watchdog_timeout else "HEALTHY"
        )

    def get_trajectory(self) -> DeadReckoningTrajectory:
        df = pd.DataFrame(self.nav_history_records)
        if df.empty:
            return DeadReckoningTrajectory(
                np.array([0.0]), np.array([0.1]), np.array([0.0]), np.array([0.0]),
                np.array([0.0]), np.array([0.0]), np.array([0.0]), f"OBJ8_{self.deployment_mode}", 0.0
            )
        e = df["pos_east_m"].to_numpy()
        n = df["pos_north_m"].to_numpy()
        dists = np.sqrt(np.diff(e)**2 + np.diff(n)**2)
        total_dist = float(np.sum(dists)) if len(e) > 1 else 0.0

        return DeadReckoningTrajectory(
            timestamps_sec=df["time_sec"].to_numpy(),
            dt_array_sec=df["dt_sec"].to_numpy(),
            p_east_m=e,
            p_north_m=n,
            heading_rad=df["heading_rad"].to_numpy(),
            forward_speed_ms=df["forward_speed_ms"].to_numpy(),
            yaw_rate_rads=df["yaw_rate_rads"].to_numpy(),
            baseline_name=f"OBJ8_{self.deployment_mode}",
            total_distance_m=total_dist
        )

    def get_telemetry(self) -> pd.DataFrame:
        return self.telemetry_logger.to_dataframe()

    def shutdown(self) -> None:
        self.window_buffer.clear()
        self.hidden_state = None
        self.is_initialized = False

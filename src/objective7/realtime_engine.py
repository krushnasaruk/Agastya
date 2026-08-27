"""
Real-Time Navigation Engine for Objective 7.
Integrates classical physics, feature windowing, frozen AI inference, watchdog budget supervision,
Objective 6 selective correction policy, and microsecond telemetry logging.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from navigation_engine.state import PlanarNavigationState, DeadReckoningTrajectory, wrap_to_2pi
from navigation_engine.dead_reckoning import ClassicalDeadReckoningEngine
from ai_residual.model import CausalResidualGRU
from ai_residual.scaler import TrainOnlyScaler, TargetScaler
from ai_residual.feature_registry import CANONICAL_FEATURE_NAMES
from services.ml.src.features.causal_features import CausalFeatureExtractor
from objective6.selective_policy import SelectiveCorrectionPolicy, PolicyDecision
from objective6.distribution_monitor import TrainingDistributionMonitor
from objective6.temporal_consistency import TemporalConsistencyMonitor
from objective6.confidence import PredictiveConfidenceEstimator

from .sensor_validator import SensorValidator, SensorValidationResult
from .latency_monitor import LatencyMonitor, EpochLatencyBreakdown
from .watchdog import AIWatchdog
from .telemetry import TelemetryLogger, TelemetryFrame
from .inference_runner import InferenceRunner
from typing import Union


@dataclass
class SensorPacket:
    timestamp_sec: float
    dt_sec: float
    wheel_speed_fl_ms: Optional[float] = None
    wheel_speed_fr_ms: Optional[float] = None
    wheel_speed_rl_ms: Optional[float] = None
    wheel_speed_rr_ms: Optional[float] = None
    accel_x_ms2: Optional[float] = None
    yaw_rate_rads: Optional[float] = None


@dataclass
class NavigationStepResult:
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


class RealtimeNavigationEngine:
    """
    Production-ready real-time navigation pipeline.
    """
    def __init__(
        self,
        model: CausalResidualGRU,
        feature_scaler: TrainOnlyScaler,
        target_scaler: TargetScaler,
        distribution_monitor: Optional[TrainingDistributionMonitor] = None,
        selective_policy: Optional[SelectiveCorrectionPolicy] = None,
        execution_budget_ms: float = 25.0,
        window_size: int = 10
    ):
        self.window_size = window_size
        self.sensor_validator = SensorValidator()
        self.latency_monitor = LatencyMonitor(deadline_ms=100.0, preferred_target_ms=50.0)
        self.watchdog = AIWatchdog(execution_budget_ms=execution_budget_ms)
        self.telemetry_logger = TelemetryLogger()

        self.inference_runner = InferenceRunner(model, feature_scaler, target_scaler)
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
        self.window_buffer: List[np.ndarray] = []  # Rolling W=10 buffer
        self.nav_history_records: List[Dict[str, Any]] = []
        self.is_initialized = False

    def initialize(
        self,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0,
        initial_time_sec: float = 0.0
    ) -> None:
        """Initialize engine state at sequence start."""
        self.classical_engine.initialize(
            initial_p_east_m=initial_p_east_m,
            initial_p_north_m=initial_p_north_m,
            initial_heading_rad=initial_heading_rad,
            initial_time_sec=initial_time_sec
        )
        self.current_state = PlanarNavigationState(
            time_sec=initial_time_sec,
            p_east_m=initial_p_east_m,
            p_north_m=initial_p_north_m,
            heading_rad=wrap_to_2pi(initial_heading_rad),
            forward_speed_ms=0.0,
            yaw_rate_rads=0.0,
            accel_longitudinal_ms2=0.0,
            is_stationary=True,
            quality_status="INITIALIZED"
        )
        self.sensor_validator.reset()
        self.latency_monitor.reset()
        self.watchdog.reset()
        self.telemetry_logger.reset()
        self.policy.reset()
        self.prev_ax = 0.0
        self.prev_yr = 0.0
        self.window_buffer.clear()
        self.nav_history_records.clear()
        self.is_initialized = True

    def reset(self) -> None:
        """Reset internal engine state."""
        self.initialize()

    def process_sensor_sample(
        self,
        timestamp_sec: float,
        dt_sec: float,
        wheel_fl: Optional[float],
        wheel_fr: Optional[float],
        wheel_rl: Optional[float],
        wheel_rr: Optional[float],
        accel_x: Optional[float],
        yaw_rate: Optional[float],
        artificial_ai_delay_ms: float = 0.0,
        inject_ai_exception: bool = False
    ) -> PlanarNavigationState:
        """
        Execute 11-step real-time processing cycle for single sensor frame.
        """
        if not self.is_initialized:
            self.initialize(initial_time_sec=timestamp_sec if timestamp_sec is not None else 0.0)

        t_epoch_start = time.perf_counter()

        # ----------------------------------------------------------------------
        # 1 & 2 & 3. Sensor Validation
        # ----------------------------------------------------------------------
        t_v0 = time.perf_counter()
        val_res: SensorValidationResult = self.sensor_validator.validate_sample(
            timestamp_sec=timestamp_sec,
            dt_sec=dt_sec,
            wheel_fl=wheel_fl,
            wheel_fr=wheel_fr,
            wheel_rl=wheel_rl,
            wheel_rr=wheel_rr,
            accel_x=accel_x,
            yaw_rate=yaw_rate
        )
        t_v1 = time.perf_counter()
        lat_sensor_val = (t_v1 - t_v0) * 1000.0

        # ----------------------------------------------------------------------
        # 4. Classical Physics Engine Update
        # ----------------------------------------------------------------------
        t_p0 = time.perf_counter()
        class_state = self.classical_engine.step(
            time_sec=timestamp_sec if val_res.is_valid else (self.current_state.time_sec + val_res.cleaned_dt),
            dt_sec=val_res.cleaned_dt,
            wheel_speed_fl=val_res.cleaned_wheel_fl,
            wheel_speed_fr=val_res.cleaned_wheel_fr,
            wheel_speed_rl=val_res.cleaned_wheel_rl,
            wheel_speed_rr=val_res.cleaned_wheel_rr,
            accel_x=val_res.cleaned_accel_x,
            yaw_rate=val_res.cleaned_yaw_rate
        )
        t_p1 = time.perf_counter()
        lat_classical = (t_p1 - t_p0) * 1000.0

        # ----------------------------------------------------------------------
        # 5 & 6. Causal Feature Construction & Window Update
        # ----------------------------------------------------------------------
        t_f0 = time.perf_counter()
        # Compute instantaneous 16 causal features exactly matching CausalFeatureExtractor
        v_fl = val_res.cleaned_wheel_fl
        v_fr = val_res.cleaned_wheel_fr
        v_rl = val_res.cleaned_wheel_rl
        v_rr = val_res.cleaned_wheel_rr
        v_rear_mean = (v_rl + v_rr) * 0.5
        v_rear_diff = (v_rr - v_rl)
        v_front_mean = (v_fl + v_fr) * 0.5
        v_axle_diff = v_front_mean - v_rear_mean
        ax = val_res.cleaned_accel_x
        yr = val_res.cleaned_yaw_rate
        dt_k = val_res.cleaned_dt
        v_class = class_state.forward_speed_ms
        is_stat = class_state.is_stationary

        dt_safe = max(dt_k, 0.005)
        jerk = (ax - self.prev_ax) / dt_safe if self.window_buffer else 0.0
        yaw_acc = (yr - self.prev_yr) / dt_safe if self.window_buffer else 0.0
        self.prev_ax = ax
        self.prev_yr = yr

        curv = yr / max(v_class, 0.1)
        is_stat_flag = float(v_class < 0.08)
        slip_flag = float((abs(v_rear_diff) > 2.5) and (v_class > 2.0))

        feat_16 = np.array([
            v_fl, v_fr, v_rl, v_rr, v_rear_mean, v_rear_diff, v_axle_diff,
            ax, jerk, yr, yaw_acc, dt_k, v_class, curv,
            is_stat_flag, slip_flag
        ], dtype=np.float64)

        if len(self.window_buffer) >= self.window_size:
            self.window_buffer.pop(0)
        self.window_buffer.append(feat_16)
        t_f1 = time.perf_counter()
        lat_features = (t_f1 - t_f0) * 1000.0
        lat_window = 0.001

        # ----------------------------------------------------------------------
        # 7 & 8. Model Inference & Watchdog Supervision
        # ----------------------------------------------------------------------
        raw_dv = 0.0
        raw_dw = 0.0
        lat_infer = 0.0
        watchdog_timeout = False
        ai_exception = False

        self.watchdog.start_cycle()

        if len(self.window_buffer) == self.window_size and val_res.is_valid:
            try:
                win_arr = np.array(self.window_buffer, dtype=np.float64)
                raw_dv, raw_dw, lat_infer = self.inference_runner.predict_residual(
                    win_arr,
                    artificial_delay_ms=artificial_ai_delay_ms,
                    inject_model_exception=inject_ai_exception
                )
            except Exception:
                ai_exception = True
                raw_dv, raw_dw = 0.0, 0.0

            wd_status = self.watchdog.check_deadline()
            if wd_status.is_timed_out:
                watchdog_timeout = True
        else:
            raw_dv, raw_dw = 0.0, 0.0

        # ----------------------------------------------------------------------
        # 9. Selective Policy Evaluation
        # ----------------------------------------------------------------------
        t_pol0 = time.perf_counter()
        if watchdog_timeout:
            decision = PolicyDecision(
                is_applied=False, is_fallback=True, fallback_reason="AI_TIMEOUT",
                is_clamped=False, applied_delta_v=0.0, applied_delta_w=0.0,
                raw_delta_v=raw_dv, raw_delta_w=raw_dw, ood_score=0.0, is_ood=False,
                velocity_jump_ms=0.0, is_temporal_consistent=False, confidence_score=0.0, is_confident=False
            )
        elif ai_exception:
            decision = PolicyDecision(
                is_applied=False, is_fallback=True, fallback_reason="AI_EXCEPTION",
                is_clamped=False, applied_delta_v=0.0, applied_delta_w=0.0,
                raw_delta_v=0.0, raw_delta_w=0.0, ood_score=0.0, is_ood=False,
                velocity_jump_ms=0.0, is_temporal_consistent=False, confidence_score=0.0, is_confident=False
            )
        elif len(self.window_buffer) < self.window_size:
            decision = PolicyDecision(
                is_applied=False, is_fallback=True, fallback_reason="WINDOW_NOT_READY",
                is_clamped=False, applied_delta_v=0.0, applied_delta_w=0.0,
                raw_delta_v=0.0, raw_delta_w=0.0, ood_score=0.0, is_ood=False,
                velocity_jump_ms=0.0, is_temporal_consistent=True, confidence_score=0.0, is_confident=False
            )
        else:
            decision = self.policy.evaluate(
                raw_delta_v=raw_dv,
                raw_delta_w=raw_dw,
                feature_vector_or_window=np.array(self.window_buffer),
                classical_speed_ms=v_class,
                is_stationary=is_stat,
                is_sensor_valid=val_res.is_valid
            )
        t_pol1 = time.perf_counter()
        lat_policy = (t_pol1 - t_pol0) * 1000.0

        # ----------------------------------------------------------------------
        # 10. Navigation State Integration
        # ----------------------------------------------------------------------
        delta_v_used = decision.applied_delta_v if decision.is_applied else 0.0
        delta_w_used = decision.applied_delta_w if decision.is_applied else 0.0

        v_corr = max(0.0, v_class + delta_v_used) if not is_stat else 0.0
        w_corr = (class_state.yaw_rate_rads + delta_w_used) if not is_stat else 0.0

        # Midpoint integration on local ENU plane
        delta_psi = w_corr * dt_k
        psi_mid = self.current_state.heading_rad + 0.5 * delta_psi
        d_east = v_corr * np.sin(psi_mid) * dt_k
        d_north = v_corr * np.cos(psi_mid) * dt_k

        new_east = self.current_state.p_east_m + d_east
        new_north = self.current_state.p_north_m + d_north
        new_heading = wrap_to_2pi(self.current_state.heading_rad + delta_psi)

        self.current_state = PlanarNavigationState(
            time_sec=timestamp_sec if timestamp_sec is not None else (self.current_state.time_sec + dt_k),
            p_east_m=new_east,
            p_north_m=new_north,
            heading_rad=new_heading,
            forward_speed_ms=v_corr,
            yaw_rate_rads=w_corr,
            accel_longitudinal_ms2=val_res.cleaned_accel_x,
            is_stationary=is_stat,
            quality_status="VALID" if val_res.is_valid else "DEGRADED"
        )

        # ----------------------------------------------------------------------
        # 11. Telemetry Emission
        # ----------------------------------------------------------------------
        t_tel0 = time.perf_counter()
        t_epoch_end = time.perf_counter()
        total_lat = (t_epoch_end - t_epoch_start) * 1000.0
        lat_telem = (time.perf_counter() - t_tel0) * 1000.0

        breakdown = EpochLatencyBreakdown(
            sensor_validation_ms=round(lat_sensor_val, 4),
            classical_physics_ms=round(lat_classical, 4),
            feature_extraction_ms=round(lat_features, 4),
            window_update_ms=round(lat_window, 4),
            neural_inference_ms=round(lat_infer, 4),
            policy_evaluation_ms=round(lat_policy, 4),
            telemetry_ms=round(lat_telem, 4),
            total_latency_ms=round(total_lat, 4)
        )
        self.latency_monitor.record_epoch(breakdown)

        telem_frame = TelemetryFrame(
            timestamp=self.current_state.time_sec,
            dt=dt_k,
            classical_velocity=round(v_class, 5),
            corrected_velocity=round(v_corr, 5),
            predicted_delta_velocity=round(raw_dv, 5),
            predicted_delta_yaw=round(raw_dw, 5),
            ai_applied=decision.is_applied,
            fallback=decision.is_fallback,
            fallback_reason=decision.fallback_reason,
            confidence=decision.confidence_score,
            ood_score=decision.ood_score,
            inference_latency_ms=round(lat_infer, 3),
            total_latency_ms=round(total_lat, 3),
            watchdog_timeout=watchdog_timeout,
            sensor_valid=val_res.is_valid,
            stationary=is_stat,
            navigation_state_valid=True
        )
        self.telemetry_logger.log_frame(telem_frame)

        self.nav_history_records.append({
            "time_sec": self.current_state.time_sec,
            "dt_sec": dt_k,
            "pos_east_m": new_east,
            "pos_north_m": new_north,
            "heading_rad": new_heading,
            "forward_speed_ms": v_corr,
            "yaw_rate_rads": w_corr
        })

        return self.current_state

    def get_navigation_state(self) -> PlanarNavigationState:
        if self.current_state is None:
            raise RuntimeError("RealtimeNavigationEngine is not initialized.")
        return self.current_state

    def get_runtime_status(self) -> Dict[str, Any]:
        return {
            "is_initialized": self.is_initialized,
            "window_size": self.window_size,
            "buffer_length": len(self.window_buffer),
            "latency": self.latency_monitor.get_summary_statistics(),
            "watchdog": self.watchdog.get_summary(),
            "telemetry_frames_count": len(self.telemetry_logger.frames)
        }

    def get_telemetry(self) -> pd.DataFrame:
        return self.telemetry_logger.to_dataframe()

    def get_trajectory(self) -> DeadReckoningTrajectory:
        """Export accumulated run history as DeadReckoningTrajectory."""
        df = pd.DataFrame(self.nav_history_records)
        if df.empty:
            return DeadReckoningTrajectory(
                np.array([0.0]), np.array([0.1]), np.array([0.0]), np.array([0.0]),
                np.array([0.0]), np.array([0.0]), np.array([0.0]), "OBJ7_REALTIME", 0.0
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
            baseline_name="OBJ7_REALTIME_INTEGRATED",
            total_distance_m=total_dist
        )

    def shutdown(self) -> None:
        """Cleanly shutdown engine and free buffers."""
        self.window_buffer.clear()
        self.is_initialized = False

    def step(self, packet: Union[SensorPacket, Dict[str, Any]]) -> NavigationStepResult:
        """
        Process single sensor packet using structured step API.
        """
        if isinstance(packet, dict):
            t = packet.get("timestamp_sec", packet.get("time_sec", 0.0))
            dt = packet.get("dt_sec", 0.1)
            w_fl = packet.get("wheel_speed_fl_ms", packet.get("wheel_fl", None))
            w_fr = packet.get("wheel_speed_fr_ms", packet.get("wheel_fr", None))
            w_rl = packet.get("wheel_speed_rl_ms", packet.get("wheel_rl", None))
            w_rr = packet.get("wheel_speed_rr_ms", packet.get("wheel_rr", None))
            ax = packet.get("accel_x_ms2", packet.get("accel_x", None))
            yr = packet.get("yaw_rate_rads", packet.get("yaw_rate", None))
        else:
            t = packet.timestamp_sec
            dt = packet.dt_sec
            w_fl = packet.wheel_speed_fl_ms
            w_fr = packet.wheel_speed_fr_ms
            w_rl = packet.wheel_speed_rl_ms
            w_rr = packet.wheel_speed_rr_ms
            ax = packet.accel_x_ms2
            yr = packet.yaw_rate_rads

        st = self.process_sensor_sample(t, dt, w_fl, w_fr, w_rl, w_rr, ax, yr)
        latest_telem = self.telemetry_logger.frames[-1] if self.telemetry_logger.frames else None
        class_st = self.classical_engine.state or st
        is_stable = not (np.isnan(st.p_east_m) or np.isnan(st.p_north_m) or np.isnan(st.forward_speed_ms))

        return NavigationStepResult(
            timestamp=st.time_sec,
            position=(st.p_east_m, st.p_north_m),
            velocity=st.forward_speed_ms,
            heading=st.heading_rad,
            classical_state=class_st,
            corrected_state=st,
            delta_v=latest_telem.predicted_delta_velocity if latest_telem else 0.0,
            delta_omega_z=latest_telem.predicted_delta_yaw if latest_telem else 0.0,
            ai_applied=latest_telem.ai_applied if latest_telem else False,
            fallback_active=latest_telem.fallback if latest_telem else True,
            fallback_reason=latest_telem.fallback_reason if latest_telem else "NONE",
            confidence=latest_telem.confidence if latest_telem else 0.0,
            ood_score=latest_telem.ood_score if latest_telem else 0.0,
            inference_latency_ms=latest_telem.inference_latency_ms if latest_telem else 0.0,
            total_latency_ms=latest_telem.total_latency_ms if latest_telem else 0.0,
            numerical_status="STABLE" if is_stable else "UNSTABLE",
            dt=dt,
            watchdog_status="TIMEOUT" if (latest_telem and latest_telem.watchdog_timeout) else "HEALTHY"
        )

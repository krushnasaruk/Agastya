"""
Selective Correction Policy and Multi-Gate Decision Layer for Objective 6.
Decides whether to apply AI residual corrections or fall back to deterministic classical physics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

from .distribution_monitor import TrainingDistributionMonitor
from .temporal_consistency import TemporalConsistencyMonitor
from .confidence import PredictiveConfidenceEstimator


@dataclass
class PolicyDecision:
    is_applied: bool
    is_fallback: bool
    fallback_reason: str
    is_clamped: bool
    applied_delta_v: float
    applied_delta_w: float
    raw_delta_v: float
    raw_delta_w: float
    ood_score: float
    is_ood: bool
    velocity_jump_ms: float
    is_temporal_consistent: bool
    confidence_score: float
    is_confident: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelectiveCorrectionPolicy:
    """
    Evaluates multi-stage gating policy for residual application.
    """
    def __init__(
        self,
        distribution_monitor: Optional[TrainingDistributionMonitor] = None,
        temporal_monitor: Optional[TemporalConsistencyMonitor] = None,
        confidence_estimator: Optional[PredictiveConfidenceEstimator] = None,
        enable_velocity_correction: bool = True,
        enable_yaw_correction: bool = False,
        enable_sensor_gate: bool = True,
        enable_stationary_gate: bool = True,
        enable_ood_gate: bool = True,
        enable_temporal_gate: bool = True,
        enable_confidence_gate: bool = True,
        hard_velocity_bound_ms: float = 3.0,
        hard_yaw_bound_rads: float = 0.50
    ):
        self.distribution_monitor = distribution_monitor
        self.temporal_monitor = temporal_monitor or TemporalConsistencyMonitor()
        self.confidence_estimator = confidence_estimator or PredictiveConfidenceEstimator()

        self.enable_v = enable_velocity_correction
        self.enable_w = enable_yaw_correction
        self.enable_sensor_gate = enable_sensor_gate
        self.enable_stationary_gate = enable_stationary_gate
        self.enable_ood_gate = enable_ood_gate
        self.enable_temporal_gate = enable_temporal_gate
        self.enable_confidence_gate = enable_confidence_gate

        self.hard_v_bound = hard_velocity_bound_ms
        self.hard_w_bound = hard_yaw_bound_rads

    def reset(self) -> None:
        """Reset internal sequence monitors."""
        self.temporal_monitor.reset()

    def evaluate(
        self,
        raw_delta_v: float,
        raw_delta_w: float,
        feature_vector_or_window: np.ndarray,
        classical_speed_ms: float,
        is_stationary: bool,
        is_sensor_valid: bool = True
    ) -> PolicyDecision:
        """
        Evaluate sequential decision logic.
        """
        # Step 1: Sensor Validity Gate
        if self.enable_sensor_gate and not is_sensor_valid:
            self.temporal_monitor.reset()
            return PolicyDecision(
                is_applied=False,
                is_fallback=True,
                fallback_reason="FALLBACK_SENSOR_DEGRADED",
                is_clamped=False,
                applied_delta_v=0.0,
                applied_delta_w=0.0,
                raw_delta_v=raw_delta_v,
                raw_delta_w=raw_delta_w,
                ood_score=0.0,
                is_ood=False,
                velocity_jump_ms=0.0,
                is_temporal_consistent=False,
                confidence_score=0.0,
                is_confident=False
            )

        # Step 2: Stationary Gate
        if self.enable_stationary_gate and (is_stationary or classical_speed_ms < 0.08):
            self.temporal_monitor.reset()
            return PolicyDecision(
                is_applied=False,
                is_fallback=True,
                fallback_reason="FALLBACK_STATIONARY",
                is_clamped=False,
                applied_delta_v=0.0,
                applied_delta_w=0.0,
                raw_delta_v=raw_delta_v,
                raw_delta_w=raw_delta_w,
                ood_score=0.0,
                is_ood=False,
                velocity_jump_ms=0.0,
                is_temporal_consistent=True,
                confidence_score=0.0,
                is_confident=False
            )

        # Step 3: Out-of-Distribution (OOD) Gate
        ood_score = 0.0
        is_in_dist = True
        ood_thresh = 3.5
        if self.distribution_monitor is not None and self.distribution_monitor.is_fitted:
            ood_score = self.distribution_monitor.compute_ood_score(feature_vector_or_window)
            ood_thresh = self.distribution_monitor.ood_threshold
            is_in_dist = (ood_score <= ood_thresh)

        if self.enable_ood_gate and not is_in_dist:
            return PolicyDecision(
                is_applied=False,
                is_fallback=True,
                fallback_reason="FALLBACK_OOD_FEATURE_SHIFT",
                is_clamped=False,
                applied_delta_v=0.0,
                applied_delta_w=0.0,
                raw_delta_v=raw_delta_v,
                raw_delta_w=raw_delta_w,
                ood_score=round(ood_score, 4),
                is_ood=True,
                velocity_jump_ms=0.0,
                is_temporal_consistent=False,
                confidence_score=0.0,
                is_confident=False
            )

        # Step 4: Temporal Consistency Gate
        temp_eval = self.temporal_monitor.evaluate_step(raw_delta_v, raw_delta_w)
        v_jump = temp_eval["velocity_jump_ms"]
        is_temp_ok = temp_eval["is_consistent"]

        if self.enable_temporal_gate and not is_temp_ok:
            return PolicyDecision(
                is_applied=False,
                is_fallback=True,
                fallback_reason=f"FALLBACK_TEMPORAL_JUMP ({temp_eval['reason']})",
                is_clamped=False,
                applied_delta_v=0.0,
                applied_delta_w=0.0,
                raw_delta_v=raw_delta_v,
                raw_delta_w=raw_delta_w,
                ood_score=round(ood_score, 4),
                is_ood=False,
                velocity_jump_ms=v_jump,
                is_temporal_consistent=False,
                confidence_score=0.0,
                is_confident=False
            )

        # Step 5: Uncertainty & Confidence Gate
        conf_eval = self.confidence_estimator.estimate_confidence(
            raw_delta_v=raw_delta_v,
            raw_delta_w=raw_delta_w,
            ood_score=ood_score,
            ood_threshold=ood_thresh,
            v_jump=v_jump,
            max_v_jump=self.temporal_monitor.max_v_jump,
            is_stationary=False,
            is_sensor_valid=is_sensor_valid
        )
        conf_score = conf_eval["confidence"]
        is_conf = conf_eval["is_confident"]

        if self.enable_confidence_gate and not is_conf:
            return PolicyDecision(
                is_applied=False,
                is_fallback=True,
                fallback_reason=f"FALLBACK_LOW_CONFIDENCE (C={conf_score:.2f})",
                is_clamped=False,
                applied_delta_v=0.0,
                applied_delta_w=0.0,
                raw_delta_v=raw_delta_v,
                raw_delta_w=raw_delta_w,
                ood_score=round(ood_score, 4),
                is_ood=False,
                velocity_jump_ms=v_jump,
                is_temporal_consistent=True,
                confidence_score=conf_score,
                is_confident=False
            )

        # Step 6: Physical Safety Bounding & Channel Enablement
        target_dv = raw_delta_v if self.enable_v else 0.0
        target_dw = raw_delta_w if self.enable_w else 0.0

        clamped_dv = float(np.clip(target_dv, -self.hard_v_bound, self.hard_v_bound))
        clamped_dw = float(np.clip(target_dw, -self.hard_w_bound, self.hard_w_bound))
        is_clamped = (clamped_dv != target_dv) or (clamped_dw != target_dw)

        return PolicyDecision(
            is_applied=True,
            is_fallback=False,
            fallback_reason="NONE_APPLIED",
            is_clamped=is_clamped,
            applied_delta_v=clamped_dv,
            applied_delta_w=clamped_dw,
            raw_delta_v=raw_delta_v,
            raw_delta_w=raw_delta_w,
            ood_score=round(ood_score, 4),
            is_ood=False,
            velocity_jump_ms=v_jump,
            is_temporal_consistent=True,
            confidence_score=conf_score,
            is_confident=True
        )

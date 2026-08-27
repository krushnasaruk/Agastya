"""
Canonical Causal Feature Registry for Project AGASTYA (Objective 5).
Defines exact 16 causal features, deterministic ordering, mathematical formulation,
units, causal status, and normalization policies.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass(frozen=True)
class CausalFeatureSpec:
    index: int
    name: str
    source_signal: str
    formula: str
    units: str
    causal_status: str
    normalization_policy: str
    physical_meaning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CANONICAL_FEATURES: List[CausalFeatureSpec] = [
    CausalFeatureSpec(
        index=0,
        name="wheel_speed_fl_ms",
        source_signal="CAN Front-Left Wheel Speed",
        formula="v_FL",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Steered front-left wheel linear velocity"
    ),
    CausalFeatureSpec(
        index=1,
        name="wheel_speed_fr_ms",
        source_signal="CAN Front-Right Wheel Speed",
        formula="v_FR",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Steered front-right wheel linear velocity"
    ),
    CausalFeatureSpec(
        index=2,
        name="wheel_speed_rl_ms",
        source_signal="CAN Rear-Left Wheel Speed",
        formula="v_RL",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Unsteered rear-left wheel linear velocity"
    ),
    CausalFeatureSpec(
        index=3,
        name="wheel_speed_rr_ms",
        source_signal="CAN Rear-Right Wheel Speed",
        formula="v_RR",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Unsteered rear-right wheel linear velocity"
    ),
    CausalFeatureSpec(
        index=4,
        name="wheel_speed_rear_mean_ms",
        source_signal="CAN Rear Axle Average",
        formula="(v_RL + v_RR) / 2",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Primary unsteered forward rolling velocity"
    ),
    CausalFeatureSpec(
        index=5,
        name="wheel_speed_rear_diff_ms",
        source_signal="CAN Rear Axle Differential",
        formula="v_RR - v_RL",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Differential wheel speed directly proportional to vehicle turning rate"
    ),
    CausalFeatureSpec(
        index=6,
        name="wheel_speed_front_rear_diff_ms",
        source_signal="Axle Speed Difference",
        formula="(v_FL + v_FR)/2 - (v_RL + v_RR)/2",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Longitudinal axle slip indicator during traction or braking"
    ),
    CausalFeatureSpec(
        index=7,
        name="accel_x_ms2",
        source_signal="CAN Longitudinal Acceleration",
        formula="a_x",
        units="m/s^2",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Chassis longitudinal specific force along body +X"
    ),
    CausalFeatureSpec(
        index=8,
        name="jerk_longitudinal_ms3",
        source_signal="Causal Accel Backward Difference",
        formula="(a_x[k] - a_x[k-1]) / dt",
        units="m/s^3",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Rate of change of acceleration indicating aggressive driving transitions"
    ),
    CausalFeatureSpec(
        index=9,
        name="yaw_rate_rads",
        source_signal="CAN Gyroscope Yaw Rate",
        formula="omega_z",
        units="rad/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Vehicle body rotation rate around vertical Z axis"
    ),
    CausalFeatureSpec(
        index=10,
        name="yaw_acceleration_rads2",
        source_signal="Causal Yaw Backward Difference",
        formula="(omega_z[k] - omega_z[k-1]) / dt",
        units="rad/s^2",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Angular acceleration during turn entry/exit"
    ),
    CausalFeatureSpec(
        index=11,
        name="dt_sec",
        source_signal="Sampling Interval",
        formula="t[k] - t[k-1]",
        units="seconds",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Dynamic timestep capturing sensor loop jitter"
    ),
    CausalFeatureSpec(
        index=12,
        name="classical_forward_speed_ms",
        source_signal="Objective 3 Baseline Estimator",
        formula="v_classical[k]",
        units="m/s",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Current unassisted dead-reckoning speed estimate"
    ),
    CausalFeatureSpec(
        index=13,
        name="estimated_curvature_inv_m",
        source_signal="Kinematic Curvature",
        formula="omega_z / max(v_classical, 0.1)",
        units="1/m",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="Z_SCORE",
        physical_meaning="Inverse path radius indicating sharpness of vehicle turning arc"
    ),
    CausalFeatureSpec(
        index=14,
        name="is_stationary_flag",
        source_signal="ZUPT Detector",
        formula="1 if v < 0.08 m/s else 0",
        units="boolean bit",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="PASS_THROUGH",
        physical_meaning="Stationary vehicle flag"
    ),
    CausalFeatureSpec(
        index=15,
        name="slip_detected_flag",
        source_signal="Kinematic Slip Gate",
        formula="1 if |v_RR - v_RL| > 2.5 m/s and v > 2.0 else 0",
        units="boolean bit",
        causal_status="STRICTLY CAUSAL",
        normalization_policy="PASS_THROUGH",
        physical_meaning="Wheel spin / slip event detection flag"
    )
]

CANONICAL_FEATURE_NAMES: List[str] = [f.name for f in CANONICAL_FEATURES]
NUM_CANONICAL_FEATURES: int = len(CANONICAL_FEATURES)  # Exactly 16


def validate_feature_matrix_columns(columns: List[str]) -> bool:
    """
    Validates that exactly 16 canonical features exist in exact deterministic order.
    """
    if len(columns) != NUM_CANONICAL_FEATURES:
        raise ValueError(f"Feature count mismatch: expected {NUM_CANONICAL_FEATURES}, got {len(columns)}")
    for i, (expected, actual) in enumerate(zip(CANONICAL_FEATURE_NAMES, columns)):
        if expected != actual:
            raise ValueError(f"Feature order mismatch at index {i}: expected '{expected}', got '{actual}'")
    return True

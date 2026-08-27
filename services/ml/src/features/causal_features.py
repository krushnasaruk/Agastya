"""
Causal Feature Engineering & Registry for Project AGASTYA (Objective 4).
Extracts strictly causal kinematic and dynamic features from onboard vehicle sensors
without future lookahead and with zero reference/GNSS leakage.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class CausalFeatureMetadata:
    feature_name: str
    source_signal: str
    mathematical_formula: str
    units: str
    temporal_dependency: str
    causal_status: str
    physical_interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CAUSAL_FEATURE_REGISTRY: Dict[str, CausalFeatureMetadata] = {
    "wheel_speed_fl_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_fl_ms",
        source_signal="CAN Front-Left Wheel Speed",
        mathematical_formula="v_FL",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Steered front-left wheel linear velocity"
    ),
    "wheel_speed_fr_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_fr_ms",
        source_signal="CAN Front-Right Wheel Speed",
        mathematical_formula="v_FR",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Steered front-right wheel linear velocity"
    ),
    "wheel_speed_rl_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_rl_ms",
        source_signal="CAN Rear-Left Wheel Speed",
        mathematical_formula="v_RL",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Unsteered rear-left wheel linear velocity"
    ),
    "wheel_speed_rr_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_rr_ms",
        source_signal="CAN Rear-Right Wheel Speed",
        mathematical_formula="v_RR",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Unsteered rear-right wheel linear velocity"
    ),
    "wheel_speed_rear_mean_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_rear_mean_ms",
        source_signal="CAN Rear Axle Average",
        mathematical_formula="(v_RL + v_RR) / 2",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Primary unsteered forward rolling velocity"
    ),
    "wheel_speed_rear_diff_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_rear_diff_ms",
        source_signal="CAN Rear Axle Differential",
        mathematical_formula="v_RR - v_RL",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Differential wheel speed directly proportional to vehicle turning rate"
    ),
    "wheel_speed_front_rear_diff_ms": CausalFeatureMetadata(
        feature_name="wheel_speed_front_rear_diff_ms",
        source_signal="Axle Speed Difference",
        mathematical_formula="(v_FL + v_FR)/2 - (v_RL + v_RR)/2",
        units="m/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Longitudinal axle slip indicator during traction or braking"
    ),
    "accel_x_ms2": CausalFeatureMetadata(
        feature_name="accel_x_ms2",
        source_signal="CAN Longitudinal Acceleration",
        mathematical_formula="a_x",
        units="m/s^2",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Chassis longitudinal specific force / acceleration along body +X"
    ),
    "jerk_longitudinal_ms3": CausalFeatureMetadata(
        feature_name="jerk_longitudinal_ms3",
        source_signal="Causal Accel Derivative",
        mathematical_formula="(a_x[k] - a_x[k-1]) / dt",
        units="m/s^3",
        temporal_dependency="Past Causal Difference (k, k-1)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Rate of change of acceleration indicating aggressive driving transitions"
    ),
    "yaw_rate_rads": CausalFeatureMetadata(
        feature_name="yaw_rate_rads",
        source_signal="CAN Gyroscope Yaw Rate",
        mathematical_formula="omega_z",
        units="rad/s",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Vehicle body rotation rate around vertical Z axis"
    ),
    "yaw_acceleration_rads2": CausalFeatureMetadata(
        feature_name="yaw_acceleration_rads2",
        source_signal="Causal Yaw Derivative",
        mathematical_formula="(omega_z[k] - omega_z[k-1]) / dt",
        units="rad/s^2",
        temporal_dependency="Past Causal Difference (k, k-1)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Angular acceleration during turn entry/exit"
    ),
    "dt_sec": CausalFeatureMetadata(
        feature_name="dt_sec",
        source_signal="Sampling Interval",
        mathematical_formula="t[k] - t[k-1]",
        units="seconds",
        temporal_dependency="Causal Dynamic Step",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Dynamic timestep capturing sensor loop jitter"
    ),
    "classical_forward_speed_ms": CausalFeatureMetadata(
        feature_name="classical_forward_speed_ms",
        source_signal="Objective 3 Baseline Estimator",
        mathematical_formula="v_classical[k]",
        units="m/s",
        temporal_dependency="Causal Filter Output",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Current unassisted dead-reckoning speed estimate"
    ),
    "estimated_curvature_inv_m": CausalFeatureMetadata(
        feature_name="estimated_curvature_inv_m",
        source_signal="Kinematic Curvature",
        mathematical_formula="omega_z / max(v_classical, 0.1)",
        units="1/m",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Inverse path radius indicating sharpness of vehicle turning arc"
    ),
    "is_stationary_flag": CausalFeatureMetadata(
        feature_name="is_stationary_flag",
        source_signal="ZUPT Detector",
        mathematical_formula="1 if v < 0.08 m/s else 0",
        units="boolean bit",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Stationary vehicle flag"
    ),
    "slip_detected_flag": CausalFeatureMetadata(
        feature_name="slip_detected_flag",
        source_signal="Kinematic Slip Gate",
        mathematical_formula="1 if |v_RR - v_RL| > 2.5 m/s and v > 2.0 else 0",
        units="boolean bit",
        temporal_dependency="Instantaneous (Epoch k)",
        causal_status="STRICTLY CAUSAL",
        physical_interpretation="Wheel spin / slip event detection flag"
    )
}


class CausalFeatureExtractor:
    """
    Extracts strictly causal feature representations from onboard navigation inputs.
    """
    @classmethod
    def extract_features(
        cls,
        navigation_inputs_df: pd.DataFrame,
        classical_speed_ms: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Extract all causal features into a clean, leakage-free DataFrame.
        """
        df = navigation_inputs_df.copy()
        n = len(df)
        t = df["time_sec"].to_numpy()
        dt = df["dt_sec"].to_numpy()

        v_fl = df.get("wheel_speed_fl_ms", pd.Series(np.zeros(n))).to_numpy()
        v_fr = df.get("wheel_speed_fr_ms", pd.Series(np.zeros(n))).to_numpy()
        v_rl = df.get("wheel_speed_rl_ms", pd.Series(np.zeros(n))).to_numpy()
        v_rr = df.get("wheel_speed_rr_ms", pd.Series(np.zeros(n))).to_numpy()
        ax = df.get("accel_x_ms2", pd.Series(np.zeros(n))).to_numpy()
        yr = df.get("yaw_rate_rads", pd.Series(np.zeros(n))).to_numpy()

        # 1. Axle Averages and Differences
        v_rear_mean = 0.5 * (v_rl + v_rr)
        v_rear_diff = v_rr - v_rl
        v_front_diff = v_fr - v_fl
        v_front_mean = 0.5 * (v_fl + v_fr)
        v_axle_diff = v_front_mean - v_rear_mean

        # 2. Causal Backward Derivatives (Zero Future Lookahead)
        jerk = np.zeros(n, dtype=np.float64)
        yaw_acc = np.zeros(n, dtype=np.float64)
        for k in range(1, n):
            dt_k = max(dt[k], 0.005)
            jerk[k] = (ax[k] - ax[k - 1]) / dt_k
            yaw_acc[k] = (yr[k] - yr[k - 1]) / dt_k

        # 3. Classical Speed & Curvature
        v_class = classical_speed_ms if classical_speed_ms is not None else v_rear_mean
        curvature = yr / np.maximum(v_class, 0.1)

        # 4. Status Flags
        is_stat = (v_class < 0.08).astype(np.float64)
        is_slip = ((np.abs(v_rear_diff) > 2.5) & (v_class > 2.0)).astype(np.float64)

        features_df = pd.DataFrame({
            "wheel_speed_fl_ms": v_fl,
            "wheel_speed_fr_ms": v_fr,
            "wheel_speed_rl_ms": v_rl,
            "wheel_speed_rr_ms": v_rr,
            "wheel_speed_rear_mean_ms": v_rear_mean,
            "wheel_speed_rear_diff_ms": v_rear_diff,
            "wheel_speed_front_rear_diff_ms": v_axle_diff,
            "accel_x_ms2": ax,
            "jerk_longitudinal_ms3": jerk,
            "yaw_rate_rads": yr,
            "yaw_acceleration_rads2": yaw_acc,
            "dt_sec": dt,
            "classical_forward_speed_ms": v_class,
            "estimated_curvature_inv_m": curvature,
            "is_stationary_flag": is_stat,
            "slip_detected_flag": is_slip
        })

        return features_df

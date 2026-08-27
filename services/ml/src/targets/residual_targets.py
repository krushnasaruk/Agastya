"""
Residual Learning Targets Formulation for Project AGASTYA (Objective 4).
Mathematically formulates and extracts candidate residual targets:
  - Target A: Forward Velocity Residual (delta_v) [RECOMMENDED PRIMARY TARGET]
  - Target B: Yaw Rate / Heading Residual (delta_yaw_rate, delta_psi)
  - Target C: Incremental Displacement Residual (delta_dx, delta_dy)
  - Target D: Global Position Residual (delta_p_east, delta_p_north)
  - Target E: Wheel-Odometry Speed Residual (delta_v_wheel)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class ResidualTargetsContainer:
    """
    Container storing all candidate residual targets aligned with sequence timesteps.
    """
    timestamps_sec: np.ndarray
    delta_velocity_ms: np.ndarray                # Target A: v_ref - v_classical
    delta_yaw_rate_rads: np.ndarray             # Target B1: omega_ref - omega_classical
    delta_heading_rad: np.ndarray               # Target B2: wrap_to_pi(psi_ref - psi_classical)
    delta_disp_east_m: np.ndarray               # Target C1: dE_ref - dE_classical
    delta_disp_north_m: np.ndarray              # Target C2: dN_ref - dN_classical
    delta_pos_east_m: np.ndarray                # Target D1: pE_ref - pE_classical
    delta_pos_north_m: np.ndarray               # Target D2: pN_ref - pN_classical
    delta_wheel_speed_ms: np.ndarray            # Target E: v_ref - v_wheel_raw

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time_sec": self.timestamps_sec,
            "target_a_delta_v_ms": self.delta_velocity_ms,
            "target_b_delta_yaw_rate_rads": self.delta_yaw_rate_rads,
            "target_b_delta_heading_rad": self.delta_heading_rad,
            "target_b_delta_heading_deg": np.degrees(self.delta_heading_rad),
            "target_c_delta_disp_east_m": self.delta_disp_east_m,
            "target_c_delta_disp_north_m": self.delta_disp_north_m,
            "target_d_delta_pos_east_m": self.delta_pos_east_m,
            "target_d_delta_pos_north_m": self.delta_pos_north_m,
            "target_e_delta_wheel_speed_ms": self.delta_wheel_speed_ms
        })


class ResidualTargetExtractor:
    """
    Computes aligned residual learning targets between classical dead-reckoning outputs
    and offline VBOX reference ground truth.
    """
    @classmethod
    def extract_all_targets(
        cls,
        time_sec: np.ndarray,
        dt_sec: np.ndarray,
        classical_pos_east_m: np.ndarray,
        classical_pos_north_m: np.ndarray,
        classical_heading_rad: np.ndarray,
        classical_speed_ms: np.ndarray,
        classical_yaw_rate_rads: np.ndarray,
        raw_wheel_speed_ms: np.ndarray,
        ref_pos_east_m: np.ndarray,
        ref_pos_north_m: np.ndarray,
        ref_heading_rad: np.ndarray,
        ref_speed_ms: np.ndarray
    ) -> ResidualTargetsContainer:
        """
        Extract all 5 candidate residual target streams.
        """
        n = len(time_sec)
        
        # Target A: Forward Velocity Residual
        delta_v = ref_speed_ms - classical_speed_ms

        # Target B1 & B2: Yaw Rate & Heading Residuals
        # Numerically compute reference yaw rate from heading derivative if not direct
        unwrapped_ref_heading = np.unwrap(ref_heading_rad)
        ref_yaw_rate = np.gradient(unwrapped_ref_heading, time_sec)
        delta_yaw_rate = ref_yaw_rate - classical_yaw_rate_rads
        delta_heading = np.array([wrap_to_pi(ref_heading_rad[i] - classical_heading_rad[i]) for i in range(n)])

        # Target C: Incremental Displacement Residual
        d_east_ref = np.zeros(n)
        d_north_ref = np.zeros(n)
        d_east_class = np.zeros(n)
        d_north_class = np.zeros(n)

        if n > 1:
            d_east_ref[1:] = np.diff(ref_pos_east_m)
            d_north_ref[1:] = np.diff(ref_pos_north_m)
            d_east_class[1:] = np.diff(classical_pos_east_m)
            d_north_class[1:] = np.diff(classical_pos_north_m)

        delta_disp_east = d_east_ref - d_east_class
        delta_disp_north = d_north_ref - d_north_class

        # Target D: Global Position Residual
        delta_pos_east = ref_pos_east_m - classical_pos_east_m
        delta_pos_north = ref_pos_north_m - classical_pos_north_m

        # Target E: Raw Wheel Speed Residual
        delta_wheel_v = ref_speed_ms - raw_wheel_speed_ms

        return ResidualTargetsContainer(
            timestamps_sec=np.asarray(time_sec),
            delta_velocity_ms=delta_v,
            delta_yaw_rate_rads=delta_yaw_rate,
            delta_heading_rad=delta_heading,
            delta_disp_east_m=delta_disp_east,
            delta_disp_north_m=delta_disp_north,
            delta_pos_east_m=delta_pos_east,
            delta_pos_north_m=delta_pos_north,
            delta_wheel_speed_ms=delta_wheel_v
        )

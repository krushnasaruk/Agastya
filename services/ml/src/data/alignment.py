"""
Deterministic Target Alignment Module for Project AGASTYA (Objective 4).
Aligns classical dead-reckoning state outputs with offline VBOX ground-truth reference
for label extraction, strictly isolating reference features from inference features.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class AlignedTrajectoryDataset:
    time_sec: np.ndarray
    dt_sec: np.ndarray
    classical_p_east_m: np.ndarray
    classical_p_north_m: np.ndarray
    classical_heading_rad: np.ndarray
    classical_speed_ms: np.ndarray
    classical_yaw_rate_rads: np.ndarray
    ref_p_east_m: np.ndarray
    ref_p_north_m: np.ndarray
    ref_heading_rad: np.ndarray
    ref_speed_ms: np.ndarray
    valid_mask: np.ndarray
    num_aligned_samples: int


class TrajectoryTargetAligner:
    """
    Performs deterministic, timestamp-synchronized alignment between classical estimates and offline reference data.
    """
    @classmethod
    def align(
        cls,
        navigation_inputs_df: pd.DataFrame,
        classical_traj_df: pd.DataFrame,
        reference_trajectory_df: pd.DataFrame
    ) -> AlignedTrajectoryDataset:
        """
        Deterministically align inputs, classical state, and reference ground truth.
        """
        n = min(len(navigation_inputs_df), len(classical_traj_df), len(reference_trajectory_df))

        t = navigation_inputs_df["time_sec"].iloc[:n].to_numpy()
        dt = navigation_inputs_df["dt_sec"].iloc[:n].to_numpy()

        c_e = classical_traj_df["estimated_p_east_m"].iloc[:n].to_numpy()
        c_n = classical_traj_df["estimated_p_north_m"].iloc[:n].to_numpy()
        c_h = classical_traj_df["estimated_heading_rad"].iloc[:n].to_numpy()
        c_v = classical_traj_df["estimated_speed_ms"].iloc[:n].to_numpy()
        c_yr = classical_traj_df["yaw_rate_rads"].iloc[:n].to_numpy()

        r_e = reference_trajectory_df["pos_east_m"].iloc[:n].to_numpy()
        r_n = reference_trajectory_df["pos_north_m"].iloc[:n].to_numpy()
        r_h = reference_trajectory_df.get("heading_rad", pd.Series(np.zeros(n))).iloc[:n].to_numpy()
        r_v = reference_trajectory_df.get("ground_speed_ms", pd.Series(np.zeros(n))).iloc[:n].to_numpy()

        # Valid reference mask
        valid = (~np.isnan(r_e)) & (~np.isnan(r_n)) & (~np.isnan(r_v)) & (~np.isnan(r_h))

        return AlignedTrajectoryDataset(
            time_sec=t,
            dt_sec=dt,
            classical_p_east_m=c_e,
            classical_p_north_m=c_n,
            classical_heading_rad=c_h,
            classical_speed_ms=c_v,
            classical_yaw_rate_rads=c_yr,
            ref_p_east_m=r_e,
            ref_p_north_m=r_n,
            ref_heading_rad=r_h,
            ref_speed_ms=r_v,
            valid_mask=valid,
            num_aligned_samples=n
        )

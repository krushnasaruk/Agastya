"""
Physical Consistency & Kinematic Validation Engine for Project AGASTYA.
Diagnoses sensor coherence, tire slip episodes, yaw rate consistency,
and GPS discontinuity/teleportation anomalies with documented threshold rationales.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class PhysicalConsistencyReport:
    total_samples: int
    num_gps_jump_anomalies: int
    num_wheel_slip_anomalies: int
    num_yaw_coherence_anomalies: int
    max_detected_slip_ms: float
    max_position_jump_ms: float
    mean_wheel_gyro_yaw_correlation: float
    is_physically_consistent: bool
    threshold_definitions: Dict[str, Any]
    diagnostics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhysicalConsistencyChecker:
    """
    Non-destructive kinematic diagnostics for ground vehicle data streams.
    Thresholds are strictly documented with physical rationale and provisional status.
    """
    def __init__(
        self,
        track_width_m: float = 1.47,             # [VERIFIED SPEC] Ford Fiesta nominal rear track
        max_valid_vehicle_speed_ms: float = 70.0,# [PROVISIONAL] ~250 km/h passenger car physical limit
        slip_threshold_ms: float = 2.5,          # [PROVISIONAL] ~9 km/h longitudinal slip detection limit
        yaw_rate_mismatch_rads: float = 0.5      # [PROVISIONAL] ~28.6 deg/s kinematic vs gyro mismatch limit
    ):
        self.track_width_m = track_width_m
        self.max_valid_vehicle_speed_ms = max_valid_vehicle_speed_ms
        self.slip_threshold_ms = slip_threshold_ms
        self.yaw_rate_mismatch_rads = yaw_rate_mismatch_rads

    def get_threshold_definitions(self) -> Dict[str, Any]:
        return {
            "track_width_m": {
                "value": self.track_width_m,
                "unit": "meters",
                "status": "VERIFIED FROM VEHICLE SPEC",
                "rationale": "Ford Fiesta Mk7 rear track width for differential odometry"
            },
            "max_valid_vehicle_speed_ms": {
                "value": self.max_valid_vehicle_speed_ms,
                "unit": "m/s",
                "status": "PROVISIONAL",
                "rationale": "Upper bound (252 km/h) to flag impossible GPS teleportation jumps"
            },
            "slip_threshold_ms": {
                "value": self.slip_threshold_ms,
                "unit": "m/s",
                "status": "PROVISIONAL",
                "rationale": "Threshold to distinguish transient tire slip from steady-state rolling"
            },
            "yaw_rate_mismatch_rads": {
                "value": self.yaw_rate_mismatch_rads,
                "unit": "rad/s",
                "status": "PROVISIONAL",
                "rationale": "Tolerance between differential wheel rate and CAN gyro rate"
            }
        }

    def check_consistency(
        self,
        normalized_data: Dict[str, np.ndarray],
        dt_array: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], PhysicalConsistencyReport]:
        """
        Evaluate physical consistency across wheel speeds, gyroscopes, accelerometers, and GPS.

        Returns:
            anomaly_masks: Dict of boolean masks flagging anomaly conditions
            report: PhysicalConsistencyReport summary
        """
        n = len(dt_array)
        diagnostics: List[str] = []

        gps_jump_mask = np.zeros(n, dtype=bool)
        wheel_slip_mask = np.zeros(n, dtype=bool)
        yaw_coherence_mask = np.zeros(n, dtype=bool)

        max_slip = 0.0
        max_jump_speed = 0.0
        yaw_corr = 1.0

        # 1. Wheel Odometry vs GPS Ground Speed Consistency (Slip Detection)
        has_wheels = all(k in normalized_data for k in ["wheel_speed_rl_ms", "wheel_speed_rr_ms"])
        has_gps_speed = "gps_speed_ms" in normalized_data

        if has_wheels and has_gps_speed:
            v_rl = normalized_data["wheel_speed_rl_ms"]
            v_rr = normalized_data["wheel_speed_rr_ms"]
            v_wheel_avg = (v_rl + v_rr) * 0.5
            v_gps = normalized_data["gps_speed_ms"]

            slip_magnitude = np.abs(v_wheel_avg - v_gps)
            max_slip = float(np.max(slip_magnitude))
            # Flag slip when difference exceeds threshold during active travel
            wheel_slip_mask = (slip_magnitude > self.slip_threshold_ms) & (v_gps > 1.0)
            num_slip = int(np.sum(wheel_slip_mask))
            if num_slip > 0:
                diagnostics.append(f"Detected {num_slip} wheel slip / discrepancy samples (max: {max_slip:.2f} m/s).")

        # 2. Wheel Differential Yaw Rate vs Gyroscope Yaw Rate
        has_yaw_gyro = "yaw_rate_rads" in normalized_data
        if has_wheels and has_yaw_gyro:
            v_rl = normalized_data["wheel_speed_rl_ms"]
            v_rr = normalized_data["wheel_speed_rr_ms"]
            gyro_yaw = normalized_data["yaw_rate_rads"]

            # Differential kinematic yaw rate: omega_z = (v_RR - v_RL) / W_track
            kinematic_yaw_rate = (v_rr - v_rl) / max(self.track_width_m, 0.1)

            yaw_diff = np.abs(kinematic_yaw_rate - gyro_yaw)
            yaw_coherence_mask = (yaw_diff > self.yaw_rate_mismatch_rads)

            # Compute correlation during turning (gyro > 0.05 rad/s)
            active_turn = np.abs(gyro_yaw) > 0.05
            if np.sum(active_turn) > 10:
                corr_mat = np.corrcoef(kinematic_yaw_rate[active_turn], gyro_yaw[active_turn])
                yaw_corr = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 1.0

            num_yaw_mismatch = int(np.sum(yaw_coherence_mask))
            if num_yaw_mismatch > 0:
                diagnostics.append(f"Detected {num_yaw_mismatch} kinematic vs gyro yaw rate mismatches.")

        # 3. GPS Position Jump / Discontinuity Check
        has_coords = all(k in normalized_data for k in ["latitude_deg", "longitude_deg"])
        if has_coords and n >= 2:
            lat = normalized_data["latitude_deg"]
            lon = normalized_data["longitude_deg"]

            # Approximate metric deltas (deg to meters: 1 deg lat ~ 111km)
            d_lat_m = np.diff(lat) * 111139.0
            mid_lat_rad = np.radians(lat[:-1])
            d_lon_m = np.diff(lon) * (111139.0 * np.cos(mid_lat_rad))
            step_dist = np.sqrt(d_lat_m ** 2 + d_lon_m ** 2)

            safe_dt = np.where(dt_array[1:] > 1e-4, dt_array[1:], 0.1)
            apparent_speed = step_dist / safe_dt

            jump_indices = np.where(apparent_speed > self.max_valid_vehicle_speed_ms)[0]
            if len(jump_indices) > 0:
                gps_jump_mask[jump_indices + 1] = True
                max_jump_speed = float(np.max(apparent_speed[jump_indices]))
                diagnostics.append(f"Detected {len(jump_indices)} GPS position jump anomalies (max apparent speed: {max_jump_speed:.2f} m/s).")

        num_jumps = int(np.sum(gps_jump_mask))
        num_slips = int(np.sum(wheel_slip_mask))
        num_yaw_anom = int(np.sum(yaw_coherence_mask))

        is_consistent = (num_jumps == 0) and (num_slips < int(0.15 * n))

        report = PhysicalConsistencyReport(
            total_samples=n,
            num_gps_jump_anomalies=num_jumps,
            num_wheel_slip_anomalies=num_slips,
            num_yaw_coherence_anomalies=num_yaw_anom,
            max_detected_slip_ms=round(max_slip, 3),
            max_position_jump_ms=round(max_jump_speed, 3),
            mean_wheel_gyro_yaw_correlation=round(yaw_corr, 3),
            is_physically_consistent=is_consistent,
            threshold_definitions=self.get_threshold_definitions(),
            diagnostics=diagnostics
        )

        anomaly_masks = {
            "gps_jump_anomaly_mask": gps_jump_mask,
            "wheel_slip_anomaly_mask": wheel_slip_mask,
            "yaw_coherence_anomaly_mask": yaw_coherence_mask
        }

        return anomaly_masks, report

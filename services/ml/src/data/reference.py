"""
Reference Trajectory Builder for Project AGASTYA.
Extracts, converts, and structures the high-accuracy VBOX reference GNSS stream
into standardized metric ground-truth positions, velocities, and headings.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .coordinates import GeodeticConverter


@dataclass
class ReferenceTrajectory:
    timestamps_sec: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    up_m: np.ndarray
    ground_speed_ms: np.ndarray
    velocity_east_ms: np.ndarray
    velocity_north_ms: np.ndarray
    heading_rad: np.ndarray
    total_distance_m: float
    origin_metadata: Dict[str, Any]
    roll_pitch_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_distance_m": round(self.total_distance_m, 2),
            "num_steps": len(self.timestamps_sec),
            "origin_metadata": self.origin_metadata,
            "roll_pitch_available": self.roll_pitch_available
        }


class ReferenceTrajectoryBuilder:
    """
    Constructs local metric reference trajectories from verified VBOX GPS observations.
    """
    def __init__(self, geodetic_converter: Optional[GeodeticConverter] = None):
        self.geo_conv = geodetic_converter if geodetic_converter is not None else GeodeticConverter()

    def build_reference(
        self,
        time_sec: np.ndarray,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
        altitude_m: Optional[np.ndarray] = None,
        gps_speed_ms: Optional[np.ndarray] = None,
        heading_rad: Optional[np.ndarray] = None
    ) -> ReferenceTrajectory:
        """
        Build a complete local metric ground-truth trajectory.
        """
        time_arr = np.asarray(time_sec, dtype=np.float64)
        lat_arr = np.asarray(latitude_deg, dtype=np.float64)
        lon_arr = np.asarray(longitude_deg, dtype=np.float64)
        alt_arr = np.asarray(altitude_m, dtype=np.float64) if altitude_m is not None else np.zeros_like(lat_arr)
        n = len(time_arr)

        # 1. Metric Local ENU Transformation
        if not self.geo_conv.is_initialized:
            self.geo_conv.initialize_origin(lat_arr[0], lon_arr[0], alt_arr[0])

        east_m, north_m, up_m = self.geo_conv.geodetic_to_enu(lat_arr, lon_arr, alt_arr)

        # 2. Total Cumulative Trajectory Distance
        if n >= 2:
            d_east = np.diff(east_m)
            d_north = np.diff(north_m)
            step_distances = np.sqrt(d_east ** 2 + d_north ** 2)
            total_dist = float(np.sum(step_distances))
        else:
            total_dist = 0.0

        # 3. Ground Speed (from VBOX or calculated from positions if unavailable)
        if gps_speed_ms is not None:
            speed_arr = np.asarray(gps_speed_ms, dtype=np.float64)
        else:
            speed_arr = np.zeros(n, dtype=np.float64)
            if n >= 2:
                dt = np.diff(time_arr)
                dt[dt == 0] = 0.1  # Prevent div/0
                computed_speed = step_distances / dt
                speed_arr[1:] = computed_speed
                speed_arr[0] = speed_arr[1]

        # 4. Ground Heading & Decomposed Velocity Vector
        if heading_rad is not None:
            head_arr = np.asarray(heading_rad, dtype=np.float64)
        else:
            head_arr = np.zeros(n, dtype=np.float64)
            if n >= 2:
                # Heading in ENU: angle from North (0 rad at North, pi/2 at East)
                computed_heading = np.arctan2(d_east, d_north)
                # Wrap to [0, 2*pi)
                computed_heading = (computed_heading + 2 * np.pi) % (2 * np.pi)
                head_arr[1:] = computed_heading
                head_arr[0] = head_arr[1]

        # Velocity in East & North: v_E = speed * sin(heading), v_N = speed * cos(heading)
        vel_east = speed_arr * np.sin(head_arr)
        vel_north = speed_arr * np.cos(head_arr)

        return ReferenceTrajectory(
            timestamps_sec=time_arr,
            east_m=east_m,
            north_m=north_m,
            up_m=up_m,
            ground_speed_ms=speed_arr,
            velocity_east_ms=vel_east,
            velocity_north_ms=vel_north,
            heading_rad=head_arr,
            total_distance_m=total_dist,
            origin_metadata=self.geo_conv.get_origin_dict(),
            roll_pitch_available=False  # Verified limitation of single-antenna VBOX
        )

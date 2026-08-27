"""
GNSS (Global Navigation Satellite System) Receiver Model.
Simulates satellite constellations, Dilution of Precision (DOP),
multipath noise, latency, and signal dropouts.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List


class GNSSFixType(IntEnum):
    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2
    DGPS = 3
    RTK_FLOAT = 4
    RTK_FIX = 5


@dataclass
class GNSSReading:
    timestamp: float
    position: np.ndarray        # [North, East, Down] in meters, shape (3,)
    velocity: np.ndarray        # [v_north, v_east, v_down] in m/s, shape (3,)
    fix_type: GNSSFixType
    satellites_in_view: int
    hdop: float                 # Horizontal Dilution of Precision
    vdop: float                 # Vertical Dilution of Precision
    covariance: np.ndarray      # 3x3 position covariance matrix
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "position": [float(x) for x in self.position],
            "velocity": [float(x) for x in self.velocity],
            "fix_type": int(self.fix_type),
            "satellites_in_view": int(self.satellites_in_view),
            "hdop": float(self.hdop),
            "vdop": float(self.vdop),
            "cov_diag": [float(self.covariance[i, i]) for i in range(3)],
            "is_valid": self.is_valid
        }


class GNSSReceiver:
    def __init__(
        self,
        base_horizontal_std: float = 1.2,    # meters
        base_vertical_std: float = 2.5,      # meters
        base_velocity_std: float = 0.15,     # m/s
        nominal_satellites: int = 12,
        seed: Optional[int] = None
    ):
        self.base_horizontal_std = base_horizontal_std
        self.base_vertical_std = base_vertical_std
        self.base_velocity_std = base_velocity_std
        self.nominal_satellites = nominal_satellites
        self.is_jammed = False
        self.multipath_active = False
        self.multipath_bias = np.zeros(3)
        self.rng = np.random.RandomState(seed)

    def set_jamming(self, jammed: bool):
        """Simulate EW jamming or total GNSS signal denial."""
        self.is_jammed = jammed

    def set_multipath(self, active: bool, bias_magnitude: float = 15.0):
        """Simulate urban canyon multipath reflections."""
        self.multipath_active = active
        if active:
            self.multipath_bias = self.rng.normal(0, bias_magnitude, 3)
            self.multipath_bias[2] *= 0.5  # Vertical multipath usually smaller
        else:
            self.multipath_bias = np.zeros(3)

    def step(
        self,
        timestamp: float,
        true_position_ned: np.ndarray,
        true_velocity_ned: np.ndarray
    ) -> GNSSReading:
        if self.is_jammed:
            # Total GNSS denial
            return GNSSReading(
                timestamp=timestamp,
                position=np.zeros(3),
                velocity=np.zeros(3),
                fix_type=GNSSFixType.NO_FIX,
                satellites_in_view=0,
                hdop=99.9,
                vdop=99.9,
                covariance=np.eye(3) * 1e6,
                is_valid=False
            )

        # Dynamic satellite geometry & DOP
        satellites = max(4, int(self.nominal_satellites + self.rng.randint(-2, 3)))
        hdop = 1.0 + (14 - min(satellites, 14)) * 0.25
        vdop = hdop * 1.6

        # Standard deviations scaled by DOP
        h_std = self.base_horizontal_std * hdop
        v_std = self.base_vertical_std * vdop
        vel_std = self.base_velocity_std * hdop

        # Add noise + multipath
        pos_noise = np.array([
            self.rng.normal(0, h_std),
            self.rng.normal(0, h_std),
            self.rng.normal(0, v_std)
        ])
        vel_noise = self.rng.normal(0, vel_std, 3)

        if self.multipath_active:
            # Random walk on multipath bias
            self.multipath_bias += self.rng.normal(0, 0.5, 3)
            pos_noise += self.multipath_bias
            h_std *= 3.0
            v_std *= 3.0

        meas_pos = true_position_ned + pos_noise
        meas_vel = true_velocity_ned + vel_noise

        cov = np.diag([h_std**2, h_std**2, v_std**2])

        return GNSSReading(
            timestamp=timestamp,
            position=meas_pos,
            velocity=meas_vel,
            fix_type=GNSSFixType.FIX_3D,
            satellites_in_view=satellites,
            hdop=hdop,
            vdop=vdop,
            covariance=cov,
            is_valid=True
        )

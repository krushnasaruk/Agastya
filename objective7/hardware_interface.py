"""
Hardware Abstraction Layer and Sensor Source Interface for Objective 7.
Decouples navigation processing from physical hardware, replay streams, and software HIL emulators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterator
import numpy as np
import pandas as pd


@dataclass
class RawSensorSample:
    sample_index: int
    timestamp_sec: float
    dt_sec: float
    wheel_fl_ms: float
    wheel_fr_ms: float
    wheel_rl_ms: float
    wheel_rr_ms: float
    accel_x_ms2: float
    yaw_rate_rads: float


class SensorSource(ABC):
    """
    Abstract interface for all hardware, simulation, and replay sensor sources.
    """
    @abstractmethod
    def read_sample(self) -> Optional[RawSensorSample]:
        """Fetch next sensor frame from source."""
        pass

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get latest sample timestamp."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Verify source connectivity and health."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying hardware or file streams."""
        pass


class ReplaySensorSource(SensorSource):
    """
    Streams pre-recorded flight/drive trajectories from DataFrame at nominal frequency.
    """
    def __init__(self, nav_df: pd.DataFrame):
        self.nav_df = nav_df.reset_index(drop=True)
        self.total_samples = len(self.nav_df)
        self.current_idx = 0
        self.latest_timestamp = 0.0

    def read_sample(self) -> Optional[RawSensorSample]:
        if self.current_idx >= self.total_samples:
            return None

        row = self.nav_df.iloc[self.current_idx]
        self.current_idx += 1

        t = float(row.get("time_sec", 0.0))
        dt = float(row.get("dt_sec", 0.1))
        self.latest_timestamp = t

        v_fl = float(row.get("wheel_speed_fl_ms", 0.0))
        v_fr = float(row.get("wheel_speed_fr_ms", 0.0))
        v_rl = float(row.get("wheel_speed_rl_ms", 0.0))
        v_rr = float(row.get("wheel_speed_rr_ms", 0.0))
        ax = float(row.get("accel_x_ms2", 0.0))
        yr = float(row.get("yaw_rate_rads", 0.0))

        return RawSensorSample(
            sample_index=self.current_idx - 1,
            timestamp_sec=t,
            dt_sec=dt,
            wheel_fl_ms=v_fl,
            wheel_fr_ms=v_fr,
            wheel_rl_ms=v_rl,
            wheel_rr_ms=v_rr,
            accel_x_ms2=ax,
            yaw_rate_rads=yr
        )

    def get_timestamp(self) -> float:
        return self.latest_timestamp

    def validate(self) -> bool:
        return self.total_samples > 0

    def close(self) -> None:
        self.current_idx = self.total_samples


class HardwareSensorSource(SensorSource):
    """
    Software-emulated Hardware-in-the-Loop (HIL) interface.
    Explicitly labeled SOFTWARE-HIL for scientific transparency.
    """
    def __init__(self, replay_source: ReplaySensorSource):
        self.source = replay_source
        self.hardware_label = "SOFTWARE-HIL (Emulated Hardware Stream)"

    def read_sample(self) -> Optional[RawSensorSample]:
        return self.source.read_sample()

    def get_timestamp(self) -> float:
        return self.source.get_timestamp()

    def validate(self) -> bool:
        return self.source.validate()

    def close(self) -> None:
        self.source.close()

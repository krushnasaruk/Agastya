"""
Sequence Replay and Pipeline Integration Runner for Objective 7.
Replays test sequences through the RealtimeNavigationEngine and benchmarks complete navigation accuracy.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from navigation_engine.state import DeadReckoningTrajectory
from navigation_engine.evaluation import DeadReckoningEvaluator, NavigationMetrics
from .realtime_engine import RealtimeNavigationEngine
from .hardware_interface import ReplaySensorSource, RawSensorSample


@dataclass
class ReplayResult:
    trajectory: DeadReckoningTrajectory
    telemetry_df: pd.DataFrame
    metrics: NavigationMetrics
    latency_summary: Dict[str, Any]
    watchdog_summary: Dict[str, Any]
    total_samples: int
    application_rate_pct: float
    fallback_rate_pct: float


class RealtimeReplayEngine:
    """
    Feeds DataFrame sequences through the RealtimeNavigationEngine.
    """
    @classmethod
    def run_replay(
        cls,
        engine: RealtimeNavigationEngine,
        nav_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0
    ) -> ReplayResult:
        source = ReplaySensorSource(nav_df)
        engine.initialize(
            initial_p_east_m=initial_p_east_m,
            initial_p_north_m=initial_p_north_m,
            initial_heading_rad=initial_heading_rad,
            initial_time_sec=float(nav_df["time_sec"].iloc[0]) if not nav_df.empty else 0.0
        )

        while True:
            sample: Optional[RawSensorSample] = source.read_sample()
            if sample is None:
                break

            engine.process_sensor_sample(
                timestamp_sec=sample.timestamp_sec,
                dt_sec=sample.dt_sec,
                wheel_fl=sample.wheel_fl_ms,
                wheel_fr=sample.wheel_fr_ms,
                wheel_rl=sample.wheel_rl_ms,
                wheel_rr=sample.wheel_rr_ms,
                accel_x=sample.accel_x_ms2,
                yaw_rate=sample.yaw_rate_rads
            )

        traj = engine.get_trajectory()
        telem_df = engine.get_telemetry()
        lat_sum = engine.latency_monitor.get_summary_statistics()
        wd_sum = engine.watchdog.get_summary()

        ref_e = ref_df["pos_east_m"].to_numpy()
        ref_n = ref_df["pos_north_m"].to_numpy()
        ref_h = ref_df.get("heading_rad", None)
        ref_v = ref_df.get("ground_speed_ms", None)

        metrics, _, _ = DeadReckoningEvaluator.evaluate(traj, ref_e, ref_n, ref_h, ref_v)

        app_count = int(np.sum(telem_df["ai_applied"])) if not telem_df.empty else 0
        total_s = len(telem_df)
        app_rate = (app_count / max(total_s, 1)) * 100.0

        return ReplayResult(
            trajectory=traj,
            telemetry_df=telem_df,
            metrics=metrics,
            latency_summary=lat_sum,
            watchdog_summary=wd_sum,
            total_samples=total_s,
            application_rate_pct=round(app_rate, 2),
            fallback_rate_pct=round(100.0 - app_rate, 2)
        )

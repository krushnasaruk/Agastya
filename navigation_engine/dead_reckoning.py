"""
Classical Dead-Reckoning Navigation Engine for Project AGASTYA (Objective 3).

COORDINATE & HEADING CONVENTION:
  - Coordinate System: Local East-North-Up (ENU) Metric Tangent Plane
  - Easting (p_east_m): Meters along East axis (+E positive East)
  - Northing (p_north_m): Meters along North axis (+N positive North)
  - Heading (psi): Angle in radians measured CLOCKWISE from True North:
      * psi = 0 rad (0 deg)     -> True North
      * psi = pi/2 rad (90 deg)  -> East
      * psi = pi rad (180 deg)   -> South
      * psi = 3pi/2 rad (270 deg)-> West
  - Positive Yaw Rate (psi_dot > 0): Turning CLOCKWISE (Right turn)
  - Planar Displacement Equations:
      * dE = v * sin(psi_mid) * dt
      * dN = v * cos(psi_mid) * dt

BASELINES REGISTRY:
  - Baseline A: Differential Wheel Odometry + CAN Yaw Rate (Primary Classical Benchmark)
  - Baseline B: Kinematic Wheel Odometry + CAN Yaw Rate + Longitudinal Accel Fusion
  - Baseline C: CAN Inertial-Only Planar Integration Baseline (Diagnostic Baseline)
  - Smartphone SINS: BLOCKED — SENSOR FRAME CALIBRATION REQUIRED
"""

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .state import PlanarNavigationState, DeadReckoningTrajectory, wrap_to_2pi
from .wheel_odometry import WheelOdometryEstimator
from .yaw import YawPropagator
from .quality_gate import CausalQualityGate


@dataclass
class VehicleParameter:
    name: str
    value: float
    unit: str
    status: str
    provenance: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineMetadata:
    baseline_id: str
    name: str
    sensor_inputs: List[str]
    equations: str
    causal_status: str
    purpose: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassicalDeadReckoningConfig:
    """
    Configuration parameters with explicit physical provenance and baseline registry.
    """
    baseline_type: str = "BASELINE_A"
    track_width_m: float = 1.47               # [PROVISIONAL / CONFIG REQUIRED: Ford Fiesta Mk7 OEM Rear Track]
    gyro_bias_rads: float = 0.0               # [CONFIG: Initial CAN gyro bias offset]
    zero_speed_threshold_ms: float = 0.08     # [PROVISIONAL: ~0.28 km/h stationary threshold for ZUPT]
    accel_weight_baseline_b: float = 0.15     # [PROVISIONAL: Weight on integrated accel vs wheel speed in Baseline B]
    slip_diff_threshold_ms: float = 2.5       # [PROVISIONAL: Discrepancy threshold for wheel slip]
    max_plausible_speed_ms: float = 70.0      # [PROVISIONAL: Max passenger vehicle speed]

    @classmethod
    def from_json(cls, config_path: str, baseline_type: Optional[str] = None) -> "ClassicalDeadReckoningConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            return cls(baseline_type=baseline_type or "BASELINE_A")
        with open(config_path, "r") as f:
            data = json.load(f)
        params = data.get("parameters", {})
        return cls(
            baseline_type=baseline_type or "BASELINE_A",
            track_width_m=float(params.get("track_width_m", {}).get("value", 1.47)),
            zero_speed_threshold_ms=float(params.get("zero_speed_threshold_ms", {}).get("value", 0.08)),
            accel_weight_baseline_b=float(params.get("accel_weight_baseline_b", {}).get("value", 0.15)),
            slip_diff_threshold_ms=float(params.get("slip_diff_threshold_ms", {}).get("value", 2.5)),
            max_plausible_speed_ms=float(params.get("max_plausible_speed_ms", {}).get("value", 70.0))
        )

    def get_parameter_registry(self) -> Dict[str, VehicleParameter]:
        return {
            "track_width_m": VehicleParameter(
                name="track_width_m",
                value=self.track_width_m,
                unit="meters",
                status="PROVISIONAL / CONFIGURATION REQUIRED",
                provenance="Ford Fiesta Mk7 OEM specification (not present in raw CSV headers)",
                rationale="Required for differential wheel yaw rate calculation during gyro dropout."
            ),
            "zero_speed_threshold_ms": VehicleParameter(
                name="zero_speed_threshold_ms",
                value=self.zero_speed_threshold_ms,
                unit="m/s",
                status="PROVISIONAL",
                provenance="Empirical stationary sensor noise floor threshold",
                rationale="Velocity below ~0.28 km/h indicates stopped vehicle; activates ZUPT."
            ),
            "accel_weight_baseline_b": VehicleParameter(
                name="accel_weight_baseline_b",
                value=self.accel_weight_baseline_b,
                unit="dimensionless ratio",
                status="PROVISIONAL",
                provenance="Kinematic complementary filter tuning",
                rationale="Weight on integrated longitudinal acceleration vs wheel speed in Baseline B."
            )
        }

    @staticmethod
    def get_baseline_registry() -> Dict[str, BaselineMetadata]:
        return {
            "BASELINE_A": BaselineMetadata(
                baseline_id="BASELINE_A",
                name="Differential Wheel Odometry + CAN Yaw Rate",
                sensor_inputs=["wheel_speed_rl_ms", "wheel_speed_rr_ms", "yaw_rate_rads", "dt_sec"],
                equations="v_fwd = (v_RL + v_RR)/2; psi_k = psi_{k-1} + omega_z*dt; planar ENU integration",
                causal_status="STRICTLY CAUSAL",
                purpose="Primary classical ground-vehicle dead-reckoning benchmark"
            ),
            "BASELINE_B": BaselineMetadata(
                baseline_id="BASELINE_B",
                name="Kinematic Wheel Odometry + CAN Yaw Rate + Longitudinal Acceleration Fusion",
                sensor_inputs=["wheel_speed_rl_ms", "wheel_speed_rr_ms", "accel_x_ms2", "yaw_rate_rads", "dt_sec"],
                equations="v_filt = (1-w)*v_wheel + w*(v_filt + a_x*dt); psi_k = psi_{k-1} + omega_z*dt; planar ENU",
                causal_status="STRICTLY CAUSAL",
                purpose="Slip-resilient kinematic dead-reckoning baseline with acceleration complementary fusion"
            ),
            "BASELINE_C": BaselineMetadata(
                baseline_id="BASELINE_C",
                name="CAN Inertial-Only Planar Integration Baseline",
                sensor_inputs=["accel_x_ms2", "yaw_rate_rads", "dt_sec"],
                equations="v_fwd = v_{k-1} + a_x*dt; psi_k = psi_{k-1} + omega_z*dt; planar ENU integration",
                causal_status="STRICTLY CAUSAL",
                purpose="Diagnostic baseline demonstrating rapid unbounded quadratic drift without wheel odometry feedback"
            ),
            "SMARTPHONE_SINS": BaselineMetadata(
                baseline_id="SMARTPHONE_SINS",
                name="Smartphone Cabin 3D SINS Mechanization",
                sensor_inputs=["phone_acc_x", "phone_acc_y", "phone_acc_z", "phone_gyro_x", "phone_gyro_y", "phone_gyro_z"],
                equations="3D Strapdown integration with gravity compensation",
                causal_status="BLOCKED — SENSOR FRAME CALIBRATION REQUIRED",
                purpose="Blocked until extrinsic mounting matrix T_phone^vehicle is calibrated in future objective"
            )
        }


class SmartphoneCalibrationGuard:
    """
    Guard preventing uncalibrated smartphone IMU data from entering the vehicle navigation engine.
    """
    @staticmethod
    def assert_calibrated(is_calibrated: bool = False, mounting_matrix: Optional[np.ndarray] = None) -> None:
        if not is_calibrated or mounting_matrix is None:
            raise RuntimeError(
                "SMARTPHONE IMU BLOCKED: Smartphone-to-vehicle extrinsic mounting matrix T_phone^vehicle is UNKNOWN. "
                "Uncalibrated smartphone IMU integration is blocked to prevent invalid navigation states."
            )


class ClassicalDeadReckoningEngine:
    """
    Production-grade Classical Dead-Reckoning Engine operating strictly on causal onboard sensors.
    """
    def __init__(
        self,
        baseline_type: str = "BASELINE_A",
        config: Optional[ClassicalDeadReckoningConfig] = None,
        track_width_m: Optional[float] = None,
        gyro_bias_rads: float = 0.0,
        zero_speed_threshold_ms: float = 0.08
    ):
        if config is not None:
            self.cfg = config
        else:
            t_width = track_width_m if track_width_m is not None else 1.47
            self.cfg = ClassicalDeadReckoningConfig(
                baseline_type=baseline_type.upper(),
                track_width_m=t_width,
                gyro_bias_rads=gyro_bias_rads,
                zero_speed_threshold_ms=zero_speed_threshold_ms
            )

        self.baseline_type = self.cfg.baseline_type
        if self.baseline_type == "SMARTPHONE_SINS":
            SmartphoneCalibrationGuard.assert_calibrated(is_calibrated=False)

        self.wheel_estimator = WheelOdometryEstimator(
            track_width_m=self.cfg.track_width_m,
            zero_speed_threshold_ms=self.cfg.zero_speed_threshold_ms,
            max_plausible_speed_ms=self.cfg.max_plausible_speed_ms,
            slip_diff_threshold_ms=self.cfg.slip_diff_threshold_ms
        )
        self.yaw_propagator = YawPropagator(gyro_bias_rads=self.cfg.gyro_bias_rads)
        self.quality_gate = CausalQualityGate()

        # State containers
        self.state = PlanarNavigationState()
        self.is_initialized = False

        # Baseline B & C Filter State
        self._v_filtered = 0.0

    def initialize(
        self,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0,
        initial_heading_rad: float = 0.0,
        initial_time_sec: float = 0.0
    ) -> None:
        """
        Explicit offline state initialization (OFFLINE INITIALIZATION PROTOCOL).
        Initial position and heading represent the known launch pose before entering dead reckoning.
        """
        self.state = PlanarNavigationState(
            time_sec=initial_time_sec,
            p_east_m=initial_p_east_m,
            p_north_m=initial_p_north_m,
            heading_rad=wrap_to_2pi(initial_heading_rad),
            forward_speed_ms=0.0,
            yaw_rate_rads=0.0,
            is_stationary=True,
            quality_status="INITIALIZED"
        )
        self.yaw_propagator.reset(initial_heading_rad)
        self._v_filtered = 0.0
        self.is_initialized = True

    def step(
        self,
        time_sec: float,
        dt_sec: float,
        wheel_speed_fl: Optional[float] = None,
        wheel_speed_fr: Optional[float] = None,
        wheel_speed_rl: Optional[float] = None,
        wheel_speed_rr: Optional[float] = None,
        accel_x: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        is_valid_mask: bool = True
    ) -> PlanarNavigationState:
        """
        Step dead reckoning forward by dt_sec using active causal baseline.
        """
        if not self.is_initialized:
            self.initialize(initial_time_sec=time_sec)

        # 1. Sanitize Sensor Inputs
        clean = self.quality_gate.sanitize_epoch(
            time_sec=time_sec,
            dt_sec=dt_sec,
            v_fl=wheel_speed_fl,
            v_fr=wheel_speed_fr,
            v_rl=wheel_speed_rl,
            v_rr=wheel_speed_rr,
            accel_x=accel_x,
            yaw_rate=yaw_rate,
            mask_valid=is_valid_mask
        )

        dt = clean.dt_sec

        # 2. Estimate Wheel Odometry Speed & Kinematic Yaw Rate
        wheel_est = self.wheel_estimator.estimate_speed(
            v_fl=clean.wheel_speed_fl_ms,
            v_fr=clean.wheel_speed_fr_ms,
            v_rl=clean.wheel_speed_rl_ms,
            v_rr=clean.wheel_speed_rr_ms
        )

        # 3. Baseline-Specific Forward Speed Determination
        if self.baseline_type == "BASELINE_A":
            # Baseline A: Pure Differential Wheel Odometry Speed
            forward_speed = wheel_est.forward_speed_ms
            is_stat = wheel_est.is_stationary

        elif self.baseline_type == "BASELINE_B":
            # Baseline B: Kinematic Wheel Odometry + Longitudinal Acceleration Fusion
            v_raw_wheel = wheel_est.forward_speed_ms
            a_meas = clean.accel_x_ms2 if clean.accel_x_ms2 is not None else 0.0

            if wheel_est.is_stationary:
                self._v_filtered = 0.0
                is_stat = True
            else:
                # Complementary fusion
                v_integrated = self._v_filtered + a_meas * dt
                w_acc = self.cfg.accel_weight_baseline_b
                if wheel_est.slip_detected:
                    w_acc = 0.65  # Rely more on acceleration during detected slip
                self._v_filtered = (1.0 - w_acc) * v_raw_wheel + w_acc * max(0.0, v_integrated)
                is_stat = False

            forward_speed = self._v_filtered

        elif self.baseline_type == "BASELINE_C":
            # Baseline C: CAN Inertial-Only Planar Integration (Longitudinal Acceleration Integration)
            a_meas = clean.accel_x_ms2 if clean.accel_x_ms2 is not None else 0.0
            self._v_filtered = max(0.0, self._v_filtered + a_meas * dt)
            forward_speed = self._v_filtered
            is_stat = forward_speed < self.cfg.zero_speed_threshold_ms
        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

        # 4. Heading Propagation (Trapezoidal / Midpoint Integration)
        prev_heading = self.state.heading_rad
        next_heading, effective_yaw, yaw_source = self.yaw_propagator.step(
            yaw_rate_can_rads=clean.yaw_rate_rads,
            dt_sec=dt,
            is_stationary=is_stat,
            kinematic_yaw_fallback_rads=wheel_est.kinematic_yaw_rate_rads
        )

        # Midpoint heading: psi_mid = psi_prev + 0.5 * delta_psi
        delta_psi = effective_yaw * dt
        psi_mid = prev_heading + 0.5 * delta_psi

        # 5. Position Propagation in Metric Local ENU
        # Heading 0 = North, pi/2 = East
        # d_East = v * sin(psi_mid) * dt, d_North = v * cos(psi_mid) * dt
        d_east = forward_speed * np.sin(psi_mid) * dt
        d_north = forward_speed * np.cos(psi_mid) * dt

        # Update State
        self.state.time_sec = time_sec
        self.state.p_east_m += d_east
        self.state.p_north_m += d_north
        self.state.heading_rad = next_heading
        self.state.forward_speed_ms = forward_speed
        self.state.yaw_rate_rads = effective_yaw
        self.state.accel_longitudinal_ms2 = clean.accel_x_ms2 if clean.accel_x_ms2 is not None else 0.0
        self.state.is_stationary = is_stat
        self.state.quality_status = clean.quality_status

        return self.state.clone()

    def update(self, *args, **kwargs) -> PlanarNavigationState:
        """Alias for step() method."""
        return self.step(*args, **kwargs)

    def run_sequence(
        self,
        navigation_inputs_df: pd.DataFrame,
        initial_heading_rad: float = 0.0,
        initial_p_east_m: float = 0.0,
        initial_p_north_m: float = 0.0
    ) -> DeadReckoningTrajectory:
        """
        Process a full sequence DataFrame causally and return the dead-reckoning trajectory.
        """
        n = len(navigation_inputs_df)
        t_arr = navigation_inputs_df["time_sec"].to_numpy()
        dt_arr = navigation_inputs_df["dt_sec"].to_numpy()

        p_east = np.empty(n, dtype=np.float64)
        p_north = np.empty(n, dtype=np.float64)
        headings = np.empty(n, dtype=np.float64)
        speeds = np.empty(n, dtype=np.float64)
        yaw_rates = np.empty(n, dtype=np.float64)

        self.initialize(
            initial_p_east_m=initial_p_east_m,
            initial_p_north_m=initial_p_north_m,
            initial_heading_rad=initial_heading_rad,
            initial_time_sec=t_arr[0] if n > 0 else 0.0
        )

        v_fl = navigation_inputs_df.get("wheel_speed_fl_ms", pd.Series([None] * n)).to_numpy()
        v_fr = navigation_inputs_df.get("wheel_speed_fr_ms", pd.Series([None] * n)).to_numpy()
        v_rl = navigation_inputs_df.get("wheel_speed_rl_ms", pd.Series([None] * n)).to_numpy()
        v_rr = navigation_inputs_df.get("wheel_speed_rr_ms", pd.Series([None] * n)).to_numpy()
        accel_x = navigation_inputs_df.get("accel_x_ms2", pd.Series([None] * n)).to_numpy()
        yaw_rate = navigation_inputs_df.get("yaw_rate_rads", pd.Series([None] * n)).to_numpy()

        for k in range(n):
            st = self.step(
                time_sec=t_arr[k],
                dt_sec=dt_arr[k],
                wheel_speed_fl=v_fl[k],
                wheel_speed_fr=v_fr[k],
                wheel_speed_rl=v_rl[k],
                wheel_speed_rr=v_rr[k],
                accel_x=accel_x[k],
                yaw_rate=yaw_rate[k]
            )
            p_east[k] = st.p_east_m
            p_north[k] = st.p_north_m
            headings[k] = st.heading_rad
            speeds[k] = st.forward_speed_ms
            yaw_rates[k] = st.yaw_rate_rads

        # Cumulative distance
        if n >= 2:
            step_dists = np.sqrt(np.diff(p_east)**2 + np.diff(p_north)**2)
            total_dist = float(np.sum(step_dists))
        else:
            total_dist = 0.0

        return DeadReckoningTrajectory(
            timestamps_sec=t_arr,
            dt_array_sec=dt_arr,
            p_east_m=p_east,
            p_north_m=p_north,
            heading_rad=headings,
            forward_speed_ms=speeds,
            yaw_rate_rads=yaw_rates,
            baseline_name=self.baseline_type,
            total_distance_m=total_dist
        )

"""
Unit Normalization Engine for IO-VNBD Data Engineering.
Converts raw dataset signals from imperial/automotive units (km/h, deg, ms)
into strictly standard SI units (m/s, rad, rad/s, s) with explicit audit logs.
"""

from typing import Dict, Any, Tuple
import numpy as np


class UnitNormalizer:
    """
    Deterministically transforms verified raw IO-VNBD units to SI navigation standards.
    """
    KMH_TO_MS = 1.0 / 3.6
    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEG = 180.0 / np.pi
    MS_TO_SEC = 1e-3

    @classmethod
    def kmh_to_ms(cls, val: np.ndarray) -> np.ndarray:
        """Convert speed from km/h to m/s."""
        return np.asarray(val, dtype=np.float64) * cls.KMH_TO_MS

    @classmethod
    def deg_to_rad(cls, val: np.ndarray) -> np.ndarray:
        """Convert angle/rate from degrees to radians."""
        return np.asarray(val, dtype=np.float64) * cls.DEG_TO_RAD

    @classmethod
    def ms_to_sec(cls, val: np.ndarray) -> np.ndarray:
        """Convert milliseconds to seconds."""
        return np.asarray(val, dtype=np.float64) * cls.MS_TO_SEC

    @classmethod
    def wrap_to_pi(cls, angles_rad: np.ndarray) -> np.ndarray:
        """Wrap angles in radians to [-pi, pi]."""
        return (np.asarray(angles_rad, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def wrap_to_2pi(cls, angles_rad: np.ndarray) -> np.ndarray:
        """Wrap angles in radians to [0, 2*pi)."""
        return np.asarray(angles_rad, dtype=np.float64) % (2.0 * np.pi)

    @classmethod
    def normalize_vehicle_dataframe(cls, df_raw) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        Normalize vehicle CAN & VBOX DataFrame into SI numpy arrays.
        """
        normalized: Dict[str, np.ndarray] = {}
        provenance: Dict[str, str] = {}

        # Timestamps
        if "Time" in df_raw:
            normalized["time_sec"] = cls.ms_to_sec(df_raw["Time"].to_numpy())
            provenance["time_sec"] = "Time(ms) * 1e-3 -> seconds"

        # 4-Wheel Speeds
        for col, target in [
            ("Wheel speed FL", "wheel_speed_fl_ms"),
            ("Wheel speed FR", "wheel_speed_fr_ms"),
            ("Wheel speed RL", "wheel_speed_rl_ms"),
            ("Wheel speed RR", "wheel_speed_rr_ms"),
        ]:
            if col in df_raw:
                normalized[target] = cls.kmh_to_ms(df_raw[col].to_numpy())
                provenance[target] = f"{col}(km/h) / 3.6 -> m/s"

        # Longitudinal Acceleration (already m/s^2)
        if "Longitudinal acceleration" in df_raw:
            normalized["accel_x_ms2"] = df_raw["Longitudinal acceleration"].to_numpy(dtype=np.float64)
            provenance["accel_x_ms2"] = "Longitudinal acceleration(m/s^2) -> m/s^2"

        # Yaw Rate
        if "Yaw rate" in df_raw:
            normalized["yaw_rate_rads"] = cls.deg_to_rad(df_raw["Yaw rate"].to_numpy())
            provenance["yaw_rate_rads"] = "Yaw rate(deg/s) * (pi / 180) -> rad/s"

        # VBOX GPS
        if "GPS latitude" in df_raw:
            normalized["latitude_deg"] = df_raw["GPS latitude"].to_numpy(dtype=np.float64)
            provenance["latitude_deg"] = "GPS latitude(deg) [WGS-84]"

        if "GPS longitude" in df_raw:
            normalized["longitude_deg"] = df_raw["GPS longitude"].to_numpy(dtype=np.float64)
            provenance["longitude_deg"] = "GPS longitude(deg) [WGS-84]"

        if "GPS altitude" in df_raw:
            normalized["altitude_m"] = df_raw["GPS altitude"].to_numpy(dtype=np.float64)
            provenance["altitude_m"] = "GPS altitude(m) [MSL]"

        if "GPS speed" in df_raw:
            normalized["gps_speed_ms"] = cls.kmh_to_ms(df_raw["GPS speed"].to_numpy())
            provenance["gps_speed_ms"] = "GPS speed(km/h) / 3.6 -> m/s"

        if "GPS orientation" in df_raw:
            raw_heading_deg = df_raw["GPS orientation"].to_numpy(dtype=np.float64)
            normalized["heading_rad"] = cls.deg_to_rad(raw_heading_deg)
            provenance["heading_rad"] = "GPS orientation(deg) * (pi / 180) -> radians [0, 2*pi)"

        if "GPS accuracy" in df_raw:
            normalized["gps_accuracy_m"] = df_raw["GPS accuracy"].to_numpy(dtype=np.float64)
            provenance["gps_accuracy_m"] = "GPS accuracy(m)"

        if "GPS satellites" in df_raw:
            normalized["satellites_count"] = df_raw["GPS satellites"].to_numpy(dtype=np.int32)
            provenance["satellites_count"] = "GPS satellites count"

        return normalized, provenance

    @classmethod
    def normalize_smartphone_dataframe(cls, df_raw) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        Normalize smartphone sensor DataFrame into SI numpy arrays.
        """
        normalized: Dict[str, np.ndarray] = {}
        provenance: Dict[str, str] = {}

        if "Time" in df_raw:
            normalized["phone_time_sec"] = cls.ms_to_sec(df_raw["Time"].to_numpy())
            provenance["phone_time_sec"] = "Time(ms) * 1e-3 -> seconds"

        # Accelerometer (m/s^2)
        for ax in ["X", "Y", "Z"]:
            col = f"Acc_{ax}"
            if col in df_raw:
                target = f"phone_acc_{ax.lower()}_ms2"
                normalized[target] = df_raw[col].to_numpy(dtype=np.float64)
                provenance[target] = f"{col}(m/s^2) [Android Frame]"

        # Gyroscope (rad/s)
        for ax in ["X", "Y", "Z"]:
            col = f"Gyro_{ax}"
            if col in df_raw:
                target = f"phone_gyro_{ax.lower()}_rads"
                normalized[target] = df_raw[col].to_numpy(dtype=np.float64)
                provenance[target] = f"{col}(rad/s) [Android Frame]"

        # Magnetometer (uT)
        for ax in ["X", "Y", "Z"]:
            col = f"Mag_{ax}"
            if col in df_raw:
                target = f"phone_mag_{ax.lower()}_uT"
                normalized[target] = df_raw[col].to_numpy(dtype=np.float64)
                provenance[target] = f"{col}(uT) [Android Frame]"

        # Phone GPS
        if "GPS_Lat" in df_raw:
            normalized["phone_lat_deg"] = df_raw["GPS_Lat"].to_numpy(dtype=np.float64)
        if "GPS_Long" in df_raw:
            normalized["phone_long_deg"] = df_raw["GPS_Long"].to_numpy(dtype=np.float64)
        if "GPS_Speed" in df_raw:
            normalized["phone_gps_speed_ms"] = df_raw["GPS_Speed"].to_numpy(dtype=np.float64)
        if "GPS_Bearing" in df_raw:
            normalized["phone_bearing_rad"] = cls.deg_to_rad(df_raw["GPS_Bearing"].to_numpy())

        return normalized, provenance

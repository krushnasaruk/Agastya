"""
IO-VNBD Schema Registry & Signal Specifications for Project AGASTYA.
Formalizes verified dataset signals across Vehicle CAN, VBOX Reference, and
Android Smartphone MEMS streams, documenting units, coordinate frames,
sampling behavior, stream availability, and verification status.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"     # Confirmed from official IO-VNBD repo & publications
    DERIVED = "DERIVED"       # Computed by data pipeline from raw signals
    UNKNOWN = "UNKNOWN"       # Parameter not explicitly specified in official source


class SignalSource(str, Enum):
    VEHICLE_CAN = "vehicle_can"
    VBOX_REFERENCE = "vbox_reference"
    SMARTPHONE_MEMS = "smartphone_mems"
    SMARTPHONE_GPS = "smartphone_gps"
    DERIVED_STATE = "derived_state"


class CoordinateFrame(str, Enum):
    WGS84 = "wgs84"
    LOCAL_ENU = "local_enu"
    LOCAL_NED = "local_ned"
    VEHICLE_BODY = "vehicle_body"
    PHONE_BODY = "phone_body"
    WHEEL_HUB = "wheel_hub"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalSpec:
    raw_name: str
    standard_name: str
    physical_meaning: str
    raw_unit: str
    standard_unit: str
    source: SignalSource
    frame: CoordinateFrame
    nominal_rate_hz: float
    available_in_v_dataset: bool
    available_in_s_dataset: bool
    available_in_sync_dataset: bool
    is_ground_truth: bool = False
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "standard_name": self.standard_name,
            "physical_meaning": self.physical_meaning,
            "raw_unit": self.raw_unit,
            "standard_unit": self.standard_unit,
            "source": self.source.value,
            "frame": self.frame.value,
            "nominal_rate_hz": self.nominal_rate_hz,
            "available_in_v_dataset": self.available_in_v_dataset,
            "available_in_s_dataset": self.available_in_s_dataset,
            "available_in_sync_dataset": self.available_in_sync_dataset,
            "is_ground_truth": self.is_ground_truth,
            "verification_status": self.verification_status.value,
            "description": self.description
        }


class IOVNBDSchemaRegistry:
    """
    Central verified schema registry for the IO-VNBD dataset.
    """
    # --------------------------------------------------------------------------
    # Vehicle CAN & VBOX Reference Signals (V_dataset_*.csv)
    # --------------------------------------------------------------------------
    TIME_VEHICLE = SignalSpec(
        raw_name="Time",
        standard_name="timestamp_vehicle_ms",
        physical_meaning="Elapsed time since vehicle recording start",
        raw_unit="ms",
        standard_unit="s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.TEMPORAL,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=None,
        verification_status=VerificationStatus.VERIFIED,
        description="Relative millisecond timestamp from VBOX/CAN logger"
    )

    WHEEL_SPEED_FL = SignalSpec(
        raw_name="Wheel speed FL",
        standard_name="wheel_speed_fl_ms",
        physical_meaning="Front-Left wheel linear speed",
        raw_unit="km/h",
        standard_unit="m/s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.WHEEL_HUB,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=220.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Rotational wheel speed converted to linear velocity by ECU"
    )

    WHEEL_SPEED_FR = SignalSpec(
        raw_name="Wheel speed FR",
        standard_name="wheel_speed_fr_ms",
        physical_meaning="Front-Right wheel linear speed",
        raw_unit="km/h",
        standard_unit="m/s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.WHEEL_HUB,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=220.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Rotational wheel speed converted to linear velocity by ECU"
    )

    WHEEL_SPEED_RL = SignalSpec(
        raw_name="Wheel speed RL",
        standard_name="wheel_speed_rl_ms",
        physical_meaning="Rear-Left unsteered wheel linear speed",
        raw_unit="km/h",
        standard_unit="m/s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.WHEEL_HUB,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=220.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Rear unsteered wheel speed used for differential odometry"
    )

    WHEEL_SPEED_RR = SignalSpec(
        raw_name="Wheel speed RR",
        standard_name="wheel_speed_rr_ms",
        physical_meaning="Rear-Right unsteered wheel linear speed",
        raw_unit="km/h",
        standard_unit="m/s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.WHEEL_HUB,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=220.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Rear unsteered wheel speed used for differential odometry"
    )

    LONGITUDINAL_ACCEL = SignalSpec(
        raw_name="Longitudinal acceleration",
        standard_name="accel_x_ms2",
        physical_meaning="Vehicle longitudinal body acceleration (+X forward)",
        raw_unit="m/s^2",
        standard_unit="m/s^2",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.VEHICLE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-15.0,
        expected_max=15.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Vehicle chassis longitudinal acceleration from CAN bus"
    )

    YAW_RATE = SignalSpec(
        raw_name="Yaw rate",
        standard_name="yaw_rate_rads",
        physical_meaning="Vehicle yaw angular velocity around vertical axis",
        raw_unit="deg/s",
        standard_unit="rad/s",
        source=SignalSource.VEHICLE_CAN,
        frame=CoordinateFrame.VEHICLE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-120.0,
        expected_max=120.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Chassis yaw rate gyro measurement from CAN bus"
    )

    # --------------------------------------------------------------------------
    # VBOX Reference Ground-Truth Signals (STRICTLY ISOLATED FROM INPUTS)
    # --------------------------------------------------------------------------
    GPS_LATITUDE = SignalSpec(
        raw_name="GPS latitude",
        standard_name="latitude_deg",
        physical_meaning="WGS-84 Geographic Latitude",
        raw_unit="deg",
        standard_unit="deg",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.WGS84,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=-90.0,
        expected_max=90.0,
        verification_status=VerificationStatus.VERIFIED,
        description="High-accuracy VBOX reference GPS latitude (Evaluation Only)"
    )

    GPS_LONGITUDE = SignalSpec(
        raw_name="GPS longitude",
        standard_name="longitude_deg",
        physical_meaning="WGS-84 Geographic Longitude",
        raw_unit="deg",
        standard_unit="deg",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.WGS84,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=-180.0,
        expected_max=180.0,
        verification_status=VerificationStatus.VERIFIED,
        description="High-accuracy VBOX reference GPS longitude (Evaluation Only)"
    )

    GPS_ALTITUDE = SignalSpec(
        raw_name="GPS altitude",
        standard_name="altitude_m",
        physical_meaning="WGS-84 Ellipsoidal Height / Altitude above MSL",
        raw_unit="m",
        standard_unit="m",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.WGS84,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=-200.0,
        expected_max=9000.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Height above mean sea level reported by VBOX (Evaluation Only)"
    )

    GPS_SPEED = SignalSpec(
        raw_name="GPS speed",
        standard_name="gps_speed_ms",
        physical_meaning="True 2D ground speed from Doppler GNSS",
        raw_unit="km/h",
        standard_unit="m/s",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.LOCAL_ENU,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=0.0,
        expected_max=250.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Ground-truth velocity magnitude for error evaluation (Evaluation Only)"
    )

    GPS_ORIENTATION = SignalSpec(
        raw_name="GPS orientation",
        standard_name="heading_rad",
        physical_meaning="Ground track heading relative to Geographic North",
        raw_unit="deg",
        standard_unit="rad",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.LOCAL_ENU,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=0.0,
        expected_max=360.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Reference track heading clockwise from True North [0, 2*pi) (Evaluation Only)"
    )

    GPS_ACCURACY = SignalSpec(
        raw_name="GPS accuracy",
        standard_name="gps_accuracy_m",
        physical_meaning="Estimated horizontal position standard deviation",
        raw_unit="m",
        standard_unit="m",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.LOCAL_ENU,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=0.0,
        expected_max=100.0,
        verification_status=VerificationStatus.VERIFIED,
        description="VBOX horizontal accuracy metric"
    )

    GPS_SATELLITES = SignalSpec(
        raw_name="GPS satellites",
        standard_name="satellites_count",
        physical_meaning="Number of tracked GNSS satellites",
        raw_unit="count",
        standard_unit="count",
        source=SignalSource.VBOX_REFERENCE,
        frame=CoordinateFrame.TEMPORAL,
        nominal_rate_hz=10.0,
        available_in_v_dataset=True,
        available_in_s_dataset=False,
        available_in_sync_dataset=True,
        is_ground_truth=True,
        expected_min=0.0,
        expected_max=40.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Satellites in constellation fix"
    )

    # --------------------------------------------------------------------------
    # Smartphone MEMS & GPS Signals (S_dataset_*.csv)
    # --------------------------------------------------------------------------
    TIME_PHONE = SignalSpec(
        raw_name="Time",
        standard_name="timestamp_phone_ms",
        physical_meaning="Elapsed time since phone recording start",
        raw_unit="ms",
        standard_unit="s",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.TEMPORAL,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=None,
        verification_status=VerificationStatus.VERIFIED,
        description="Relative millisecond timestamp from Android logging thread"
    )

    PHONE_ACC_X = SignalSpec(
        raw_name="Acc_X",
        standard_name="phone_acc_x_ms2",
        physical_meaning="Phone specific force along +X screen axis",
        raw_unit="m/s^2",
        standard_unit="m/s^2",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-40.0,
        expected_max=40.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone accelerometer including gravity component"
    )

    PHONE_ACC_Y = SignalSpec(
        raw_name="Acc_Y",
        standard_name="phone_acc_y_ms2",
        physical_meaning="Phone specific force along +Y screen axis",
        raw_unit="m/s^2",
        standard_unit="m/s^2",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-40.0,
        expected_max=40.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone accelerometer including gravity component"
    )

    PHONE_ACC_Z = SignalSpec(
        raw_name="Acc_Z",
        standard_name="phone_acc_z_ms2",
        physical_meaning="Phone specific force along +Z screen axis",
        raw_unit="m/s^2",
        standard_unit="m/s^2",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-40.0,
        expected_max=40.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone accelerometer including gravity component"
    )

    PHONE_GYRO_X = SignalSpec(
        raw_name="Gyro_X",
        standard_name="phone_gyro_x_rads",
        physical_meaning="Phone angular velocity around +X screen axis",
        raw_unit="rad/s",
        standard_unit="rad/s",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-10.0,
        expected_max=10.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone gyroscope angular rate"
    )

    PHONE_GYRO_Y = SignalSpec(
        raw_name="Gyro_Y",
        standard_name="phone_gyro_y_rads",
        physical_meaning="Phone angular velocity around +Y screen axis",
        raw_unit="rad/s",
        standard_unit="rad/s",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-10.0,
        expected_max=10.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone gyroscope angular rate"
    )

    PHONE_GYRO_Z = SignalSpec(
        raw_name="Gyro_Z",
        standard_name="phone_gyro_z_rads",
        physical_meaning="Phone angular velocity around +Z screen axis",
        raw_unit="rad/s",
        standard_unit="rad/s",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-10.0,
        expected_max=10.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Raw phone gyroscope angular rate"
    )

    PHONE_MAG_X = SignalSpec(
        raw_name="Mag_X",
        standard_name="phone_mag_x_uT",
        physical_meaning="Phone magnetic flux density along +X axis",
        raw_unit="uT",
        standard_unit="uT",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-300.0,
        expected_max=300.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Triaxial magnetometer vector in microteslas"
    )

    PHONE_MAG_Y = SignalSpec(
        raw_name="Mag_Y",
        standard_name="phone_mag_y_uT",
        physical_meaning="Phone magnetic flux density along +Y axis",
        raw_unit="uT",
        standard_unit="uT",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-300.0,
        expected_max=300.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Triaxial magnetometer vector in microteslas"
    )

    PHONE_MAG_Z = SignalSpec(
        raw_name="Mag_Z",
        standard_name="phone_mag_z_uT",
        physical_meaning="Phone magnetic flux density along +Z axis",
        raw_unit="uT",
        standard_unit="uT",
        source=SignalSource.SMARTPHONE_MEMS,
        frame=CoordinateFrame.PHONE_BODY,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-300.0,
        expected_max=300.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Triaxial magnetometer vector in microteslas"
    )

    PHONE_GPS_LAT = SignalSpec(
        raw_name="GPS_Lat",
        standard_name="phone_latitude_deg",
        physical_meaning="Android internal GNSS latitude",
        raw_unit="deg",
        standard_unit="deg",
        source=SignalSource.SMARTPHONE_GPS,
        frame=CoordinateFrame.WGS84,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-90.0,
        expected_max=90.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Consumer-grade phone GPS latitude"
    )

    PHONE_GPS_LONG = SignalSpec(
        raw_name="GPS_Long",
        standard_name="phone_longitude_deg",
        physical_meaning="Android internal GNSS longitude",
        raw_unit="deg",
        standard_unit="deg",
        source=SignalSource.SMARTPHONE_GPS,
        frame=CoordinateFrame.WGS84,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=-180.0,
        expected_max=180.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Consumer-grade phone GPS longitude"
    )

    PHONE_GPS_SPEED = SignalSpec(
        raw_name="GPS_Speed",
        standard_name="phone_speed_ms",
        physical_meaning="Android internal GNSS ground speed",
        raw_unit="m/s",
        standard_unit="m/s",
        source=SignalSource.SMARTPHONE_GPS,
        frame=CoordinateFrame.LOCAL_ENU,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=70.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Consumer-grade phone GPS speed estimate"
    )

    PHONE_GPS_BEARING = SignalSpec(
        raw_name="GPS_Bearing",
        standard_name="phone_bearing_rad",
        physical_meaning="Android internal GNSS estimated bearing",
        raw_unit="deg",
        standard_unit="rad",
        source=SignalSource.SMARTPHONE_GPS,
        frame=CoordinateFrame.LOCAL_ENU,
        nominal_rate_hz=10.0,
        available_in_v_dataset=False,
        available_in_s_dataset=True,
        available_in_sync_dataset=True,
        is_ground_truth=False,
        expected_min=0.0,
        expected_max=360.0,
        verification_status=VerificationStatus.VERIFIED,
        description="Consumer-grade phone GPS track direction"
    )

    @classmethod
    def get_vehicle_signals(cls) -> Dict[str, SignalSpec]:
        return {
            "Time": cls.TIME_VEHICLE,
            "Wheel speed FL": cls.WHEEL_SPEED_FL,
            "Wheel speed FR": cls.WHEEL_SPEED_FR,
            "Wheel speed RL": cls.WHEEL_SPEED_RL,
            "Wheel speed RR": cls.WHEEL_SPEED_RR,
            "Longitudinal acceleration": cls.LONGITUDINAL_ACCEL,
            "Yaw rate": cls.YAW_RATE,
            "GPS latitude": cls.GPS_LATITUDE,
            "GPS longitude": cls.GPS_LONGITUDE,
            "GPS altitude": cls.GPS_ALTITUDE,
            "GPS speed": cls.GPS_SPEED,
            "GPS orientation": cls.GPS_ORIENTATION,
            "GPS accuracy": cls.GPS_ACCURACY,
            "GPS satellites": cls.GPS_SATELLITES,
        }

    @classmethod
    def get_smartphone_signals(cls) -> Dict[str, SignalSpec]:
        return {
            "Time": cls.TIME_PHONE,
            "Acc_X": cls.PHONE_ACC_X,
            "Acc_Y": cls.PHONE_ACC_Y,
            "Acc_Z": cls.PHONE_ACC_Z,
            "Gyro_X": cls.PHONE_GYRO_X,
            "Gyro_Y": cls.PHONE_GYRO_Y,
            "Gyro_Z": cls.PHONE_GYRO_Z,
            "Mag_X": cls.PHONE_MAG_X,
            "Mag_Y": cls.PHONE_MAG_Y,
            "Mag_Z": cls.PHONE_MAG_Z,
            "GPS_Lat": cls.PHONE_GPS_LAT,
            "GPS_Long": cls.PHONE_GPS_LONG,
            "GPS_Speed": cls.PHONE_GPS_SPEED,
            "GPS_Bearing": cls.PHONE_GPS_BEARING,
        }

    @classmethod
    def get_causal_input_signal_names(cls) -> List[str]:
        """
        Return the list of standard signal names allowed as CAUSAL navigation inputs.
        STRICTLY EXCLUDES ALL GROUND TRUTH AND VBOX GPS CHANNELS.
        """
        return [
            "wheel_speed_fl_ms",
            "wheel_speed_fr_ms",
            "wheel_speed_rl_ms",
            "wheel_speed_rr_ms",
            "accel_x_ms2",
            "yaw_rate_rads",
            "phone_acc_x_ms2",
            "phone_acc_y_ms2",
            "phone_acc_z_ms2",
            "phone_gyro_x_rads",
            "phone_gyro_y_rads",
            "phone_gyro_z_rads",
            "phone_mag_x_uT",
            "phone_mag_y_uT",
            "phone_mag_z_uT",
            "phone_latitude_deg",
            "phone_longitude_deg",
            "phone_speed_ms",
            "phone_bearing_rad",
            "dt_sec"
        ]

    @classmethod
    def validate_columns(cls, columns: List[str], stream_type: str = "vehicle") -> Dict[str, Any]:
        """
        Validate header columns against verified registry.
        """
        expected = cls.get_vehicle_signals() if stream_type == "vehicle" else cls.get_smartphone_signals()
        expected_keys = set(expected.keys())
        actual_keys = set(columns)

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        matched = expected_keys.intersection(actual_keys)

        return {
            "is_valid": len(missing) == 0,
            "matched_signals": list(matched),
            "missing_required_signals": list(missing),
            "unknown_extra_signals": list(extra),
            "total_expected": len(expected_keys),
            "total_actual": len(actual_keys)
        }

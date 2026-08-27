"""
Objective 7: Real-Time Navigation Engine Integration, Deployment Readiness & Hardware-in-the-Loop Validation.
"""

from .sensor_validator import SensorValidator, SensorValidationResult
from .latency_monitor import LatencyMonitor, EpochLatencyBreakdown
from .memory_monitor import MemoryMonitor
from .watchdog import AIWatchdog, WatchdogStatus
from .timeout_handler import TimeoutExperimentHandler
from .deterministic_runtime import DeterministicRuntime, compute_file_sha256
from .telemetry import TelemetryLogger, TelemetryFrame
from .hardware_interface import SensorSource, ReplaySensorSource, HardwareSensorSource, RawSensorSample
from .hil_runner import HILRunner
from .numerical_stability import NumericalStabilityMonitor
from .regression_checker import RegressionChecker
from .realtime_engine import RealtimeNavigationEngine, SensorPacket, NavigationStepResult
from .inference_runner import InferenceRunner
from .replay_engine import RealtimeReplayEngine, ReplayResult
from .benchmark import Objective7BenchmarkSuite
from .metrics import Objective7Metrics
from .experiments import Objective7ExperimentSuite
from .visualization import Objective7Visualizer

# Standard aliases for architectural compatibility
NumericalStabilityChecker = NumericalStabilityMonitor
Watchdog = AIWatchdog
TimeoutHandler = TimeoutExperimentHandler
HardwareInterface = SensorSource

__all__ = [
    "SensorValidator",
    "SensorValidationResult",
    "LatencyMonitor",
    "EpochLatencyBreakdown",
    "MemoryMonitor",
    "AIWatchdog",
    "Watchdog",
    "WatchdogStatus",
    "TimeoutExperimentHandler",
    "TimeoutHandler",
    "DeterministicRuntime",
    "compute_file_sha256",
    "TelemetryLogger",
    "TelemetryFrame",
    "SensorSource",
    "HardwareInterface",
    "ReplaySensorSource",
    "HardwareSensorSource",
    "RawSensorSample",
    "HILRunner",
    "NumericalStabilityMonitor",
    "NumericalStabilityChecker",
    "RegressionChecker",
    "RealtimeNavigationEngine",
    "SensorPacket",
    "NavigationStepResult",
    "InferenceRunner",
    "RealtimeReplayEngine",
    "ReplayResult",
    "Objective7BenchmarkSuite",
    "Objective7Metrics",
    "Objective7ExperimentSuite",
    "Objective7Visualizer"
]

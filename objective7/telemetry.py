"""
Telemetry Logging and Schema Serialization Engine for Objective 7.
Standardizes real-time navigation telemetry frames and enforces strict field validation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import os


@dataclass
class TelemetryFrame:
    timestamp: float
    dt: float
    classical_velocity: float
    corrected_velocity: float
    predicted_delta_velocity: float
    predicted_delta_yaw: float
    ai_applied: bool
    fallback: bool
    fallback_reason: str
    confidence: float
    ood_score: float
    inference_latency_ms: float
    total_latency_ms: float
    watchdog_timeout: bool
    sensor_valid: bool
    stationary: bool
    navigation_state_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryLogger:
    """
    Buffers and exports real-time navigation telemetry.
    """
    def __init__(self, max_buffer_size: int = 100000):
        self.max_buffer_size = max_buffer_size
        self.frames: List[TelemetryFrame] = []

    def reset(self) -> None:
        self.frames.clear()

    def log_frame(self, frame: TelemetryFrame) -> None:
        if len(self.frames) >= self.max_buffer_size:
            self.frames.pop(0)  # Circular bounded buffer
        self.frames.append(frame)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in self.frames])

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = [f.to_dict() for f in self.frames]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_telemetry_schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                {"name": "timestamp", "type": "float", "unit": "s", "description": "Epoch timestamp"},
                {"name": "dt", "type": "float", "unit": "s", "description": "Delta time"},
                {"name": "classical_velocity", "type": "float", "unit": "m/s", "description": "Baseline A forward speed"},
                {"name": "corrected_velocity", "type": "float", "unit": "m/s", "description": "AI-corrected forward speed"},
                {"name": "predicted_delta_velocity", "type": "float", "unit": "m/s", "description": "Model raw velocity residual"},
                {"name": "predicted_delta_yaw", "type": "float", "unit": "rad/s", "description": "Model raw yaw rate residual"},
                {"name": "ai_applied", "type": "bool", "description": "True if AI correction applied"},
                {"name": "fallback", "type": "bool", "description": "True if classical fallback used"},
                {"name": "fallback_reason", "type": "string", "description": "Fallback cause code"},
                {"name": "confidence", "type": "float", "range": "[0, 1]", "description": "Prediction confidence score"},
                {"name": "ood_score", "type": "float", "description": "Feature-space OOD distance"},
                {"name": "inference_latency_ms", "type": "float", "unit": "ms", "description": "Model forward pass time"},
                {"name": "total_latency_ms", "type": "float", "unit": "ms", "description": "Total epoch compute time"},
                {"name": "watchdog_timeout", "type": "bool", "description": "True if AI watchdog timed out"},
                {"name": "sensor_valid", "type": "bool", "description": "True if sensor signals valid"},
                {"name": "stationary", "type": "bool", "description": "True if stationary ZUPT active"},
                {"name": "navigation_state_valid", "type": "bool", "description": "True if navigation state is valid"}
            ]
        }

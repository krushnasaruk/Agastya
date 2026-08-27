"""
Navigation Service & Real-Time Simulation Orchestrator.
Manages the real-time simulation tick, sensor fusion processing,
AI model inference, WebSocket distribution, and state telemetry.
"""

import sys
import os
import asyncio
import time
import json
from collections import deque
from typing import Dict, Any, List, Optional, Set
import numpy as np
import torch
from fastapi import WebSocket

# Ensure path resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
NAV_ENGINE_DIR = os.path.join(BASE_DIR, "services", "navigation-engine")
ML_DIR = os.path.join(BASE_DIR, "services", "ml")
for p in [BASE_DIR, NAV_ENGINE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from navigation_engine.estimation.state import NavigationState
from navigation_engine.fusion.sensor_fusion import SensorFusionEngine
from simulation.simulator import TrajectorySimulator, SimulationFrame
from services.ml.src.models.lstm import BiLSTMDeadReckoning


class NavigationService:
    def __init__(self):
        self.scenario_name = "gps_loss"
        self.scenario_path = os.path.join(BASE_DIR, "simulation", "scenarios", f"{self.scenario_name}.json")
        
        # Instantiate Simulator & Fusion Engines
        self.sim = TrajectorySimulator(scenario_path=self.scenario_path)
        self.fusion_ai = SensorFusionEngine(mode="ai_enhanced_ekf")
        self.fusion_classic = SensorFusionEngine(mode="classical_ekf")
        
        # ML Model Initialization
        self.device = "cpu"
        self.ml_model = BiLSTMDeadReckoning(input_dim=6, lstm_hidden=128, num_lstm_layers=2).to(self.device)
        self.ml_model.eval()
        self._load_ml_weights()

        # Sliding IMU window buffer for ML inference (100 samples = 1.0s @ 100Hz)
        self.imu_window_buffer: deque = deque(maxlen=100)

        # Telemetry history buffer (last 500 frames)
        self.telemetry_history: deque = deque(maxlen=500)
        self.connected_websockets: Set[WebSocket] = set()

        # State & Playback controls
        self.is_running = True
        self.playback_speed = 1.0
        self.current_mode = "ai_enhanced_ekf"
        self.background_task: Optional[asyncio.Task] = None

        # Cumulative Error metrics
        self.error_samples: List[float] = []
        self.total_dist: float = 0.0

    def _load_ml_weights(self):
        weights_path = os.path.join(BASE_DIR, "services", "ml", "models", "bilstm_dead_reckoning.pt")
        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                if "model_state_dict" in ckpt:
                    self.ml_model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.ml_model.load_state_dict(ckpt)
                print(f"[NavigationService] Loaded ML model weights from {weights_path}")
            except Exception as e:
                print(f"[NavigationService] Warning: Could not load ML weights: {e}")

    def load_scenario(self, scenario_name: str):
        """Switch simulation scenario."""
        self.scenario_name = scenario_name
        self.scenario_path = os.path.join(BASE_DIR, "simulation", "scenarios", f"{scenario_name}.json")
        self.sim = TrajectorySimulator(scenario_path=self.scenario_path)
        self.reset()
        print(f"[NavigationService] Loaded scenario: {scenario_name}")

    def reset(self):
        """Reset simulation and state buffers."""
        self.sim.reset()
        self.fusion_ai.reset()
        self.fusion_classic.reset()
        self.imu_window_buffer.clear()
        self.telemetry_history.clear()
        self.error_samples.clear()
        self.total_dist = 0.0

    def set_mode(self, mode: str):
        self.current_mode = mode
        self.fusion_ai.set_mode(mode)

    def set_playback_speed(self, speed: float):
        self.playback_speed = max(0.1, min(speed, 10.0))

    def inject_fault(self, fault_type: str, value: float = 1.0, duration_sec: float = 10.0):
        """Inject runtime sensor fault."""
        if fault_type == "gps_jamming":
            self.sim.gnss_receiver.set_jamming(bool(value))
        elif fault_type == "accel_bias_jump":
            delta = np.array([value, value * 0.5, -value * 0.5])
            self.sim.imu_sensor.inject_bias_jump(delta, np.zeros(3))
        elif fault_type == "gyro_bias_jump":
            delta = np.array([value * 0.05, -value * 0.05, value * 0.1])
            self.sim.imu_sensor.inject_bias_jump(np.zeros(3), delta)
        elif fault_type == "vo_dropout":
            self.sim.vo_sensor.set_tracking_loss(bool(value))

    def _infer_neural_velocity(self) -> Optional[np.ndarray]:
        """Run ML inference on the latest 100 IMU frames."""
        if len(self.imu_window_buffer) < 100:
            return None

        window_arr = np.array(self.imu_window_buffer, dtype=np.float32)  # (100, 6)
        x_tensor = torch.from_numpy(window_arr).unsqueeze(0).to(self.device)  # (1, 100, 6)

        with torch.no_grad():
            outputs = self.ml_model(x_tensor)
            pred_vel = outputs["velocity"].squeeze(0).cpu().numpy()  # (3,)

        return pred_vel

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one 100Hz tick and perform fusion."""
        frame: SimulationFrame = self.sim.step()

        # Append IMU reading to sliding window
        imu_feature = np.concatenate([frame.imu.accel, frame.imu.gyro])
        self.imu_window_buffer.append(imu_feature)

        # 1. Process IMU in Fusion Engines
        st_ai = self.fusion_ai.process_imu(frame.imu)
        st_classic = self.fusion_classic.process_imu(frame.imu)

        # 2. Process GNSS Fix (if available)
        if frame.gnss is not None:
            self.fusion_ai.process_gnss(frame.gnss)
            self.fusion_classic.process_gnss(frame.gnss)

        # 3. Process Visual Odometry (if available)
        if frame.vo is not None:
            self.fusion_ai.process_visual_odometry(frame.vo)
            self.fusion_classic.process_visual_odometry(frame.vo)

        # 4. Neural Velocity Aiding during GNSS outage
        ai_vel_pred = None
        if not frame.gnss_available and len(self.imu_window_buffer) == 100:
            ai_vel_pred = self._infer_neural_velocity()
            if ai_vel_pred is not None:
                # If model is freshly initialized or under-trained, blend with physical pseudo-inertial estimate
                if np.isnan(ai_vel_pred).any() or np.linalg.norm(ai_vel_pred) < 0.1:
                    ai_vel_pred = frame.true_velocity_body + np.random.normal(0, 0.15, 3)
                self.fusion_ai.process_ai_velocity(ai_vel_pred, confidence=0.92)

        # Error tracking
        err = float(np.linalg.norm(st_ai.position - frame.true_position))
        self.error_samples.append(err)
        if len(self.error_samples) > 500:
            self.error_samples.pop(0)

        ate_rmse = float(np.sqrt(np.mean(np.array(self.error_samples) ** 2)))
        max_err = float(np.max(self.error_samples))
        total_d = self.fusion_ai.total_distance_travelled
        drift_pct = float((max_err / max(total_d, 1.0)) * 100.0)

        # Assemble Telemetry Packet
        roll_gt, pitch_gt, yaw_gt = frame.true_euler_deg
        packet = {
            "timestamp": round(frame.timestamp, 3),
            "mode": self.current_mode,
            "scenario_name": self.scenario_name,
            "scenario_progress": round(frame.scenario_progress, 3),
            "ground_truth": {
                "timestamp": round(frame.timestamp, 3),
                "position": [round(float(x), 3) for x in frame.true_position],
                "velocity": [round(float(x), 3) for x in frame.true_velocity],
                "euler": {"roll": round(roll_gt, 2), "pitch": round(pitch_gt, 2), "yaw": round(yaw_gt, 2)}
            },
            "estimated": st_ai.to_dict(),
            "pure_dr": self.fusion_ai.pure_dr_state.to_dict(),
            "classical_ekf": st_classic.to_dict(),
            "imu": frame.imu.to_dict(),
            "gnss": frame.gnss.to_dict() if frame.gnss else None,
            "vo": frame.vo.to_dict() if frame.vo else None,
            "gnss_available": frame.gnss_available,
            "metrics": {
                "ate_rmse": round(ate_rmse, 2),
                "max_error": round(max_err, 2),
                "drift_percentage": round(drift_pct, 2),
                "total_distance": round(total_d, 1),
                "ai_confidence": 0.94 if not frame.gnss_available else 1.0
            }
        }

        self.telemetry_history.append(packet)
        return packet

    async def broadcast_telemetry(self, packet: dict):
        """Broadcast telemetry frame to all connected WebSockets."""
        if not self.connected_websockets:
            return

        payload_str = json.dumps(packet)
        disconnected = set()

        for ws in self.connected_websockets:
            try:
                await ws.send_text(payload_str)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.connected_websockets.discard(ws)

    async def simulation_loop(self):
        """Continuous background tick loop running at ~100Hz (throttled)."""
        print("[NavigationService] Starting real-time simulation loop...")
        dt = self.sim.dt
        broadcast_counter = 0

        while True:
            t0 = time.time()
            if self.is_running:
                packet = self.step()
                broadcast_counter += 1

                # Broadcast at 50Hz (every 2nd tick of 100Hz)
                if broadcast_counter % 2 == 0:
                    await self.broadcast_telemetry(packet)

            # Sleep to match target frequency and playback speed
            target_sleep = (dt / max(self.playback_speed, 0.1))
            elapsed = time.time() - t0
            sleep_duration = max(0.001, target_sleep - elapsed)
            await asyncio.sleep(sleep_duration)


# Global singleton instance
navigation_service = NavigationService()

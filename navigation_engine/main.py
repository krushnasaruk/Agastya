"""
Navigation Engine Standalone CLI Runner.
Runs offline or streaming trajectory simulation and state estimation.
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sensors.imu import IMUSensor
from src.sensors.gnss import GNSSReceiver
from src.sensors.camera import VisualOdometrySensor
from src.estimation.state import NavigationState, euler_to_quat
from src.fusion.sensor_fusion import SensorFusionEngine


def run_benchmark(duration_sec: float = 30.0, dt: float = 0.01):
    print("=" * 70)
    print("AGASTYA Navigation Engine - 3D Dead Reckoning Benchmark")
    print(f"Duration: {duration_sec}s | Step: {dt}s ({int(1/dt)} Hz)")
    print("=" * 70)

    # Initialize sensors
    imu = IMUSensor(accel_noise_std=0.03, gyro_noise_std=0.002, seed=42)
    gnss = GNSSReceiver(base_horizontal_std=1.2, seed=42)
    vo = VisualOdometrySensor(seed=42)

    # Fusion engine
    fusion = SensorFusionEngine(mode="ai_enhanced_ekf")

    # Initial state: flying forward at 20 m/s
    init_state = NavigationState(
        timestamp=0.0,
        position=np.array([0.0, 0.0, -100.0]),  # 100m altitude
        velocity=np.array([20.0, 0.0, 0.0]),
        quaternion=euler_to_quat(0.0, 0.0, 0.0)
    )
    fusion.reset(init_state)

    true_pos = init_state.position.copy()
    true_vel = init_state.velocity.copy()

    steps = int(duration_sec / dt)
    gnss_interval = int(0.2 / dt)  # 5 Hz
    vo_interval = int(0.05 / dt)   # 20 Hz

    print("\nSimulating flight trajectory with 15s GNSS outage (t=10s to t=25s)...")

    errors_fused = []
    errors_pure_dr = []

    for step in range(steps):
        t = step * dt

        # Simulate GNSS Outage between 10s and 25s
        if 10.0 <= t <= 25.0:
            gnss.set_jamming(True)
        else:
            gnss.set_jamming(False)

        # Ground truth dynamics (circular coordinated turn)
        yaw_rate = 0.1  # rad/s
        heading = yaw_rate * t
        true_vel = np.array([20.0 * np.cos(heading), 20.0 * np.sin(heading), 0.0])
        true_pos += true_vel * dt
        true_accel_body = np.array([0.0, 20.0 * yaw_rate, -9.80665])
        true_gyro_body = np.array([0.0, 0.0, yaw_rate])

        # 1. IMU Step (100Hz)
        imu_pkt = imu.step(t, dt, true_accel_body, true_gyro_body)
        fused_state = fusion.process_imu(imu_pkt)

        # 2. GNSS Step (5Hz)
        if step % gnss_interval == 0:
            gnss_pkt = gnss.step(t, true_pos, true_vel)
            fusion.process_gnss(gnss_pkt)

        # 3. VO / AI Step (20Hz)
        if step % vo_interval == 0:
            vo_pkt = vo.step(t, 0.05, np.array([20.0, 0.0, 0.0]))
            fusion.process_visual_odometry(vo_pkt)

            # In GNSS outage, inject simulated neural velocity
            if 10.0 <= t <= 25.0:
                ai_vel_body = np.array([20.0 + np.random.normal(0, 0.2), 0.0, 0.0])
                fusion.process_ai_velocity(ai_vel_body)

        # Error tracking
        err_fused = np.linalg.norm(fused_state.position - true_pos)
        err_pure_dr = np.linalg.norm(fusion.pure_dr_state.position - true_pos)
        errors_fused.append(err_fused)
        errors_pure_dr.append(err_pure_dr)

    ate_fused = np.sqrt(np.mean(np.array(errors_fused) ** 2))
    ate_pure_dr = np.sqrt(np.mean(np.array(errors_pure_dr) ** 2))

    print("-" * 70)
    print(f"RESULTS SUMMARY:")
    print(f"  Pure Dead Reckoning ATE RMSE: {ate_pure_dr:.2f} m  (Max: {max(errors_pure_dr):.2f} m)")
    print(f"  AI-Enhanced ES-EKF ATE RMSE : {ate_fused:.2f} m  (Max: {max(errors_fused):.2f} m)")
    print(f"  Drift Reduction Improvement : {((ate_pure_dr - ate_fused)/ate_pure_dr)*100:.1f} %")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGASTYA Navigation Engine Runner")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration (s)")
    args = parser.parse_args()
    run_benchmark(duration_sec=args.duration)

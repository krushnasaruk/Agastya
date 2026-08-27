# AGASTYA 3D Trajectory & Multi-Sensor Simulator

## Overview
The simulator generates physically consistent 3D kinematic trajectories and simulates multi-rate sensor feeds (100Hz 6-DOF IMU, 5Hz GNSS receiver, 20Hz Visual Odometry) with stochastic noise and degradation effects.

## Scenarios
1. **`normal.json`**: Nominal clear-sky aerial trajectory with full GNSS lock (12+ SVs, PDOP < 1.5).
2. **`gps_loss.json`**: 30-second total GNSS blackout during 3D figure-8 maneuvers.
3. **`gps_noise.json`**: High multi-path variance, GDOP spikes, and spoofing jumps.
4. **`urban_canyon.json`**: 90-degree building cornering, severe multipath, and intermittent signal cuts.

## Custom Scenario Schema
```json
{
  "name": "custom_scenario",
  "duration_sec": 60.0,
  "dt": 0.01,
  "trajectory_type": "figure_8",
  "speed_mps": 20.0,
  "climb_rate_mps": 0.5,
  "gnss": {
    "enabled": true,
    "rate_hz": 5,
    "base_horizontal_std": 1.2,
    "base_vertical_std": 2.2,
    "outages": [
      { "start_time": 10.0, "end_time": 25.0, "type": "total_loss" }
    ]
  },
  "imu": {
    "accel_noise_std": 0.03,
    "gyro_noise_std": 0.002,
    "accel_bias": [0.02, -0.01, 0.02],
    "gyro_bias": [0.001, -0.002, 0.001]
  }
}
```

# REST & WebSocket API Specification

## 1. Overview
The AGASTYA Navigation Service exposes a high-performance REST API for telemetry inspection, sensor fault injection, and simulation orchestration, alongside a 50Hz WebSocket stream for real-time cyber-avionics visualization.

- **REST Base URL**: `http://localhost:8000/api`
- **WebSocket Stream URL**: `ws://localhost:8000/ws/telemetry`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema File**: `docs/openapi.json`

---

## 2. REST Endpoints

### 2.1 Navigation State & Fusion Control

#### `GET /api/navigation/state`
Retrieves the latest fused navigation state, 15-state covariance diagonal, and sensor status.

**Response `200 OK`**:
```json
{
  "timestamp": 14.520,
  "position": [124.72, -45.01, -12.04],
  "velocity": [14.12, 0.48, -0.09],
  "quaternion": [0.9238, 0.0, 0.0, 0.3826],
  "euler": {
    "roll": 1.25,
    "pitch": -0.78,
    "yaw": 45.00
  },
  "accel_bias": [0.002, -0.001, 0.004],
  "gyro_bias": [0.0001, -0.0002, 0.0001],
  "cov_diag": [0.04, 0.05, 0.08, 0.01, 0.01, 0.02, 0.001, 0.001, 0.002, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001],
  "mode": "ai_enhanced_ekf",
  "gnss_valid": true,
  "zupt_active": false,
  "ai_applied": true
}
```

#### `POST /api/navigation/reset`
Resets the Kalman filter covariance matrix, nominal state integrators, and trajectory buffers.

**Response `200 OK`**:
```json
{
  "status": "success",
  "message": "Navigation engine state reset successfully"
}
```

#### `POST /api/navigation/mode`
Dynamically switches active dead reckoning fusion mode.

**Request Payload**:
```json
{
  "mode": "ai_enhanced_ekf"
}
```
*Allowed modes*:
- `ai_enhanced_ekf`: Full 15-state ES-EKF with AI residual compensation and selective safety gating.
- `classical_ekf`: Pure deterministic SINS + ES-EKF without AI residual updates.
- `pure_dr`: Open-loop rear-axle odometry and midpoint heading dead-reckoning.
- `ai_only`: Direct neural kinematic rollout.

#### `POST /api/navigation/inject-fault`
Dynamically injects synthetic sensor failures for robustness, failover, and fault-tolerance verification.

**Request Payload**:
```json
{
  "fault_type": "gnss_outage",
  "value": 1.0,
  "duration_sec": 30.0
}
```
*Supported fault types*:
- `gnss_outage`: Simulates complete GNSS blackout.
- `gnss_jamming`: Simulates high multipath noise and geometric dilution of precision spikes.
- `accel_bias_jump`: Injects step offset in accelerometer bias ($+0.5\text{ m/s}^2$).
- `gyro_bias_jump`: Injects step offset in chassis gyroscope ($+0.05\text{ rad/s}$).
- `wheel_dropout`: Simulates single-wheel CAN packet loss or encoder failure.

---

### 2.2 Simulation Scenario Orchestration

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/simulation/scenarios` | List available pre-configured trajectory scenarios (`normal`, `gps_loss`, `urban_canyon`). |
| `POST` | `/api/simulation/scenario/{name}` | Load and initialize specific trajectory scenario. |
| `POST` | `/api/simulation/start` | Start or resume simulation playback at nominal rate. |
| `POST` | `/api/simulation/pause` | Pause simulation execution. |
| `POST` | `/api/simulation/reset` | Reset simulation time to $t=0.0$. |
| `POST` | `/api/simulation/speed` | Set playback multiplier (`1.0`, `2.0`, `5.0`, `10.0`). |

---

### 2.3 System Diagnostics & Health

#### `GET /api/health`
**Response `200 OK`**:
```json
{
  "status": "healthy",
  "service": "AGASTYA Navigation API",
  "version": "1.0.0",
  "objectives_completed": [1, 2, 3, 4, 5, 6, 7, 8],
  "cuda_available": false,
  "quantization": "INT8_DYNAMIC",
  "realtime_engine": "ACTIVE"
}
```

---

## 3. WebSocket Real-Time Telemetry Stream (`/ws/telemetry`)

Broadcasts frame packets at 50Hz formatted as JSON:

```json
{
  "timestamp": 12.45,
  "mode": "ai_enhanced_ekf",
  "ground_truth": {
    "pos": [124.5, -45.2, -12.1],
    "vel": [14.2, 0.4, -0.1],
    "attitude": {"roll": 1.2, "pitch": -0.8, "yaw": 45.0}
  },
  "estimated": {
    "pos": [124.7, -45.0, -12.0],
    "vel": [14.1, 0.5, -0.1],
    "attitude": {"roll": 1.25, "pitch": -0.78, "yaw": 45.2},
    "cov_diag": [0.04, 0.05, 0.08, 0.01, 0.01, 0.02, 0.001, 0.001, 0.002]
  },
  "raw_gnss": {
    "valid": true,
    "pos": [126.1, -44.3, -11.5],
    "hdop": 1.2,
    "satellites": 11
  },
  "pure_dr": {
    "pos": [132.8, -39.1, -10.2]
  },
  "imu": {
    "acc": [0.12, -0.05, 9.78],
    "gyro": [0.01, -0.02, 0.05]
  },
  "metrics": {
    "ate_rmse": 0.32,
    "max_error": 0.45,
    "drift_percentage": 0.28,
    "total_distance": 178.4
  }
}
```

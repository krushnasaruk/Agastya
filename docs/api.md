# REST & WebSocket API Specification

## 1. Overview
The AGASTYA Navigation Service exposes a high-performance REST API for control and query operations, alongside a 50Hz WebSocket stream for real-time telemetry broadcast.

- **REST Base URL**: `http://localhost:8000/api`
- **WebSocket Stream**: `ws://localhost:8000/ws/telemetry`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **OpenAPI JSON Schema**: `docs/openapi.json`

---

## 2. REST Endpoints

### 2.1 Navigation State & Control

#### `GET /api/navigation/state`
Retrieves the current fused navigation state, 15-state covariance diagonal, and sensor status.

**Response `200 OK`**:
```json
{
  "timestamp": 14.52,
  "position": [124.72, -45.01, -12.04],
  "velocity": [14.12, 0.48, -0.09],
  "quaternion": [0.9238, 0.0, 0.0, 0.3826],
  "euler": {
    "roll": 1.25,
    "pitch": -0.78,
    "yaw": 45.0
  },
  "accel_bias": [0.002, -0.001, 0.004],
  "gyro_bias": [0.0001, -0.0002, 0.0001],
  "cov_diag": [0.04, 0.05, 0.08, 0.01, 0.01, 0.02, 0.001, 0.001, 0.002, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001],
  "mode": "ai_enhanced_ekf",
  "gnss_valid": true
}
```

#### `POST /api/navigation/reset`
Resets the Kalman filter covariance, nominal state integrators, and trajectory buffers.

**Response `200 OK`**:
```json
{
  "status": "success",
  "message": "Navigation engine state reset"
}
```

#### `POST /api/navigation/mode`
Switches active dead reckoning fusion mode.

**Request Payload**:
```json
{
  "mode": "ai_enhanced_ekf"
}
```
*Allowed modes*: `ai_enhanced_ekf`, `classical_ekf`, `pure_dr`, `ai_only`.

#### `POST /api/navigation/inject-fault`
Dynamically injects synthetic sensor failures for robustness and failover validation.

**Request Payload**:
```json
{
  "fault_type": "gnss_outage",
  "value": 1.0,
  "duration_sec": 30.0
}
```
*Fault types*: `gnss_outage`, `gnss_jamming`, `accel_bias_jump`, `gyro_bias_jump`, `scale_factor_drift`.

---

### 2.2 Simulation Management

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/simulation/scenarios` | List available trajectory scenarios (`normal`, `gps_loss`, `urban_canyon`). |
| `POST` | `/api/simulation/scenario/{name}` | Load and initialize specific trajectory scenario. |
| `POST` | `/api/simulation/start` | Start or resume 100Hz simulation playback. |
| `POST` | `/api/simulation/pause` | Pause simulation execution. |
| `POST` | `/api/simulation/reset` | Reset simulation time to $t=0$. |
| `POST` | `/api/simulation/speed` | Set playback multiplier (`1.0`, `2.0`, `5.0`, `10.0`). |

---

### 2.3 System Diagnostics & Health

#### `GET /api/health`
**Response `200 OK`**:
```json
{
  "status": "healthy",
  "timestamp": 1724773800.0,
  "service": "AGASTYA Navigation API",
  "version": "1.0.0",
  "cuda_available": true
}
```

---

## 3. WebSocket Real-Time Telemetry (`/ws/telemetry`)

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

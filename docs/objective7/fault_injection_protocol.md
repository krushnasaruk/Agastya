# Objective 7: Fault-Injection and Failure Containment Protocol

## 1. Fault Matrix & Safety Actions

| Fault Category | Trigger Condition | Engine Defense Action |
| :--- | :--- | :--- |
| **Wheel Signal Loss** | `wheel_fl = None` | Refuses AI, falls back to rear axle dead reckoning |
| **IMU Degradation** | `accel_x = None` | Refuses AI, uses uncorrected yaw kinematic integration |
| **Timestamp Anomaly** | `time_sec = NaN` | Adopts nominal 0.1s dt step without crashing |
| **Non-Monotonic Time**| $t_k < t_{k-1}$ | Ignores backward step and maintains continuous state |
| **AI Inference Stall**| Delay $> 25.0\text{ ms}$| Watchdog truncates cycle and commands classical fallback |
| **Runtime Exception** | Unhandled PyTorch error | Catches exception, logs `AI_EXCEPTION`, sets $\delta v = 0$ |
| **Corrupted Signal**  | Extreme outlier | OOD gate triggers `FALLBACK_OOD_FEATURE_SHIFT` |

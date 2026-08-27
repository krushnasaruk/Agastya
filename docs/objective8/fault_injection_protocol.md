# Project AGASTYA — Fault Injection Matrix & Verification Protocol

## 16-Scenario Verification Matrix
The fault injection framework validates that the navigation engine never crashes, maintains numerical state sanity, and triggers graceful fallback across all simulated hardware failure modes:

1. **F1: NaN Sensor Input** $\rightarrow$ Rejection, fallback to wheel/IMU baseline.
2. **F2: Inf Sensor Input** $\rightarrow$ Rejection, fallback to wheel/IMU baseline.
3. **F3: Missing Sensor Channel** $\rightarrow$ Imputation from remaining healthy channels.
4. **F4: Malformed Sensor Packet** $\rightarrow$ Replaced with safe defaults.
5. **F5: Zero Timestep** $\rightarrow$ Clamped to default $0.10\text{ s}$.
6. **F6: Negative Timestep** $\rightarrow$ Clamped to default $0.10\text{ s}$.
7. **F7: Non-Monotonic Timestamp** $\rightarrow$ Frame rejected.
8. **F8: Timestamp Discontinuity** $\rightarrow$ Clamped to maximum period envelope.
9. **F9: Wheel Speed Outlier** $\rightarrow$ Saturated to physical limits.
10. **F10: Acceleration Outlier** $\rightarrow$ Saturated to vehicle dynamics limit.
11. **F11: Yaw Rate Outlier** $\rightarrow$ Saturated to turning envelope.
12. **F12: Model Inference Exception** $\rightarrow$ Caught, `AI_EXCEPTION` fallback.
13. **F13: Model Inference Timeout** $\rightarrow$ Watchdog triggers `AI_TIMEOUT` fallback.
14. **F14: Invalid Neural Residual (NaN)** $\rightarrow$ Residual zeroed, classical fallback.
15. **F15: Stationary Zero-Velocity Update** $\rightarrow$ ZUPT gate locks motion.
16. **F16: Resource Budget Violation** $\rightarrow$ Flagged and logged without crash.

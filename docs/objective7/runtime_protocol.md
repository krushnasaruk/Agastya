# Objective 7: Real-Time Runtime Protocol

## 1. Execution Sequence per Sensor Epoch

Every $100\text{ ms}$ ($10\text{ Hz}$), the `RealtimeNavigationEngine` executes:

1. **Timestamp & dt Sanity:** Asserts monotonic timestamp progression and bounded delta times ($0.001\text{s} \le \Delta t \le 1.0\text{s}$).
2. **Signal Filtering:** Validates individual wheel speeds and IMU axes against physical thresholds.
3. **Deterministic Physics Update:** Advances `ClassicalDeadReckoningEngine` (Baseline A) to maintain the baseline state.
4. **Causal Feature Extraction:** Calculates 16 causal kinematic features with backward derivatives.
5. **FIFO Window Rolling:** Shifts the sliding buffer queue ($W=10$).
6. **Supervised Forward Pass:** Starts `AIWatchdog` timer and runs `CausalResidualGRU` forward pass.
7. **Timeout Check:** If inference exceeds $25.0\text{ ms}$, aborts correction and commands fallback.
8. **Multi-Gate Policy:** Checks sensor health, ZUPT stationary state, OOD distance, temporal jumps, and predictive confidence.
9. **Kinematic State Integration:** Midpoint planar integration using corrected velocity (or classical fallback).
10. **Telemetry Dispatch:** Emits per-epoch telemetry with execution breakdown.

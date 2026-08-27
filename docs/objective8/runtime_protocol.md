# Project AGASTYA — Real-Time Runtime Protocol

## Per-Epoch Execution Pipeline
Each $10\text{ Hz}$ sensor tick executes in under $10\text{ ms}$:

1. **Sensor Validation (Stage 1):** Range and sanity filtering ($< 0.05\text{ ms}$).
2. **Classical Physics (Stage 2):** Deterministic dead-reckoning state integration ($< 0.10\text{ ms}$).
3. **Causal Feature Extraction (Stage 3):** 16-feature vector construction ($< 0.12\text{ ms}$).
4. **Window Buffer Shift (Stage 4):** 1.0s sliding causal buffer update ($< 0.02\text{ ms}$).
5. **Dynamic INT8 Inference (Stage 5):** Quantized forward pass ($< 4.0\text{ ms}$).
6. **AI Watchdog Supervision:** Terminates inference if $> 25.0\text{ ms}$.
7. **Selective Safety Gating (Stage 6):** OOD, confidence, temporal and stationary gating ($< 0.08\text{ ms}$).
8. **Kinematic State Integration (Stage 7):** Position update ($< 0.05\text{ ms}$).
9. **Numerical Stability Monitor (Stage 8):** State bounds verification ($< 0.02\text{ ms}$).
10. **Telemetry Logger (Stage 9):** Frame serialization ($< 0.02\text{ ms}$).

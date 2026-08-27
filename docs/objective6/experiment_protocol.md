# Objective 6: Experimental Protocol

## Master Benchmark Protocol (Experiments A through J)

1. **Experiment A (Classical Baseline):**
   - Pure deterministic dead reckoning using Objective 3 Baseline A.
   - Odometry: $\bar{v}_{\text{rear}} = \frac{v_{\text{RL}} + v_{\text{RR}}}{2}$.
   - Heading: $\psi_k = \psi_{k-1} + \omega_{z, k} \Delta t_k$.
   - AI usage: $0\%$.

2. **Experiment B (Objective 5 Velocity-Only Unconditional):**
   - Unconditionally adds $\delta v$ to $v_{\text{classical}}$ at every moving epoch ($W=10$).
   - Yaw residual disabled ($\delta \omega = 0$).

3. **Experiment C (Objective 6 Selective Velocity):**
   - Applies $\delta v$ only when Sensor Gate, Stationary Gate, OOD Gate, Temporal Consistency Gate, and Confidence Gate all return `PASS`.
   - Yaw residual disabled.

4. **Experiment D (Selective Gate Ablations D1–D6):**
   - Individually activates each gate to isolate its contribution to navigation accuracy and fallback rate.

5. **Experiment E (Yaw-Only Correction):**
   - Applies $\delta \omega_z$ while setting $\delta v = 0$ to demonstrate heading integration drift failure mode.

6. **Experiment F (Full Correction):**
   - Applies both $\delta v$ and $\delta \omega_z$.

7. **Experiment G (Extended GNSS Outage Benchmarks):**
   - Simulated outages at $t = 20.0\text{ s}$ for durations of $5\text{s}, 10\text{s}, 15\text{s}, 20\text{s}, 30\text{s}, 45\text{s}$.

8. **Experiment H (Maneuver-Stratified Breakdown):**
   - Stratifies timesteps into straight cruising, moderate turning, aggressive turning, acceleration, braking, slip, and stationary.

9. **Experiment I (AI Application & Fallback Telemetry):**
   - Quantifies overall AI acceptance rate and logs fallback reasons.

10. **Experiment J (Correction Quality & Calibration Analysis):**
    - Evaluates prediction errors across confidence bins.

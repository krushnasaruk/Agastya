# Deep Neural Residual Modeling & INT8 Quantization Specification

## 1. Rationale: Physics-Grounded AI Residual Estimation

Traditional black-box end-to-end neural navigation attempts to map raw sensor streams directly to global Cartesian coordinates. This naive approach suffers from catastrophic failure modes:
1. **Unbounded Error Growth**: Small network errors compound rapidly through quadratic integration ($O(t^2)$).
2. **Lack of Physical Interpretability**: Black-box models cannot guarantee physical plausibility or kinematic bounds.
3. **Catastrophic Out-of-Distribution Failure**: When encountering novel maneuvers, black-box networks drift uncontrollably.

### The AGASTYA Residual Paradigm
AGASTYA solves these limitations through **Physics-AI Residual Decomposition**:
$$\mathbf{x}_{\text{nav}, k} = \mathbf{f}_{\text{physics}}(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{h}_{\text{AI}}(\mathbf{u}_{k-W+1:k})$$

The deterministic physics engine (rear-axle kinematic odometry and trapezoidal gyro integration) guarantees baseline stability, while the causal recurrent neural network (`CausalResidualGRU`) estimates only the high-order residual discrepancy:
- $\delta v$: Forward velocity residual compensating for dynamic tire rolling radius compression and tire micro-slip.
- $\delta \omega$: Yaw rate residual compensating for chassis vibration and sensor temperature bias.

---

## 2. Canonical 16-Channel Causal Feature Registry

Every inference timestep consumes a 16-channel causal feature vector derived exclusively from onboard sensors without future lookahead:

| Channel | Feature Name | Mathematical Definition | Physical Purpose |
| :---: | :--- | :--- | :--- |
| `0` | `v_FL` | $v_{FL}$ (Front-Left Wheel Speed) | Front steering slip indicator |
| `1` | `v_FR` | $v_{FR}$ (Front-Right Wheel Speed) | Front steering slip indicator |
| `2` | `v_RL` | $v_{RL}$ (Rear-Left Wheel Speed) | Driven rear axle reference |
| `3` | `v_RR` | $v_{RR}$ (Rear-Right Wheel Speed) | Driven rear axle reference |
| `4` | `v_rear_mean` | $(v_{RL} + v_{RR}) / 2$ | Baseline forward speed |
| `5` | `v_diff_rear` | $v_{RR} - v_{RL}$ | Differential cornering speed |
| `6` | `v_diff_front_rear`| $(v_{FL} + v_{FR})/2 - (v_{RL} + v_{RR})/2$ | Longitudinal slip & weight transfer |
| `7` | `wheel_variance` | $\frac{1}{4} \sum_{i=1}^4 (v_i - \bar{v})^2$ | Multi-wheel traction asymmetry |
| `8` | `can_yaw_rate` | $\omega_{z, \text{CAN}}$ | Chassis rate gyroscope |
| `9` | `can_accel_x` | $a_{x, \text{CAN}}$ | Longitudinal chassis accelerometer |
| `10`| `accel_lat_est` | $v_{\text{rear}} \cdot \omega_{z, \text{CAN}}$ | Centrifugal lateral acceleration |
| `11`| `accel_fwd_diff` | $(v_{\text{rear}, k} - v_{\text{rear}, k-1}) / \Delta t$ | Numerical forward acceleration |
| `12`| `wheel_ratio_rl_rr`| $v_{RL} / (v_{RR} + \epsilon)$ | Cornering curvature signature |
| `13`| `wheel_ratio_f_r` | $\bar{v}_{\text{front}} / (\bar{v}_{\text{rear}} + \epsilon)$ | Understeer / oversteer indicator |
| `14`| `classical_v_fwd` | $v_{\text{class}}$ | Authoritative classical speed state |
| `15`| `dt` | $t_k - t_{k-1}$ | Microsecond sampling interval |

---

## 3. Neural Model Architecture: `CausalResidualGRU`

```
Input Tensor: [Batch, Window=10, Features=16]  (~1.0 second history at 10 Hz)
  │
  ├── Linear Input Projection Layer: Linear(16 → 64) + LeakyReLU(negative_slope=0.1)
  │
  ├── 1-Layer Causal GRU Core:
  │     Input Dim = 64, Hidden Dim = 64, Batch First = True
  │     Output Tensor: [Batch, Window=10, 64]
  │     Final Hidden Slice: h_T = Output[:, -1, :]  (Shape: [Batch, 64])
  │
  ├── Multi-Task Residual Regression Head:
  │     Linear(64 → 32) + LeakyReLU(negative_slope=0.1)
  │     Linear(32 → 2)
  │
  └── Residual Output Vector: [δv (m/s), δω (rad/s)]
```

- **Total Trainable Parameters**: `28,194`
- **Computational Footprint**: `< 0.06 MFLOPs` per forward pass.

---

## 4. Multi-Task Loss Formulation & Training Provenance

The loss objective balances robust magnitude regression with directional alignment:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Huber}}(\hat{\delta v}, \delta v^*) + \lambda_1 \mathcal{L}_{\text{Huber}}(\hat{\delta \omega}, \delta \omega^*) + \lambda_2 \mathcal{L}_{\text{cos}}(\hat{\mathbf{v}}, \mathbf{v}^*)$$

Where:
- **Huber Loss ($\delta = 1.0$)**: Prevents gradient explosions from dynamic tire scrub shocks during aggressive cornering.
- **Cosine Directional Loss**: Ensures predicted residual vectors align with ground-truth dynamic travel vectors.

### Strict Scientific Integrity & Normalization
1. **Train-Only Scaler**: Feature mean and variance vectors $(\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}})$ are fitted strictly on `sync_01` ($N=600$ epochs) and frozen in `artifacts/objective5/feature_scaler.json`.
2. **Disjoint Trajectory Splitting**:
   - **Training Set**: `sync_01` (600 epochs)
   - **Early-Stopping Validation**: `v_standalone_03` (450 epochs)
   - **Unseen Held-Out Test Set**: `sync_02` (899 epochs)

---

## 5. Selective Velocity vs Yaw Residual Analysis

Comprehensive ablation studies across 899 held-out epochs on `sync_02` yielded a critical scientific discovery:

| Configuration | ATE RMSE | Final Position Error | Heading RMSE | Drift % |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Baseline A (Objective 3)** | `1.6366 m` | `1.8270 m` | **`0.156°`** | `0.282%` |
| **Full Residual (Velocity + Yaw)** | `2.7557 m` | `3.6982 m` | `1.019°` | `0.571%` |
| **Velocity-Only Residual (Ablation B)** | **`1.5968 m`** | **`1.7903 m`** | **`0.156°`** | **`0.275%`** |
| **Objective 6 Selective Velocity** | **`1.6062 m`** | **`1.8013 m`** | **`0.156°`** | **`0.277%`** |

### Root Cause Analysis of Yaw Integration Drift:
While the GRU predicts instantaneous yaw rate residuals with high correlation ($r = +0.4935$), trapezoidal integration $\psi_k = \psi_0 + \sum \delta \omega_i \Delta t$ accumulates tiny residual sub-milli-radian biases over time. In contrast, the automotive chassis gyroscope already exhibits sub-degree accuracy. Therefore, Objectives 6, 7, and 8 enforce **velocity-only residual correction** (`enable_yaw_correction = False`), preserving flawless heading fidelity while reducing trajectory drift.

---

## 6. Dynamic INT8 Quantization (Objective 8)

To support deployment on ultra-low-power microcontrollers and embedded edge computers, AGASTYA includes dynamic post-training INT8 quantization.

### 6.1 Quantization Specification
- **Engine**: PyTorch Dynamic Quantization (`torch.quantization.quantize_dynamic`).
- **Target Layers**: `nn.Linear`, `nn.GRU`.
- **Target Dtype**: `torch.qint8` (8-bit signed integer weights with dynamic FP32 activation scale).

### 6.2 Compression & Precision Metrics
- **FP32 Model Disk Footprint**: `115.9 KB`
- **INT8 Model Disk Footprint**: `35.7 KB` (**69.2% Storage Reduction**)
- **Residual Output Discrepancy**: $\text{MAE} = 0.00838\text{ m/s}$ (Sub-centimeter/sec scale)
- **Closed-Loop Trajectory Fidelity**: Zero regression on held-out test trajectories ($\text{ATE RMSE} = 1.4237\text{ m}$ vs FP32 golden reference).

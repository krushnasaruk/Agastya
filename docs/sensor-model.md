# Sensor Stochastic Modeling & Error Characteristics

## 1. Inertial Measurement Unit (6-DOF IMU)

### 1.1 Accelerometer Model
The measurement model for a tri-axial accelerometer is:
$$\tilde{\mathbf{f}}^b = (\mathbf{I} + \mathbf{S}_a) \mathbf{f}^b + \mathbf{b}_a + \mathbf{n}_a + \mathbf{w}_a$$

Where:
- $\mathbf{f}^b$: True specific force in body frame ($\text{m/s}^2$).
- $\mathbf{S}_a$: Scale factor and cross-axis misalignments matrix ($3\times3$).
- $\mathbf{b}_a$: Accelerometer bias instability / random walk ($\text{m/s}^2$).
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: High-frequency white Gaussian noise.
- $\mathbf{w}_a$: Dynamic vibration / engine noise.

### 1.2 Gyroscope Model
The measurement model for a tri-axial rate gyroscope is:
$$\tilde{\boldsymbol{\omega}}_{ib}^b = (\mathbf{I} + \mathbf{S}_g) \boldsymbol{\omega}_{ib}^b + \mathbf{b}_g + \mathbf{n}_g$$

Where:
- $\boldsymbol{\omega}_{ib}^b$: True angular rate ($\text{rad/s}$).
- $\mathbf{b}_g$: Gyroscope in-run bias drift ($\text{rad/s}$).
- $\mathbf{n}_g \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I})$: Angular Random Walk (ARW) white noise.

### 1.3 Allan Variance Bias Dynamics
Bias drift follows a first-order Gauss-Markov stochastic process:
$$\dot{\mathbf{b}}_a = -\frac{1}{\tau_a} \mathbf{b}_a + \boldsymbol{\eta}_{ba}, \quad \boldsymbol{\eta}_{ba} \sim \mathcal{N}(\mathbf{0}, q_{ba} \mathbf{I})$$
$$\dot{\mathbf{b}}_g = -\frac{1}{\tau_g} \mathbf{b}_g + \boldsymbol{\eta}_{bg}, \quad \boldsymbol{\eta}_{bg} \sim \mathcal{N}(\mathbf{0}, q_{bg} \mathbf{I})$$

---

## 2. Global Navigation Satellite System (GNSS)

### 2.1 Pseudorange & Geometric Dilution of Precision (GDOP)
$$\tilde{\mathbf{p}}_{gnss}^n = \mathbf{p}^n + \boldsymbol{\epsilon}_{mp} + \boldsymbol{\epsilon}_{iono} + \boldsymbol{\epsilon}_{tropo} + \mathbf{n}_{gnss}$$

$$\mathbf{R}_{gnss} = \text{diag}\left( \sigma_{h}^2 \cdot \text{HDOP}^2, \; \sigma_{h}^2 \cdot \text{HDOP}^2, \; \sigma_{v}^2 \cdot \text{VDOP}^2 \right)$$

### 2.2 Degradation Modes
1. **Clear Sky (Nominal)**: $\text{HDOP} < 1.5$, $\sigma_h = 1.2\,\text{m}$, 12+ Satellites.
2. **GPS Noise / Multipath**: $\text{HDOP} \approx 3.5 - 6.0$, $\sigma_h = 8.5\,\text{m}$, reflection jumps.
3. **Urban Canyon**: High building reflections, intermittent satellite occlusion, severe multipath biases $\boldsymbol{\epsilon}_{mp} \sim 15 - 30\,\text{m}$.
4. **GPS Loss (Electronic Warfare / Tunnel)**: Satellites = 0, fix valid = False.

---

## 3. Visual Odometry (Camera)

$$\tilde{\mathbf{v}}_{vo}^b = \mathbf{v}^b + \mathbf{n}_{vo}, \quad \mathbf{n}_{vo} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{vo})$$
$$\mathbf{R}_{vo} = \frac{\sigma_{vo}^2}{c_{confidence}} \mathbf{I}_{3\times3}$$

Where $c_{confidence} \in (0, 1]$ represents feature matching inlier ratio.

# Sensor Stochastic Modeling & Error Characteristics

## 1. Inertial Measurement Unit (6-DOF IMU)

Project AGASTYA models high-rate (100–200 Hz) industrial and automotive-grade 6-DOF MEMS Inertial Measurement Units operating in dynamic vehicle environments.

### 1.1 Tri-Axial Accelerometer Model
The measurement model for body-frame specific force $\tilde{\mathbf{f}}^b$ is:
$$\tilde{\mathbf{f}}^b = (\mathbf{I}_{3\times3} + \mathbf{S}_a + \mathbf{M}_a) \mathbf{f}^b + \mathbf{b}_a(t) + \mathbf{n}_a(t) + \mathbf{w}_{\text{vib}}(t)$$

Where:
- $\mathbf{f}^b \in \mathbb{R}^3$: True specific force in the body reference frame ($\text{m/s}^2$).
- $\mathbf{S}_a = \text{diag}(s_{ax}, s_{ay}, s_{az})$: Diagonal scale factor error matrix (typically $\pm 500\text{ ppm}$).
- $\mathbf{M}_a$: Non-orthogonal cross-axis misalignment matrix ($3\times3$).
- $\mathbf{b}_a(t) \in \mathbb{R}^3$: Accelerometer bias instability / stochastic in-run drift ($\text{m/s}^2$).
- $\mathbf{n}_a(t) \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I}_{3\times3})$: High-frequency Velocity Random Walk (VRW) white Gaussian noise ($\sigma_a \approx 0.05 \text{ m/s}/\sqrt{\text{s}}$).
- $\mathbf{w}_{\text{vib}}(t)$: Engine combustion and road vibration harmonics.

### 1.2 Tri-Axial Rate Gyroscope Model
The measurement model for body-frame angular rates $\tilde{\boldsymbol{\omega}}_{ib}^b$ is:
$$\tilde{\boldsymbol{\omega}}_{ib}^b = (\mathbf{I}_{3\times3} + \mathbf{S}_g + \mathbf{M}_g) \boldsymbol{\omega}_{ib}^b + \mathbf{b}_g(t) + \mathbf{n}_g(t) + \mathbf{G}_g \mathbf{f}^b$$

Where:
- $\boldsymbol{\omega}_{ib}^b \in \mathbb{R}^3$: True vehicle angular rate with respect to inertial space ($\text{rad/s}$).
- $\mathbf{S}_g, \mathbf{M}_g$: Gyroscope scale factor and cross-axis alignment errors.
- $\mathbf{b}_g(t) \in \mathbb{R}^3$: In-run gyroscope bias drift ($\text{rad/s}$).
- $\mathbf{n}_g(t) \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I}_{3\times3})$: Angular Random Walk (ARW) broadband white noise ($\sigma_g \approx 0.005 \text{ rad/s}/\sqrt{\text{s}}$).
- $\mathbf{G}_g \mathbf{f}^b$: $g$-sensitivity acceleration-induced angular rate bias matrix.

### 1.3 Allan Variance Bias Dynamics (1st-Order Gauss-Markov)
In-run bias drifts $\mathbf{b}_a(t)$ and $\mathbf{b}_g(t)$ are modeled as first-order Gauss-Markov (FOGM) stochastic processes characterized by correlation times $\tau_a, \tau_g$:
$$\dot{\mathbf{b}}_a(t) = -\frac{1}{\tau_a} \mathbf{b}_a(t) + \boldsymbol{\eta}_{ba}(t), \quad \boldsymbol{\eta}_{ba}(t) \sim \mathcal{N}(\mathbf{0}, q_{ba} \mathbf{I}_{3\times3})$$
$$\dot{\mathbf{b}}_g(t) = -\frac{1}{\tau_g} \mathbf{b}_g(t) + \boldsymbol{\eta}_{bg}(t), \quad \boldsymbol{\eta}_{bg}(t) \sim \mathcal{N}(\mathbf{0}, q_{bg} \mathbf{I}_{3\times3})$$

Where $q_{ba} = \frac{2 \sigma_{ba}^2}{\tau_a}$ and $q_{bg} = \frac{2 \sigma_{bg}^2}{\tau_g}$ are derived from Allan Variance empirical log-log curves.

---

## 2. CAN-Bus 4-Wheel Speed Odometry & Wheel Slip Dynamics

Automotive wheel encoders measure rotational speed of each wheel ($\omega_i$, $i \in \{FL, FR, RL, RR\}$) sampled via CAN-bus at 10–50 Hz.

### 2.1 Forward Velocity & Tire Dynamic Rolling Radius
Raw wheel linear velocity is:
$$v_i = \omega_i \cdot r_{\text{eff}, i}$$

The effective rolling radius $r_{\text{eff}}$ fluctuates dynamically as a function of tire pressure, vehicle mass transfer, and aerodynamic downforce:
$$r_{\text{eff}} = r_0 \left( 1 - c_{\text{pitch}} \sin \theta - c_{\text{roll}} \sin \phi - c_{\text{accel}} \frac{a_x}{g} \right)$$

### 2.2 Longitudinal Slip Ratio
During acceleration and braking, longitudinal tire micro-slip causes discrepancy between wheel peripheral speed $\omega r_{\text{eff}}$ and actual vehicle chassis forward speed $v_x$:
$$s_{\text{long}} = \frac{\omega r_{\text{eff}} - v_x}{\max(|v_x|, \epsilon)}$$

AGASTYA's `CausalResidualGRU` explicitly consumes the 4 individual wheel speeds alongside IMU acceleration to regress this non-linear slip residual $\delta v$.

---

## 3. Global Navigation Satellite System (GNSS)

### 3.1 Pseudorange & Geometric Dilution of Precision (GDOP)
$$\tilde{\mathbf{p}}_{gnss}^n = \mathbf{p}^n + \boldsymbol{\epsilon}_{mp} + \boldsymbol{\epsilon}_{iono} + \boldsymbol{\epsilon}_{tropo} + \mathbf{n}_{gnss}$$

The GNSS measurement noise covariance $\mathbf{R}_{gnss}$ scales dynamically with satellite geometry Dilution of Precision (HDOP/VDOP):
$$\mathbf{R}_{gnss} = \begin{bmatrix}
\sigma_h^2 \cdot \text{HDOP}^2 & 0 & 0 \\
0 & \sigma_h^2 \cdot \text{HDOP}^2 & 0 \\
0 & 0 & \sigma_v^2 \cdot \text{VDOP}^2
\end{bmatrix}$$

### 3.2 Operational Degradation Modes
1. **Open Sky (Nominal Fix)**: $\text{HDOP} < 1.2$, Satellites $\ge 10$, $\sigma_h = 1.2\text{ m}$. Full RTK or differential GNSS valid.
2. **Multipath & Urban Canyons**: Signal reflections from skyscrapers create pseudorange jumps $\boldsymbol{\epsilon}_{mp} \sim 10 - 30\text{ m}$ and inflated $\text{HDOP} > 4.0$.
3. **Electronic Warfare Jamming & Denial (Complete Outage)**: Satellites drop to 0, fix valid flag resets to `False`. The system transitions instantaneously into autonomous dead-reckoning mode.

---

## 4. Visual Odometry & Optical Flow

For aerial or camera-equipped ground vehicles, visual odometry supplies relative body-frame displacement and velocity:
$$\tilde{\mathbf{v}}_{vo}^b = \mathbf{v}^b + \mathbf{n}_{vo}, \quad \mathbf{n}_{vo} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{vo})$$
$$\mathbf{R}_{vo} = \frac{\sigma_{vo}^2}{c_{\text{inlier}}} \mathbf{I}_{3\times3}$$

Where $c_{\text{inlier}} \in (0, 1]$ represents the feature point matching inlier ratio (e.g., ORB-SLAM / FAST inliers). When $c_{\text{inlier}} < 0.20$ (such as during camera occlusion, dust, or darkness), the VO measurement is rejected by the sensor quality gate.

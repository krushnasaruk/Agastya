# Mathematical Formulation of SINS, Odometry Kinematics & Error-State Kalman Filtering

## 1. Coordinate Reference Frames

Project AGASTYA utilizes standard aerospace and automotive right-handed orthogonal coordinate frames:

1. **Inertial Frame ($i$-frame)**: Non-accelerating, non-rotating reference origin fixed at Earth's center of mass.
2. **Earth-Centered Earth-Fixed Frame ($e$-frame / ECEF)**: Origin at Earth's center of mass, rotating with the Earth at constant angular rate $\boldsymbol{\omega}_{ie} = [0, 0, 7.292115 \times 10^{-5}]^T \text{ rad/s}$.
3. **Local Navigation Tangent Frame ($n$-frame / NED / Local ENU)**:
   - **NED**: $X$-axis pointing True North, $Y$-axis pointing East, $Z$-axis pointing Downward along the local gravity vector.
   - **Local ENU**: $X$-axis pointing East, $Y$-axis pointing North, $Z$-axis pointing Upward.
4. **Body Reference Frame ($b$-frame)**: Rigidly attached to the vehicle center of gravity:
   - $X_b$: Longitudinal Forward axis along the vehicle centerline.
   - $Y_b$: Transverse Rightward lateral axis.
   - $Z_b$: Normal Downward vertical axis.

---

## 2. Strapdown Inertial Navigation System (SINS) Mechanization

### 2.1 Attitude Representation & Quaternion Propagation
Vehicle spatial orientation from body frame $b$ to navigation frame $n$ is parameterized by a normalized Hamiltonian unit quaternion:
$$\mathbf{q}_b^n = \begin{bmatrix} q_w & q_x & q_y & q_z \end{bmatrix}^T = \begin{bmatrix} q_w & \mathbf{q}_v^T \end{bmatrix}^T, \quad \|\mathbf{q}_b^n\|_2 = 1$$

The continuous kinematic rate equation governing quaternion evolution is:
$$\dot{\mathbf{q}}_b^n = \frac{1}{2} \mathbf{q}_b^n \otimes \boldsymbol{\Omega}_{nb}^b$$

Where:
- $\otimes$ denotes quaternion multiplication.
- $\boldsymbol{\Omega}_{nb}^b = \begin{bmatrix} 0 & (\boldsymbol{\omega}_{nb}^b)^T \end{bmatrix}^T \in \mathbb{R}^4$.
- $\boldsymbol{\omega}_{nb}^b = \tilde{\boldsymbol{\omega}}_{ib}^b - \mathbf{b}_g - \boldsymbol{\eta}_g - \mathbf{C}_n^b (\boldsymbol{\omega}_{ie}^n + \boldsymbol{\omega}_{en}^n)$ is the true vehicle angular rate with respect to the navigation frame.

For high-rate numerical propagation over sample period $\Delta t = t_{k+1} - t_k$, a 4th-Order Runge-Kutta (RK4) integrator computes the rotation incremental quaternion:
$$\Delta \boldsymbol{\theta}_k = \int_{t_k}^{t_{k+1}} \boldsymbol{\omega}_{nb}^b(\tau) \, d\tau \approx \boldsymbol{\omega}_k \Delta t$$
$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \begin{bmatrix} \cos\left(\frac{\|\Delta \boldsymbol{\theta}_k\|}{2}\right) \\ \frac{\Delta \boldsymbol{\theta}_k}{\|\Delta \boldsymbol{\theta}_k\|} \sin\left(\frac{\|\Delta \boldsymbol{\theta}_k\|}{2}\right) \end{bmatrix}, \quad \mathbf{q}_{k+1} \leftarrow \frac{\mathbf{q}_{k+1}}{\|\mathbf{q}_{k+1}\|}$$

### 2.2 Specific Force & Velocity Propagation
Specific force measured by the tri-axial accelerometer in the body frame $\tilde{\mathbf{f}}^b$ is rotated to the local navigation frame:
$$\dot{\mathbf{v}}^n = \mathbf{C}_b^n(\mathbf{q}) \left( \tilde{\mathbf{f}}^b - \mathbf{b}_a - \boldsymbol{\eta}_a \right) + \mathbf{g}^n - (2 \boldsymbol{\omega}_{ie}^n + \boldsymbol{\omega}_{en}^n) \times \mathbf{v}^n$$
$$\dot{\mathbf{p}}^n = \mathbf{v}^n$$

Where:
- $\mathbf{C}_b^n(\mathbf{q}) = \mathbf{I}_{3\times3} - 2 q_w [\mathbf{q}_v]_\times + 2 [\mathbf{q}_v]_\times^2$ is the Direction Cosine Matrix (DCM).
- $[\mathbf{a}]_\times$ is the skew-symmetric cross-product matrix of $\mathbf{a} = [a_1, a_2, a_3]^T$:
  $$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix}$$
- $\mathbf{g}^n = [0, 0, g_0]^T$ is the local normal gravity vector ($g_0 = 9.80665 \text{ m/s}^2$).

---

## 3. Planar Wheel Odometry & Midpoint Heading Integration

For ground vehicles equipped with CAN-bus wheel speed encoders (such as the IO-VNBD dataset platform), AGASTYA provides a deterministic classical baseline based on rear-axle kinematics.

### 3.1 Differential Wheel Speed Kinematics
Given the 4 onboard wheel rotational velocities $[v_{FL}, v_{FR}, v_{RL}, v_{RR}]$:
1. **Longitudinal Forward Speed**:
   $$v_{\text{fwd}, k} = \frac{v_{RL, k} + v_{RR, k}}{2}$$
2. **Differential Odometry Yaw Rate**:
   $$\omega_{z, \text{wheel}, k} = \frac{v_{RR, k} - v_{RL, k}}{L_{\text{track}}}$$
   Where $L_{\text{track}} = 1.540\text{ m}$ is the vehicle rear track width.

### 3.2 Midpoint Heading Integration
Using calibrated chassis rate gyroscopes $\omega_{z, \text{gyro}}$, heading angle $\psi_k$ is updated via trapezoidal integration:
$$\psi_k = \psi_{k-1} + \frac{\omega_{z, k-1} + \omega_{z, k}}{2} \Delta t_k$$

### 3.3 Metric Local ENU Dead-Reckoning
Position increments in the local tangent East-North-Up plane are integrated:
$$\Delta E_k = v_{\text{fwd}, k} \cdot \cos\left( \frac{\psi_{k-1} + \psi_k}{2} \right) \Delta t_k$$
$$\Delta N_k = v_{\text{fwd}, k} \cdot \sin\left( \frac{\psi_{k-1} + \psi_k}{2} \right) \Delta t_k$$
$$p_{E, k} = p_{E, k-1} + \Delta E_k, \quad p_{N, k} = p_{N, k-1} + \Delta N_k$$

---

## 4. 15-State Error-State Extended Kalman Filter (ES-EKF)

AGASTYA employs an indirect Error-State EKF formulation where the nominal state $\mathbf{x}$ is propagated nonlinearly at high frequency, while a 15-dimensional small-signal error state $\delta \mathbf{x}$ is estimated to correct deviations.

### 4.1 State Definitions
- **True State**: $\mathbf{x}_{\text{true}} = [\mathbf{p}^n, \mathbf{v}^n, \mathbf{q}_b^n, \mathbf{b}_a, \mathbf{b}_g]^T$
- **Nominal State**: $\hat{\mathbf{x}} = [\hat{\mathbf{p}}^n, \hat{\mathbf{v}}^n, \hat{\mathbf{q}}_b^n, \hat{\mathbf{b}}_a, \hat{\mathbf{b}}_g]^T$
- **Error State Vector**:
  $$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n \\ \delta \mathbf{v}^n \\ \delta \boldsymbol{\theta}^n \\ \delta \mathbf{b}_a \\ \delta \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

### 4.2 Linearized Error Dynamic System Matrix $\mathbf{F}$
$$\delta \dot{\mathbf{x}}(t) = \mathbf{F}(t) \delta \mathbf{x}(t) + \mathbf{G}(t) \mathbf{w}(t)$$

$$\mathbf{F} = \begin{bmatrix}
\mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -[\mathbf{C}_b^n \hat{\mathbf{f}}^b]_\times & -\mathbf{C}_b^n & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -[\boldsymbol{\omega}_{in}^n]_\times & \mathbf{0}_{3\times3} & -\mathbf{C}_b^n \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -\frac{1}{\tau_a}\mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -\frac{1}{\tau_g}\mathbf{I}_{3\times3}
\end{bmatrix}$$

$$\mathbf{G} = \begin{bmatrix}
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
-\mathbf{C}_b^n & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & -\mathbf{C}_b^n & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{I}_{3\times3}
\end{bmatrix}$$

### 4.3 Discrete Covariance Propagation
Using Taylor series matrix exponential expansion:
$$\boldsymbol{\Phi}_k = \exp(\mathbf{F} \Delta t) \approx \mathbf{I}_{15\times15} + \mathbf{F} \Delta t + \frac{1}{2} \mathbf{F}^2 \Delta t^2$$
$$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{Q}_k$$

Where discrete process noise covariance $\mathbf{Q}_k$ is computed via:
$$\mathbf{Q}_k \approx \mathbf{G}_k \mathbf{Q}_c \mathbf{G}_k^T \Delta t, \quad \mathbf{Q}_c = \text{diag}(\sigma_{acc}^2 \mathbf{I}_3, \sigma_{gyr}^2 \mathbf{I}_3, q_{ba} \mathbf{I}_3, q_{bg} \mathbf{I}_3)$$

### 4.4 Multi-Sensor Measurement Update & Joseph-Form Stabilization
When aiding measurements $\mathbf{z}_k$ (GNSS position, visual odometry velocity, or AI-predicted velocity residual) arrive:
1. **Innovation Residual**:
   $$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$$
2. **Innovation Covariance**:
   $$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k$$
3. **Optimal Kalman Gain**:
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}$$
4. **Error State Correction**:
   $$\delta \hat{\mathbf{x}}_k = \mathbf{K}_k \mathbf{y}_k$$
5. **Joseph-Form Covariance Update** (ensures symmetry and positive definiteness under finite precision):
   $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

### 4.5 State Injection & Error Reset
Corrections are applied back into the nominal state:
$$\hat{\mathbf{p}}^n \leftarrow \hat{\mathbf{p}}^n + \delta \hat{\mathbf{p}}^n$$
$$\hat{\mathbf{v}}^n \leftarrow \hat{\mathbf{v}}^n + \delta \hat{\mathbf{v}}^n$$
$$\hat{\mathbf{q}}_b^n \leftarrow \hat{\mathbf{q}}_b^n \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \hat{\boldsymbol{\theta}}^n \end{bmatrix}, \quad \hat{\mathbf{q}}_b^n \leftarrow \frac{\hat{\mathbf{q}}_b^n}{\|\hat{\mathbf{q}}_b^n\|}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta \hat{\mathbf{b}}_a, \quad \hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta \hat{\mathbf{b}}_g$$
$$\delta \hat{\mathbf{x}} \leftarrow \mathbf{0}_{15\times1}$$

---

## 5. Zero-Velocity Update (ZUPT) & Stationary Energy Detection

To eliminate unbounded drift when the vehicle is stationary (e.g., at traffic signals or staging points), AGASTYA implements a Generalized Likelihood Ratio Test (GLRT) energy detector across a sliding window of $N=10$ samples:

$$T_k = \frac{1}{\sigma_a^2 N} \sum_{j=k-N+1}^k \|\mathbf{f}_j - \bar{\mathbf{f}}\|^2 + \frac{1}{\sigma_g^2 N} \sum_{j=k-N+1}^k \|\boldsymbol{\omega}_j\|^2$$

$$\text{ZUPT State} = \begin{cases} \text{STATIONARY (True)} & \text{if } T_k < \gamma_{th} \text{ and } |v_{\text{wheel}}| < 0.05 \text{ m/s} \\ \text{MOVING (False)} & \text{otherwise} \end{cases}$$

When stationary, velocity is clamped to $\mathbf{0}$, position integration is locked, and measurement update $\mathbf{z}_{zupt} = [0, 0, 0]^T$ with observation matrix $\mathbf{H}_{zupt} = [\mathbf{0}_{3\times3}, \mathbf{I}_{3\times3}, \mathbf{0}_{3\times9}]$ directly corrects accelerometer and gyroscope bias estimates.

# Mathematical Formulation of SINS & Error-State Kalman Filtering

## 1. Coordinate Frames

1. **Inertial Frame ($i$-frame)**: Non-accelerating origin at Earth's center.
2. **Earth Frame ($e$-frame / ECEF)**: Origin at Earth's center, rotating with Earth at angular velocity $\boldsymbol{\omega}_{ie}$.
3. **Navigation Frame ($n$-frame / NED)**: Local tangent plane oriented North-East-Down.
4. **Body Frame ($b$-frame)**: Fixed to vehicle sensor center (Forward-Right-Down).

---

## 2. Strapdown Inertial Navigation System (SINS) Mechanization

### 2.1 Attitude Propagation
Using unit quaternion representation $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$:
$$\dot{\mathbf{q}}_b^n = \frac{1}{2} \mathbf{q}_b^n \otimes \boldsymbol{\Omega}_b$$
Where $\boldsymbol{\Omega}_b = [0, \boldsymbol{\omega}_{nb}^b]^T$ and $\boldsymbol{\omega}_{nb}^b = \tilde{\boldsymbol{\omega}}_{ib}^b - \mathbf{b}_g - \boldsymbol{\eta}_g$.

Using 4th-Order Runge-Kutta (RK4) integration over $\Delta t$:
$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \exp\left(\frac{1}{2} \boldsymbol{\omega}_k \Delta t\right)$$

### 2.2 Velocity & Position Propagation
$$\dot{\mathbf{v}}^n = \mathbf{C}_b^n (\tilde{\mathbf{f}}^b - \mathbf{b}_a - \boldsymbol{\eta}_a) + \mathbf{g}^n - (2 \boldsymbol{\omega}_{ie}^n + \boldsymbol{\omega}_{en}^n) \times \mathbf{v}^n$$
$$\dot{\mathbf{p}}^n = \mathbf{v}^n$$

Where:
- $\mathbf{C}_b^n = \mathbf{R}(\mathbf{q}_b^n)$ is the rotation matrix from Body to Navigation frame.
- $\tilde{\mathbf{f}}^b$ is specific force measured by the 3-axis accelerometer.
- $\mathbf{g}^n = [0, 0, g_0]^T$ is the local gravity vector ($g_0 \approx 9.80665 \, \text{m/s}^2$).

---

## 3. 15-State Error-State Extended Kalman Filter (ES-EKF)

The total error state vector in tangent space is:
$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n \\ \delta \mathbf{v}^n \\ \delta \boldsymbol{\theta}^n \\ \delta \mathbf{b}_a \\ \delta \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

### 3.1 Error Kinematics System Matrix $\mathbf{F}$
$$\delta \dot{\mathbf{x}} = \mathbf{F} \delta \mathbf{x} + \mathbf{G} \mathbf{w}$$

$$\mathbf{F} = \begin{bmatrix}
\mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -[\mathbf{C}_b^n \hat{\mathbf{f}}^b]_\times & -\mathbf{C}_b^n & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -\mathbf{C}_b^n \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -\frac{1}{\tau_a}\mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & -\frac{1}{\tau_g}\mathbf{I}_{3\times3}
\end{bmatrix}$$

Where $[\mathbf{a}]_\times$ is the skew-symmetric cross-product matrix of vector $\mathbf{a}$.

### 3.2 Discrete Covariance Propagation
$$\boldsymbol{\Phi}_k = \exp(\mathbf{F} \Delta t) \approx \mathbf{I} + \mathbf{F} \Delta t + \frac{1}{2} \mathbf{F}^2 \Delta t^2$$
$$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{Q}_k$$

Where process noise covariance $\mathbf{Q}_k = \mathbf{G} \mathbf{Q}_c \mathbf{G}^T \Delta t$.

### 3.3 Measurement Update (GNSS / Visual Odometry / AI Velocity)
For a position and velocity measurement $\mathbf{z}_{gnss} = [\mathbf{p}_{meas}, \mathbf{v}_{meas}]^T$:
$$\mathbf{H} = \begin{bmatrix} \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \\ \mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} & \mathbf{0}_{3\times3} \end{bmatrix}$$

1. **Innovation Residual**:
   $$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$$
2. **Innovation Covariance**:
   $$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k$$
3. **Kalman Gain**:
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}$$
4. **Error State Update**:
   $$\delta \hat{\mathbf{x}}_k = \mathbf{K}_k \mathbf{y}_k$$
5. **Joseph-Form Covariance Update**:
   $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

### 3.4 State Injection & Reset
$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta \hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta \hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \hat{\boldsymbol{\theta}} \end{bmatrix}, \quad \hat{\mathbf{q}} \leftarrow \frac{\hat{\mathbf{q}}}{\|\hat{\mathbf{q}}\|}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta \hat{\mathbf{b}}_a, \quad \hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta \hat{\mathbf{b}}_g$$
$$\delta \hat{\mathbf{x}} \leftarrow \mathbf{0}_{15\times1}$$

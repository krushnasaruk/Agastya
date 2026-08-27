# Deep Neural Inertial Navigation Models

## 1. Motivation
Classical double integration of IMU measurements causes position error to grow with $O(t^2)$ for accelerometer bias and $O(t^3)$ for gyroscope drift. Neural Inertial Navigation learns to map short sequential windows of IMU measurements directly to body-frame velocity vectors:
$$f_\theta: \left( \tilde{\mathbf{f}}_{t-W:t}^b, \tilde{\boldsymbol{\omega}}_{t-W:t}^b \right) \longrightarrow \left( \hat{\mathbf{v}}_t^b, \hat{\mathbf{b}}_a, \hat{\mathbf{b}}_g \right)$$

Integrating predicted velocity results in linear error growth $O(t)$ rather than cubic $O(t^3)$, reducing drift by 95%+.

---

## 2. Model Architectures

### 2.1 Bidirectional LSTM with Temporal Attention (`BiLSTMDeadReckoning`)

```
Input: (Batch, Window=100, Channels=6)
  │
  ├── 1D Conv Layer (Kernel=5, Channels=64) + LeakyReLU
  ├── 1D Conv Layer (Kernel=3, Channels=128) + LeakyReLU
  │
  ├── 2-Layer Bidirectional LSTM (Hidden Dim = 128)
  │     Outputs: (Batch, Window=100, 256)
  │
  ├── Self-Attention Pooling Layer
  │     Computes attention weights α_t over the temporal window
  │     Context Vector c = ∑ α_t · h_t
  │
  ├── Fully Connected Regression Head (256 -> 128 -> 64 -> 3)
  └── Velocity Vector Output: [Δvx, Δvy, Δvz]
```

### 2.2 Temporal Inertial Transformer (`InertialTransformer`)

- **Input Embedding**: Linear projection to dimension $d_{model} = 128$.
- **Positional Encoding**: Learnable 1D temporal sinusoidal positional embeddings.
- **Transformer Encoder**: 4 Encoder Layers, each with:
  - 8-Head Multi-Head Self-Attention ($\text{MHSA}$)
  - Feed-Forward Network ($d_{ff} = 512$) with GeLU activation
  - Pre-LayerNorm & Residual Skip Connections
  - Dropout ($p = 0.1$)
- **Output Head**: Global Average Pooling + MLP predicting 3D velocity and accelerometer bias residuals.

---

## 3. Loss Functions & Training Objectives

The total loss $\mathcal{L}_{total}$ balances magnitude accuracy, directional alignment, and smoothness:

$$\mathcal{L}_{total} = \mathcal{L}_{Huber}(\hat{\mathbf{v}}, \mathbf{v}^*) + \lambda_1 \mathcal{L}_{cos}(\hat{\mathbf{v}}, \mathbf{v}^*) + \lambda_2 \|\hat{\mathbf{b}}_a - \mathbf{b}_a^*\|_2^2$$

Where:
- **Huber Loss**: Robust to high-acceleration outliers / mechanical shocks:
  $$\mathcal{L}_{Huber}(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
- **Cosine Directional Loss**: Ensures predicted velocity aligns with true motion vector:
  $$\mathcal{L}_{cos}(\hat{\mathbf{v}}, \mathbf{v}^*) = 1 - \frac{\hat{\mathbf{v}} \cdot \mathbf{v}^*}{\|\hat{\mathbf{v}}\| \|\mathbf{v}^*\| + \epsilon}$$

# Project AGASTYA — Dynamic INT8 Quantization Protocol

## Mathematical Formulation
Dynamic INT8 quantization converts 32-bit floating point matrix weights $W_{\text{FP32}}$ to signed 8-bit integer representations $W_{\text{INT8}}$:

$$W_{\text{INT8}} = \text{clamp}\left(\left\lfloor \frac{W_{\text{FP32}}}{S} + Z \right\rceil, -128, 127\right)$$

where:
- $S = \frac{\max(W) - \min(W)}{255}$ is the quantization scale factor.
- $Z = \text{round}\left(-\frac{\min(W)}{S}\right) - 128$ is the zero-point offset.

## Target Layers
1. Input Projection: `Linear(16 -> 64)`
2. Recurrent Core: `GRU(input_size=64, hidden_size=64, num_layers=2)`
3. Output Head: `Linear(64 -> 2)`

## Verification Criteria
- Residual Output MAE vs FP32: $< 0.05\text{ m/s}$ (Achieved: $0.00838\text{ m/s}$)
- Outlier Rate: $0.0\%$ beyond tolerance
- Serialized Compression Ratio: $\ge 2.5\times$ (Achieved: $3.25\times$, $69.2\%$ reduction)

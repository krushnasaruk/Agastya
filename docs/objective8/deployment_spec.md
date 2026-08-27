# Project AGASTYA — Objective 8 Deployment Specification

## Target Architecture & System Constraints
- **Primary Platform:** Embedded CPU / Microcontroller Edge Unit
- **Execution Target:** PyTorch CPU Runtime / C++ Inference Wrapper
- **Model Topology:** `CausalResidualGRU` (Input: `[B, 10, 16]`, Hidden: 64, Layers: 2, Output: 2)
- **Quantization:** Dynamic INT8 (`torch.qint8` on `nn.Linear`, `nn.GRU`)
- **Sampling Frequency:** $10\text{ Hz}$ Nominal ($\Delta t = 100.0\text{ ms}$)
- **Real-Time Deadlines:**
  - Hard Deadline: $100.0\text{ ms}$
  - Target Budget: $50.0\text{ ms}$
  - AI Watchdog Budget: $25.0\text{ ms}$
- **RAM Ceiling:** $25.0\text{ MB}$ (Operational Peak: $3.43\text{ MB}$)
- **Degradation Authority:** Instantaneous fallback to Objective 3 Baseline A Classical Physics Engine

# Project AGASTYA Documentation Index & Master Sitemap

Welcome to the **Project AGASTYA (SIH26168)** Documentation Hub. Below is the complete catalog of architectural specifications, mathematical formulations, neural network design reports, real-time protocols, and edge deployment guides.

---

## 🏛️ Core Architecture & Math Foundations

| Document | Description | Key Focus Areas |
| :--- | :--- | :--- |
| **[System Architecture](architecture.md)** | End-to-end multi-rate dataflow and component breakdown | 6-stage pipeline, multi-frequency fusion, microsecond latency budget |
| **[Navigation Formulation](navigation.md)** | Rigorous mathematical equations of SINS & Kalman filtering | RK4 quaternion propagation, 15-state ES-EKF, Joseph-form covariance, GLRT ZUPT |
| **[Sensor Stochastic Modeling](sensor-model.md)** | IMU, CAN wheel odometry, GNSS, and visual odometry error models | Allan variance, 1st-order Gauss-Markov bias, tire rolling radius, HDOP scaling |
| **[Deep Neural Models](ml-model.md)** | Neural residual learning and INT8 model quantization | 16-channel feature registry, CausalResidualGRU, Huber loss, INT8 dynamic quantization |

---

## 🚀 Deployment, API & Developer Manuals

| Document | Description | Key Focus Areas |
| :--- | :--- | :--- |
| **[REST & WebSocket API](api.md)** | REST endpoints and 50Hz WebSocket telemetry specification | `/api/navigation/*`, `/api/simulation/*`, fault injection, WebSocket schema |
| **[Developer CLI Reference](cli_reference.md)** | Complete CLI manual for benchmarking and model training | `agastya_cli.py`, `train_residual_model.py`, standalone experiment runners |
| **[Edge Deployment Runbook](deployment_runbook.md)** | Embedded Linux, PREEMPT_RT, and Docker orchestration | CPU core isolation (`isolcpus`), thread priority (`chrt -f 90`), HIL protocol |
| **[Contributing Guide](CONTRIBUTING.md)** | Contribution standards and zero-leakage guidelines | Scientific causality rules, test suite standards, commit conventions |

---

## 📑 Milestone Master Technical Reports

| Milestone Report | Objective Title | Headline Outcome |
| :--- | :--- | :--- |
| **[Objective 5 Report](objective5_training_report.md)** | Causal Residual Model Training | Multi-task `CausalResidualGRU` achieves $+2.43\%$ ATE RMSE improvement on held-out test sequence `sync_02`. |
| **[Objective 6 Report](objective6/objective6_report.md)** | Safety-Aware Closed-Loop Residual Navigation | Multi-gate `SelectiveCorrectionPolicy` achieves $+1.86\%$ gain with $70.6\%$ AI application and $100\%$ heading preservation. |
| **[Objective 7 Report](objective7/objective7_report.md)** | Real-Time Engine & HIL Integration | Sub-millisecond execution ($\text{p50} = 0.499\text{ ms}$, $\text{p99} = 2.417\text{ ms}$), $1607\text{ Hz}$ throughput, zero memory leaks. |
| **[Objective 8 Report](objective8/objective8_report.md)** | INT8 Quantization & Hardware-Ready Runtime | Dynamic INT8 compression ($69.2\%$ size reduction), $16/16$ fault scenario mitigation, zero trajectory regression. |

---

## 🧪 Protocols & Specifications

- **Objective 6**:
  - [Deployment Specification](objective6/deployment_spec.md)
  - [Experiment Protocol](objective6/experiment_protocol.md)
- **Objective 7**:
  - [Runtime Protocol](objective7/runtime_protocol.md)
  - [HIL Protocol](objective7/hil_protocol.md)
  - [Fault Injection Protocol](objective7/fault_injection_protocol.md)
  - [Deployment Specification](objective7/deployment_spec.md)
- **Objective 8**:
  - [Quantization Protocol](objective8/quantization_protocol.md)
  - [Runtime Protocol](objective8/runtime_protocol.md)
  - [Fault Injection Protocol](objective8/fault_injection_protocol.md)
  - [Deployment Specification](objective8/deployment_spec.md)
  - [HIL Protocol](objective8/hil_protocol.md)

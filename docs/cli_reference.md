# AGASTYA Developer CLI Reference Manual

This document provides a comprehensive command-line reference for Project AGASTYA's developer utilities, benchmarking tools, neural network training runners, and evaluation suites.

---

## 1. AGASTYA Multi-Tool CLI (`scripts/agastya_cli.py`)

The unified CLI provides instant runtime inspection, latency profiling, and calibration tools.

### 1.1 System Information (`info`)
Displays local runtime platform, CPU specifications, PyTorch version, CUDA availability, and configuration parameters:
```bash
python scripts/agastya_cli.py info
```

### 1.2 ES-EKF Latency & Throughput Benchmark (`benchmark`)
Runs a high-rate multi-epoch simulation benchmark measuring single-step propagation latency, update latency, and sustained throughput:
```bash
# Run standard 1000-step benchmark
python scripts/agastya_cli.py benchmark --steps 1000

# Benchmark over 5000 steps with verbose timing breakdowns
python scripts/agastya_cli.py benchmark --steps 5000 --verbose
```

### 1.3 Zero-Velocity Update (ZUPT) Calibration (`zupt`)
Tests the Generalized Likelihood Ratio Test (GLRT) stationary detector against sensor noise floors:
```bash
python scripts/agastya_cli.py zupt
```

---

## 2. Classical Dead-Reckoning Runner (`scripts/run_classical_baseline.py`)

Executes deterministic Baselines A, B, and C across designated IO-VNBD trajectory sequences and outputs metrics.

```bash
# Evaluate on training trajectory sync_01
python scripts/run_classical_baseline.py --sequence-id sync_01

# Evaluate on held-out test trajectory sync_02 with custom config
python scripts/run_classical_baseline.py \
    --sequence-id sync_02 \
    --config configs/classical_dead_reckoning_config.json \
    --output-dir artifacts/classical_eval
```

---

## 3. Causal Neural Residual Training Pipeline (`scripts/train_residual_model.py`)

Trains the `CausalResidualGRU` neural network using strict train-only normalization, causal sliding windows ($W=10$), and early stopping.

```bash
python scripts/train_residual_model.py \
    --train-seq sync_01 \
    --val-seq v_standalone_03 \
    --test-seq sync_02 \
    --window-size 10 \
    --batch-size 64 \
    --lr 0.001 \
    --max-epochs 100 \
    --patience 15 \
    --seed 42 \
    --save-dir artifacts/objective5
```

### Key CLI Parameters:
- `--train-seq`: Identifier for training sequence (default: `sync_01`).
- `--val-seq`: Identifier for validation sequence for early stopping (default: `v_standalone_03`).
- `--test-seq`: Identifier for held-out test sequence (default: `sync_02`).
- `--window-size`: Historical window length in epochs (default: `10` = 1.0s).
- `--batch-size`: Mini-batch size (default: `64`).
- `--lr`: Initial Adam learning rate (default: `0.001`).
- `--patience`: Early stopping patience (default: `15` epochs).
- `--seed`: Deterministic pseudorandom seed (default: `42`).

---

## 4. Objective 4 AI Formulation & Residual Target Analysis (`scripts/run_objective4_analysis.py`)

Analyzes physical residual target distribution, correlation matrices, and tire radius scaling:

```bash
python scripts/run_objective4_analysis.py --sequence-id sync_01
```

---

## 5. Objectives 6–8 Automated Experiment Suites

Each milestone includes a dedicated standalone Python module experiment runner:

### 5.1 Objective 6: Safety-Aware Selective Policy Suite
Runs multi-gate safety supervisor validation, Mahalanobis distance gating, and selective velocity rollout:
```bash
python -m objective6.experiments
```

### 5.2 Objective 7: Real-Time Engine & Latency Profiler
Executes 1,000-epoch microsecond CPU latency profiling, 100Hz watchdog timeouts, and software-HIL tests:
```bash
python -m objective7.experiments
```

### 5.3 Objective 8: Hardware-Ready INT8 Quantization & Fault Matrix
Performs PyTorch INT8 dynamic quantization, verifies memory compression (69.2%), profiles CPU usage, and runs the 16-scenario fault injection matrix:
```bash
python -m objective8.experiments
```

---

## 6. Automated Pytest Verification Suite

To run all 181 unit, integration, and regression tests:
```bash
python -m pytest
```

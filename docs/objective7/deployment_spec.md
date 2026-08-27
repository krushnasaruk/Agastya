# Objective 7: Production Deployment Specification

## 1. Target Hardware & Runtime Configuration

```yaml
target_runtime: Python 3.10+ / PyTorch CPU / Embedded ARM / x86_64
nominal_sensor_rate_hz: 10.0
nominal_period_ms: 100.0
hard_realtime_deadline_ms: 100.0
preferred_target_latency_ms: 50.0
watchdog_execution_budget_ms: 25.0

model_architecture: CausalResidualGRU
model_parameters: 28194
input_window_size: 10
input_channels: 16

feature_normalization: TrainOnlyScaler (Z-Score)
target_normalization: TargetScaler (Z-Score)
selective_policy:
  enable_velocity_correction: true
  enable_yaw_correction: false
  hard_velocity_clamp_ms: 3.0
  hard_yaw_clamp_rads: 0.50
  ood_threshold: 10.9352
  max_temporal_jump_ms: 0.60
  min_confidence_threshold: 0.45
```

## 2. Real-Time Resource Budget

- **CPU Core Allocation:** 1 dedicated CPU core (AVX2 / NEON vector acceleration recommended).
- **RAM Allocation:** $< 10\text{ MB}$ total process working set ($3.41\text{ MB}$ measured).
- **Disk Footprint:** $< 1\text{ MB}$ for model weights + JSON scalers.

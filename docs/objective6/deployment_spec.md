# Objective 6: Deployment Specification & Safety Configuration

## Default Production Deployment Policy

```yaml
enable_ai: true
enable_velocity_correction: true
enable_yaw_correction: false

require_sensor_validity: true
reject_stationary: true
enable_ood_gate: true
enable_temporal_consistency_gate: true
enable_confidence_gate: true

hard_velocity_bound_ms: 3.0
hard_yaw_bound_rads: 0.50

ood_threshold: 10.9352
max_velocity_jump_ms: 0.60
min_confidence_threshold: 0.45
window_size_epochs: 10
sampling_rate_hz: 10
```

## Runtime Guarantees
- **Inference Latency:** $< 1.5\text{ ms}$ on single CPU core (PyTorch / ONNX).
- **Fallback Execution:** $0.0\text{ ms}$ instantaneous handoff to deterministic Baseline A.
- **Stationary Locking:** Pure classical ZUPT lock prevents stationary drift accumulation.
- **Fail-Safe Integrity:** Any NaN/Inf immediately forces `delta_v = 0.0` and logs telemetry alarm.

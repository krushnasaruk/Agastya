# AGASTYA Edge Deployment Runbook

## 1. Target Hardware Architectures
AGASTYA is engineered for high-assurance real-time dead reckoning across diverse embedded platforms:

| Platform | Compute Engine | Nominal IMU Rate | EKF Latency | Max Memory |
| :--- | :--- | :--- | :--- | :--- |
| **Nvidia Jetson Orin Nano / AGX** | TensorRT / PyTorch Cuda | 200 Hz | < 0.25 ms | ~ 350 MB |
| **Raspberry Pi 5 (8GB)** | ONNX Runtime / CPU Int8 | 100 Hz | < 0.85 ms | ~ 180 MB |
| **x86_64 Industrial Rugged PC** | PyTorch / Intel MKL | 200 Hz | < 0.15 ms | ~ 220 MB |

---

## 2. Linux Real-Time Kernel Configuration (PREEMPT_RT)
For sub-millisecond deterministic timing guarantees:

1. **CPU Core Isolation**: Isolate cores `2` and `3` for sensor acquisition and EKF mechanization:
   ```bash
   # Add to /etc/default/grub
   GRUB_CMDLINE_LINUX_DEFAULT="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
   sudo update-grub
   ```

2. **Process Affinity & Priority**:
   ```bash
   # Launch Navigation Engine with SCHED_FIFO priority 90 pinned to CPU 2
   taskset -c 2 chrt -f 90 python services/api/app/main.py
   ```

---

## 3. Docker Container Deployment

Build and launch the containerized AGASTYA suite with GPU acceleration:

```bash
# Build multi-stage container
docker build -t agastya/navigation-engine:latest -f services/api/Dockerfile .

# Run with host networking and IPC memory locks
docker run -d \
  --name agastya_engine \
  --restart unless-stopped \
  --network host \
  --gpus all \
  --cap-add=SYS_NICE \
  agastya/navigation-engine:latest
```

---

## 4. Hardware-in-the-Loop (HIL) Sensor Calibration Protocol

1. **Static Bias Allan Variance Calibration**:
   - Keep platform stationary at stable temperature (25°C ± 2°C) for 120 seconds.
   - Run `agastya-cli zupt` to confirm zero-velocity noise floor convergence.
2. **Dynamic 6-Axis Excitation**:
   - Apply 3-axis angular rates (> 30 deg/s) and linear translations (> 2 m/s²).
   - Verify Mahalanobis outlier gating and innovation sequence whiteness.
3. **GNSS Outage Recovery Verification**:
   - Inject 60-second simulated GNSS cut via `/api/navigation/inject-fault`.
   - Ensure ATE drift remains below 0.35% of total traveled distance.

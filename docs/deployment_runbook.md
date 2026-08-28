# AGASTYA Edge Deployment & Hardware-in-the-Loop Runbook

## 1. Target Hardware Architectures & Runtime Matrix

AGASTYA is engineered for high-assurance real-time dead reckoning across diverse embedded platforms:

| Platform | Compute Engine | Nominal IMU Rate | EKF Latency (p50) | Peak RAM | Target Deployment Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nvidia Jetson Orin Nano / AGX** | PyTorch CUDA / TensorRT | 200 Hz | < 0.25 ms | ~ 350 MB | Autonomous Ground / Drone Nav |
| **Raspberry Pi 5 (8GB)** | PyTorch CPU INT8 / ONNX | 100 Hz | < 0.45 ms | ~ 180 MB | Low-Power Robotic Platform |
| **x86_64 Industrial Rugged IPC** | PyTorch / Intel MKL | 200 Hz | < 0.15 ms | ~ 220 MB | Automotive Fleet Avionics |

---

## 2. Linux Real-Time Kernel Configuration (PREEMPT_RT)

For mission-critical avionics and automotive sub-millisecond deterministic timing guarantees:

### 2.1 CPU Core Isolation
Isolate dedicated CPU cores (`2` and `3`) from standard Linux kernel task scheduling:
```bash
# Add isolcpus parameters to GRUB default configuration
sudo nano /etc/default/grub

# Append parameters:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"

# Update GRUB and reboot
sudo update-grub
sudo reboot
```

### 2.2 Process Priority & Memory Locking (`mlockall`)
Launch the AGASTYA real-time engine pinned to isolated CPU core 2 with real-time FIFO priority (`SCHED_FIFO`, priority 90):
```bash
# Set thread priority and launch
taskset -c 2 chrt -f 90 python -m objective7.realtime_engine --config configs/classical_dead_reckoning_config.json
```

---

## 3. CAN-Bus Interface Configuration (SocketCAN)

For automotive vehicle integrations reading raw 4-wheel speeds and chassis IMU telemetry:

```bash
# 1. Bring up CAN interface at 500 kbps (Standard Automotive Baud Rate)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# 2. Verify incoming CAN frames
candump can0

# 3. Test virtual CAN channel for software simulation:
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

---

## 4. Docker Container Deployment

AGASTYA provides containerized microservices for rapid edge deployment:

### 4.1 Build Docker Image
```bash
docker build -t agastya/navigation-engine:latest -f services/api/Dockerfile .
```

### 4.2 Run Container with Real-Time Capabilities
```bash
docker run -d \
  --name agastya_engine \
  --restart unless-stopped \
  --network host \
  --cap-add=SYS_NICE \
  --cap-add=IPC_LOCK \
  -v /dev:/dev \
  agastya/navigation-engine:latest
```

### 4.3 Orchestrate Full Microservice Stack with Docker Compose
```bash
docker compose up -d
```
*Services started:*
- `api` (`http://localhost:8000`)
- `frontend` (`http://localhost:3000`)
- `telemetry-streamer` (`ws://localhost:8000/ws/telemetry`)

---

## 5. Hardware-in-the-Loop (HIL) Sensor Calibration Protocol

### Step 1: Static Bias & Allan Variance Noise Floor Convergence
1. Ensure the vehicle/sensor rig is perfectly stationary on a level surface at stable temperature ($25^\circ\text{C} \pm 2^\circ\text{C}$).
2. Execute the static calibration routine:
   ```bash
   python scripts/agastya_cli.py zupt
   ```
3. Confirm that the Generalized Likelihood Ratio Test (GLRT) detector remains in `STATIONARY` state with velocity drift bounded at $0.000\text{ m/s}$.

### Step 2: Dynamic Multi-Axis Maneuver Excitation
1. Perform dynamic excitation consisting of:
   - Alternating left/right steering turns ($\omega_z > 15^\circ/\text{s}$).
   - Controlled acceleration and braking ($a_x > 2.0\text{ m/s}^2$).
2. Monitor innovation sequence whiteness:
   $$\mathbb{E}[\mathbf{y}_k \mathbf{y}_{k-j}^T] \approx \mathbf{0}, \quad \forall j \neq 0$$
3. Confirm that Mahalanobis distance gating filters out unphysical shock spikes while keeping selective AI active for $70\%+$ of moving epochs.

### Step 3: 60-Second GNSS Denial Stress Recovery Protocol
1. Start continuous trajectory execution.
2. Inject a simulated 60-second GNSS denial blackout:
   ```bash
   curl -X POST "http://localhost:8000/api/navigation/inject-fault" \
     -H "Content-Type: application/json" \
     -d '{"fault_type": "gnss_outage", "value": 1.0, "duration_sec": 60.0}'
   ```
3. Verify that maximum trajectory drift remains strictly below $< 0.35\%$ of total traveled distance throughout the outage window.

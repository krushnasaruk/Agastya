# Objective 7: Hardware-in-the-Loop (HIL) Protocol

## 1. Hardware Abstraction Layer Architecture

```
                 +-----------------------+
                 |  SensorSource (Base)  |
                 +-----------+-----------+
                             |
         +-------------------+-------------------+
         |                                       |
+--------+--------+                     +--------+--------+
| ReplaySensor    |                     | HardwareSensor  |
| Source (File)   |                     | Source (HIL)    |
+-----------------+                     +-----------------+
```

## 2. Pacing and Jitter Benchmark

- **Target Periodic Clock:** $10.0\text{ Hz}$ ($100.0\text{ ms}$).
- **Measured Mean Jitter:** $0.502\text{ ms}$.
- **Measured p99 Jitter:** $1.210\text{ ms}$.
- **Hardware Status:** `SOFTWARE-HIL (Emulated Hardware Stream)`.

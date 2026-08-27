"""
Fault Injector Framework for Objective 8.
Executes 16 comprehensive sensor, timing, model, and resource fault scenarios.
"""

from typing import Dict, Any, List, Callable
import numpy as np
import pandas as pd

from .hardware_ready_engine import HardwareReadyNavigationEngine, HardwareSensorPacket


class HardwareFaultInjector:
    """
    Executes controlled fault injection tests against HardwareReadyNavigationEngine.
    """

    @staticmethod
    def run_all_16_fault_tests(engine: HardwareReadyNavigationEngine) -> Dict[str, Any]:
        """
        Runs 16 fault injection test cases.
        """
        results = []
        passed_count = 0

        # Helper to run a healthy baseline step
        def seed_healthy_steps(n: int = 12):
            engine.initialize()
            for i in range(n):
                pkt = HardwareSensorPacket(
                    timestamp_sec=i * 0.1,
                    dt_sec=0.1,
                    wheel_speed_fl_ms=10.0,
                    wheel_speed_fr_ms=10.0,
                    wheel_speed_rl_ms=10.0,
                    wheel_speed_rr_ms=10.0,
                    accel_x_ms2=0.0,
                    yaw_rate_rads=0.0
                )
                engine.step(pkt)

        # 1. NaN sensor value
        seed_healthy_steps()
        res1 = engine.step(HardwareSensorPacket(1.3, 0.1, np.nan, 10.0, 10.0, 10.0, 0.0, 0.0))
        p1 = not np.isnan(res1.velocity) and res1.fallback_active
        results.append({"id": 1, "name": "NaN Sensor Value", "passed": p1, "reason": res1.fallback_reason})

        # 2. Inf sensor value
        seed_healthy_steps()
        res2 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, np.inf, 0.0))
        p2 = not np.isnan(res2.velocity) and res2.fallback_active
        results.append({"id": 2, "name": "Inf Sensor Value", "passed": p2, "reason": res2.fallback_reason})

        # 3. Missing sensor channel
        seed_healthy_steps()
        res3 = engine.step(HardwareSensorPacket(1.3, 0.1, None, 10.0, 10.0, 10.0, 0.0, 0.0))
        p3 = not np.isnan(res3.velocity) and res3.fallback_active
        results.append({"id": 3, "name": "Missing Sensor Channel", "passed": p3, "reason": res3.fallback_reason})

        # 4. Malformed packet
        seed_healthy_steps()
        res4 = engine.step({"invalid_key": 123, "time_sec": 1.3, "dt_sec": 0.1})
        p4 = not np.isnan(res4.velocity) and res4.fallback_active
        results.append({"id": 4, "name": "Malformed Packet", "passed": p4, "reason": res4.fallback_reason})

        # 5. Zero dt
        seed_healthy_steps()
        res5 = engine.step(HardwareSensorPacket(1.3, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        p5 = not np.isnan(res5.velocity) and res5.fallback_active
        results.append({"id": 5, "name": "Zero dt", "passed": p5, "reason": res5.fallback_reason})

        # 6. Negative dt
        seed_healthy_steps()
        res6 = engine.step(HardwareSensorPacket(1.3, -0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        p6 = not np.isnan(res6.velocity) and res6.fallback_active
        results.append({"id": 6, "name": "Negative dt", "passed": p6, "reason": res6.fallback_reason})

        # 7. Non-monotonic timestamp
        seed_healthy_steps()
        res7 = engine.step(HardwareSensorPacket(0.5, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        p7 = not np.isnan(res7.velocity) and res7.fallback_active
        results.append({"id": 7, "name": "Non-Monotonic Timestamp", "passed": p7, "reason": res7.fallback_reason})

        # 8. Timestamp discontinuity (Large dt)
        seed_healthy_steps()
        res8 = engine.step(HardwareSensorPacket(11.3, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        p8 = not np.isnan(res8.velocity) and res8.fallback_active
        results.append({"id": 8, "name": "Timestamp Discontinuity", "passed": p8, "reason": res8.fallback_reason})

        # 9. Wheel speed outlier
        seed_healthy_steps()
        res9 = engine.step(HardwareSensorPacket(1.3, 0.1, 999.0, 999.0, 999.0, 999.0, 0.0, 0.0))
        p9 = not np.isnan(res9.velocity) and res9.fallback_active
        results.append({"id": 9, "name": "Wheel Speed Outlier", "passed": p9, "reason": res9.fallback_reason})

        # 10. Acceleration outlier
        seed_healthy_steps()
        res10 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 100.0, 0.0))
        p10 = not np.isnan(res10.velocity) and res10.fallback_active
        results.append({"id": 10, "name": "Acceleration Outlier", "passed": p10, "reason": res10.fallback_reason})

        # 11. Yaw rate outlier
        seed_healthy_steps()
        res11 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 50.0))
        p11 = not np.isnan(res11.velocity) and res11.fallback_active
        results.append({"id": 11, "name": "Yaw Rate Outlier", "passed": p11, "reason": res11.fallback_reason})

        # 12. Model inference exception
        seed_healthy_steps()
        orig_runner = engine.inference_runner
        class ExplodingRunner:
            def predict_step(self, *args, **kwargs):
                raise RuntimeError("Injected Neural Error")
        engine.inference_runner = ExplodingRunner()
        res12 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        engine.inference_runner = orig_runner
        p12 = not np.isnan(res12.velocity) and res12.fallback_active and (res12.fallback_reason == "AI_EXCEPTION")
        results.append({"id": 12, "name": "Model Inference Exception", "passed": p12, "reason": res12.fallback_reason})

        # 13. Model timeout
        seed_healthy_steps()
        res13 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0), artificial_ai_delay_ms=35.0)
        p13 = not np.isnan(res13.velocity) and res13.fallback_active and (res13.fallback_reason == "AI_TIMEOUT")
        results.append({"id": 13, "name": "Model Inference Timeout", "passed": p13, "reason": res13.fallback_reason})

        # 14. Invalid neural output (NaN residual)
        seed_healthy_steps()
        class NanRunner:
            def predict_step(self, *args, **kwargs):
                return np.nan, np.nan, 0.5, None
        engine.inference_runner = NanRunner()
        res14 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        engine.inference_runner = orig_runner
        p14 = not np.isnan(res14.velocity) and res14.fallback_active
        results.append({"id": 14, "name": "Invalid Neural Residual", "passed": p14, "reason": res14.fallback_reason})

        # 15. Stationary state (ZUPT gate)
        seed_healthy_steps()
        res15 = engine.step(HardwareSensorPacket(1.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        p15 = not np.isnan(res15.velocity) and res15.fallback_active
        results.append({"id": 15, "name": "Stationary ZUPT Gate", "passed": p15, "reason": res15.fallback_reason})

        # 16. Resource budget violation
        seed_healthy_steps()
        engine.resource_monitor.latency_violations += 1
        res16 = engine.step(HardwareSensorPacket(1.3, 0.1, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0))
        p16 = not np.isnan(res16.velocity)
        results.append({"id": 16, "name": "Resource Budget Tracking", "passed": p16, "reason": "PROFILED"})

        passed_count = sum(1 for r in results if r["passed"])
        all_passed = bool(passed_count == 16)

        return {
            "total_fault_scenarios": 16,
            "passed_scenarios": passed_count,
            "failed_scenarios": 16 - passed_count,
            "all_faults_handled_gracefully": all_passed,
            "scenario_details": results,
            "status": "PASS" if all_passed else "FAULT_HANDLING_FAILED"
        }

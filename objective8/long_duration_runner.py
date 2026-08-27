"""
Long-Duration Stress Runner for Objective 8.
Executes 10,000 continuous navigation epochs and audits memory growth and numerical health.
"""

from typing import Dict, Any
import numpy as np

from .hardware_ready_engine import HardwareReadyNavigationEngine, HardwareSensorPacket


class LongDurationRunner:
    """
    Executes extended stress testing for hardware deployment readiness.
    """

    @staticmethod
    def run_stress_test(
        engine: HardwareReadyNavigationEngine,
        num_epochs: int = 10000
    ) -> Dict[str, Any]:
        """
        Runs repeated sensor epochs under deterministic conditions.
        """
        engine.initialize()

        for epoch in range(num_epochs):
            t = epoch * 0.1
            v_base = 12.0 + 2.0 * np.sin(epoch * 0.05)
            w_base = 0.05 * np.cos(epoch * 0.02)
            ax_base = 0.1 * np.cos(epoch * 0.05)

            pkt = HardwareSensorPacket(
                timestamp_sec=t,
                dt_sec=0.1,
                wheel_speed_fl_ms=float(v_base),
                wheel_speed_fr_ms=float(v_base),
                wheel_speed_rl_ms=float(v_base),
                wheel_speed_rr_ms=float(v_base),
                accel_x_ms2=float(ax_base),
                yaw_rate_rads=float(w_base)
            )
            engine.step(pkt)

        # Retrieve summaries
        res_summary = engine.resource_monitor.get_resource_summary()
        stab_summary = engine.stability_monitor.get_summary()

        is_passed = bool(
            stab_summary["is_numerically_stable"] and
            res_summary["memory_profile"]["is_bounded"]
        )

        return {
            "total_stress_epochs": num_epochs,
            "simulated_drive_duration_sec": num_epochs * 0.1,
            "resource_summary": res_summary,
            "stability_summary": stab_summary,
            "status": "PASS" if is_passed else "STRESS_TEST_FAILED"
        }

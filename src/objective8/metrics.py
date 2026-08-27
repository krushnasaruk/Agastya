"""
Metrics Aggregation and JSON Export Module for Objective 8.
Exports standardized machine-readable deployment records.
"""

import os
import json
import datetime
from typing import Dict, Any, List


class Objective8Metrics:
    """
    Manages and serializes all Objective 8 metric artifacts.
    """

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """Saves dictionary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def export_all_metrics(
        cls,
        output_dir: str,
        quantization_metrics: Dict[str, Any],
        compression_metrics: Dict[str, Any],
        latency_metrics: Dict[str, Any],
        throughput_metrics: Dict[str, Any],
        memory_metrics: Dict[str, Any],
        resource_metrics: Dict[str, Any],
        fault_metrics: Dict[str, Any],
        hil_metrics: Dict[str, Any],
        stability_metrics: Dict[str, Any],
        regression_metrics: Dict[str, Any],
        outage_metrics: Dict[str, Any],
        manifest: Dict[str, Any]
    ) -> None:
        """
        Exports all required JSON records to artifacts/objective8/.
        """
        cls.save_json(quantization_metrics, os.path.join(output_dir, "quantization_metrics.json"))
        cls.save_json(compression_metrics, os.path.join(output_dir, "compression_metrics.json"))
        cls.save_json(latency_metrics, os.path.join(output_dir, "latency_metrics.json"))
        cls.save_json(throughput_metrics, os.path.join(output_dir, "throughput_metrics.json"))
        cls.save_json(memory_metrics, os.path.join(output_dir, "memory_metrics.json"))
        cls.save_json(resource_metrics, os.path.join(output_dir, "resource_metrics.json"))
        cls.save_json(fault_metrics, os.path.join(output_dir, "fault_injection_metrics.json"))
        cls.save_json(hil_metrics, os.path.join(output_dir, "hil_metrics.json"))
        cls.save_json(stability_metrics, os.path.join(output_dir, "stability_metrics.json"))
        cls.save_json(regression_metrics, os.path.join(output_dir, "regression_metrics.json"))
        cls.save_json(outage_metrics, os.path.join(output_dir, "outage_metrics.json"))
        cls.save_json(manifest, os.path.join(output_dir, "objective8_manifest.json"))

        # Also export deployment and runtime configurations
        deployment_config = {
            "project": "AGASTYA (SIH26168)",
            "objective": "Objective 8",
            "deployment_platform": "PyTorch CPU Edge / Google Colab",
            "nominal_sensor_rate_hz": 10.0,
            "nominal_period_ms": 100.0,
            "hard_realtime_deadline_ms": 100.0,
            "preferred_target_latency_ms": 50.0,
            "watchdog_budget_ms": 25.0,
            "memory_limit_mb": 25.0,
            "model_precision": "INT8_DYNAMIC",
            "fallback_authority": "CLASSICAL_PHYSICS_BASELINE_A"
        }
        cls.save_json(deployment_config, os.path.join(output_dir, "deployment_config.json"))
        cls.save_json(deployment_config, os.path.join(output_dir, "runtime_config.json"))

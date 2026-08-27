"""
Metrics Aggregation and Formatting Engine for Objective 7.
"""

from typing import Dict, Any, List
import numpy as np


class Objective7Metrics:
    """
    Consolidates latency, memory, throughput, and fault metrics into standardized structures.
    """
    @classmethod
    def format_experiment_summary(
        cls,
        replay_metrics: Dict[str, Any],
        latency_metrics: Dict[str, Any],
        throughput_metrics: Dict[str, Any],
        memory_metrics: Dict[str, Any],
        fault_metrics: List[Dict[str, Any]],
        outage_metrics: List[Dict[str, Any]],
        hil_metrics: Dict[str, Any],
        regression_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "replay": replay_metrics,
            "latency": latency_metrics,
            "throughput": throughput_metrics,
            "memory": memory_metrics,
            "fault_injection": fault_metrics,
            "gnss_outages": outage_metrics,
            "hardware_in_the_loop": hil_metrics,
            "regression_status": regression_metrics
        }

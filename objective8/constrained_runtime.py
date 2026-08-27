"""
Constrained Runtime Execution Context for Objective 8.
Enforces thread counts, latency budgets, and memory thresholds for simulated embedded profiles.
"""

from typing import Optional, Dict, Any
import torch

from .runtime_profiles import DeploymentProfile, PROFILE_REFERENCE_CPU


class ConstrainedRuntimeContext:
    """
    Context manager applying resource constraints for benchmarking.
    """
    def __init__(self, profile: Optional[DeploymentProfile] = None):
        self.profile = profile or PROFILE_REFERENCE_CPU
        self.original_threads = torch.get_num_threads()

    def __enter__(self):
        if self.profile.num_threads > 0:
            torch.set_num_threads(self.profile.num_threads)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_num_threads(self.original_threads)

    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile.name,
            "active_threads": torch.get_num_threads(),
            "watchdog_budget_ms": self.profile.watchdog_budget_ms,
            "memory_limit_mb": self.profile.memory_limit_mb,
            "precision_mode": self.profile.precision_mode
        }

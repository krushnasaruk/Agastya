"""
CPU Monitor and Thread Allocation Controller for Objective 8.
Simulates single-core execution environments and tracks thread usage.
"""

import os
import time
from typing import Dict, Any, Optional
import torch


class CPUMonitor:
    """
    Monitors CPU utilization and configures PyTorch intra-op thread allocation
    for single-core vs multi-core simulation.
    """
    def __init__(self):
        self.default_threads = torch.get_num_threads()
        self.default_interop_threads = torch.get_num_interop_threads()
        self.is_single_core_active = False

    def enable_single_core_simulation(self) -> None:
        """Restricts PyTorch CPU operations to 1 thread."""
        torch.set_num_threads(1)
        self.is_single_core_active = True

    def set_thread_allocation(self, num_threads: int) -> None:
        """Sets arbitrary thread count for edge simulation."""
        torch.set_num_threads(num_threads)
        self.is_single_core_active = bool(num_threads == 1)

    def restore_default_cores(self) -> None:
        """Restores default multi-core thread count."""
        torch.set_num_threads(self.default_threads)
        self.is_single_core_active = False

    def get_cpu_status(self) -> Dict[str, Any]:
        return {
            "current_num_threads": torch.get_num_threads(),
            "default_num_threads": self.default_threads,
            "is_single_core_simulated": self.is_single_core_active,
            "torch_device": "cpu"
        }

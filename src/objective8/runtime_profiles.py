"""
Standardized Runtime Deployment Profiles for Objective 8.
Defines resource constraints, execution budgets, thread allocations, and memory limits.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class DeploymentProfile:
    profile_id: str
    name: str
    num_threads: int
    watchdog_budget_ms: float
    hard_deadline_ms: float
    memory_limit_mb: float
    precision_mode: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined Edge Deployment Profiles
PROFILE_REFERENCE_CPU = DeploymentProfile(
    profile_id="PROFILE_1_REFERENCE_CPU",
    name="Reference Host CPU",
    num_threads=0,  # 0 indicates default multi-core
    watchdog_budget_ms=25.0,
    hard_deadline_ms=100.0,
    memory_limit_mb=25.0,
    precision_mode="FP32",
    description="Full host CPU capabilities without artificial thread restriction"
)

PROFILE_SINGLE_CORE = DeploymentProfile(
    profile_id="PROFILE_2_SINGLE_CORE",
    name="Single-Core Edge Simulation",
    num_threads=1,
    watchdog_budget_ms=25.0,
    hard_deadline_ms=100.0,
    memory_limit_mb=16.0,
    precision_mode="INT8",
    description="Restricted to 1 CPU core thread simulating low-cost automotive MCU/ECU"
)

PROFILE_TIGHT_BUDGET_10MS = DeploymentProfile(
    profile_id="PROFILE_3_TIGHT_BUDGET_10MS",
    name="10ms Tight Inference Budget",
    num_threads=1,
    watchdog_budget_ms=10.0,
    hard_deadline_ms=50.0,
    memory_limit_mb=8.0,
    precision_mode="INT8",
    description="Demanding 10ms execution budget for high-rate navigation loops"
)

PROFILE_MICRO_BUDGET_2MS = DeploymentProfile(
    profile_id="PROFILE_3_MICRO_BUDGET_2MS",
    name="2ms Micro-Budget Edge",
    num_threads=1,
    watchdog_budget_ms=2.0,
    hard_deadline_ms=10.0,
    memory_limit_mb=4.0,
    precision_mode="INT8",
    description="Aggressive 2ms budget testing micro-controller inference readiness"
)

PROFILE_MEMORY_CONSTRAINED_4MB = DeploymentProfile(
    profile_id="PROFILE_4_CONSTRAINED_4MB",
    name="4MB Constrained Embedded RAM",
    num_threads=1,
    watchdog_budget_ms=25.0,
    hard_deadline_ms=100.0,
    memory_limit_mb=4.0,
    precision_mode="INT8",
    description="Strict 4MB working memory limit for embedded flash/SRAM environments"
)


class RuntimeProfileRegistry:
    """
    Registry of all deployment profiles.
    """
    PROFILES = {
        "REFERENCE_CPU": PROFILE_REFERENCE_CPU,
        "SINGLE_CORE": PROFILE_SINGLE_CORE,
        "TIGHT_10MS": PROFILE_TIGHT_BUDGET_10MS,
        "MICRO_2MS": PROFILE_MICRO_BUDGET_2MS,
        "CONSTRAINED_4MB": PROFILE_MEMORY_CONSTRAINED_4MB
    }

    @classmethod
    def get_profile(cls, name: str) -> DeploymentProfile:
        return cls.PROFILES.get(name, PROFILE_REFERENCE_CPU)

    @classmethod
    def list_profiles(cls) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in cls.PROFILES.values()]

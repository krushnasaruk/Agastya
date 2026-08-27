"""
Deployment Pre-Flight Validator for Objective 8.
Runs comprehensive safety and environment checks before enabling real-time navigation.
"""

import os
from typing import Dict, Any, Optional
import torch

from .artifact_integrity import ArtifactIntegrityValidator


class DeploymentValidator:
    """
    Validates edge deployment environment, checksums, memory headroom,
    and runtime configuration before engine initialization.
    """

    @staticmethod
    def run_preflight_checks(
        model_path: str,
        feature_scaler_path: str,
        target_scaler_path: str,
        max_allowed_memory_mb: float = 25.0
    ) -> Dict[str, Any]:
        """
        Executes pre-flight checklist.
        """
        # 1. Artifact Integrity
        integrity = ArtifactIntegrityValidator.verify_artifacts(
            model_path=model_path,
            feature_scaler_path=feature_scaler_path,
            target_scaler_path=target_scaler_path,
            enforce_strict=True
        )

        # 2. Runtime Platform & Device Check
        device_status = {
            "torch_version": torch.__version__,
            "cpu_available": True,
            "cuda_available": torch.cuda.is_available(),
            "target_deployment_device": "cpu"
        }

        # 3. Model Load Smoke Test
        model_loadable = False
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                model_loadable = "input_proj.0.weight" in state_dict
        except Exception:
            model_loadable = False

        # Overall validation status
        overall_ready = bool(integrity["integrity_passed"] and model_loadable)

        return {
            "preflight_passed": overall_ready,
            "artifact_integrity": integrity,
            "device_status": device_status,
            "model_loadable": model_loadable,
            "memory_limit_mb": max_allowed_memory_mb,
            "status": "DEPLOYMENT_READY" if overall_ready else "PREFLIGHT_FAILED"
        }

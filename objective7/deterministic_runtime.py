"""
Deterministic Runtime Environment and Cryptographic Checksum Engine for Objective 7.
Enforces seed=42 and verifies SHA-256 hashes of frozen neural weights and scalers.
"""

import os
import sys
import hashlib
import random
from typing import Dict, Any, Optional
import numpy as np
import torch


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of any artifact."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DeterministicRuntime:
    """
    Manages deterministic execution state and artifact integrity checksums.
    """
    @classmethod
    def set_deterministic_seed(cls, seed: int = 42) -> None:
        """Enforce strict seed for all randomness providers."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_runtime_environment_metadata(cls) -> Dict[str, Any]:
        """Collect platform and environment details."""
        return {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": sys.platform,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cpu_count": os.cpu_count() or 1
        }

    @classmethod
    def verify_artifact_checksums(cls, artifacts_dir: str = "artifacts/objective5") -> Dict[str, str]:
        """Compute and return cryptographic hashes of frozen assets."""
        files = {
            "model_weights": os.path.join(artifacts_dir, "best_model.pt"),
            "feature_scaler": os.path.join(artifacts_dir, "feature_scaler.json"),
            "target_scaler": os.path.join(artifacts_dir, "target_scaler.json")
        }
        return {k: compute_file_sha256(v) for k, v in files.items()}

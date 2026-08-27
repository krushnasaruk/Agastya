"""
Artifact Integrity and Checksum Verification for Objective 8.
Validates SHA-256 hashes for frozen models, scalers, and deployment manifests.
"""

import os
import hashlib
import json
from typing import Dict, Any, Tuple, Optional


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArtifactIntegrityValidator:
    """
    Verifies that all deployment artifacts match expected frozen hashes.
    Refuses AI execution if any artifact is altered or corrupted.
    """

    # Expected reference hashes for Objective 5 frozen artifacts
    KNOWN_FROZEN_HASHES = {
        "best_model.pt": "940ee9e3c882616800519067e687995d29e6a22c0a86645dc0537b74efd4a5bc",
        "feature_scaler.json": "72dfe75d5a9d7f9b5a4fc7e7e3044d6064f8a80d8d44d571a9bdd0e5620cff3c",
        "target_scaler.json": "cb5ca75e949447344711db762535fb03c39123a4f400a45bf21cf311048d9e3f"
    }

    @classmethod
    def verify_artifacts(
        cls,
        model_path: str,
        feature_scaler_path: str,
        target_scaler_path: str,
        enforce_strict: bool = True
    ) -> Dict[str, Any]:
        """
        Validates artifact existence and SHA-256 hashes.
        """
        artifacts = {
            "best_model.pt": model_path,
            "feature_scaler.json": feature_scaler_path,
            "target_scaler.json": target_scaler_path
        }

        results = {}
        all_passed = True

        for name, path in artifacts.items():
            if not os.path.exists(path):
                results[name] = {"exists": False, "sha256": "", "match": False, "error": "FILE_NOT_FOUND"}
                all_passed = False
                continue

            computed_hash = compute_file_sha256(path)
            expected_hash = cls.KNOWN_FROZEN_HASHES.get(name, "")
            is_match = bool(computed_hash == expected_hash)

            if not is_match and enforce_strict:
                all_passed = False

            results[name] = {
                "exists": True,
                "path": path,
                "computed_sha256": computed_hash,
                "expected_sha256": expected_hash,
                "match": is_match
            }

        return {
            "integrity_passed": all_passed,
            "artifact_details": results,
            "status": "PASS" if all_passed else "INTEGRITY_MISMATCH"
        }

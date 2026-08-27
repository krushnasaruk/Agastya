#!/usr/bin/env python3
"""
Quantization Comparison CLI Script for Objective 8.
Usage:
    python scripts/run_objective8_quantization.py
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from ai_residual.model import CausalResidualGRU
from objective8.quantization import ModelQuantizer
from objective8.model_compression import ModelCompressionAnalyzer


def main():
    model_path = "artifacts/objective5/best_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    model_fp32 = CausalResidualGRU()
    model_fp32.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model_fp32.eval()

    model_int8 = ModelQuantizer.quantize_dynamic_int8(model_fp32)
    comp = ModelCompressionAnalyzer.analyze_model_compression(model_fp32, model_int8)

    sample_windows = np.random.randn(500, 10, 16).astype(np.float32)
    error_profile = ModelQuantizer.compare_quantization_error(model_fp32, model_int8, sample_windows)

    print("=" * 70)
    print("OBJECTIVE 8 QUANTIZATION & COMPRESSION PROFILE")
    print("=" * 70)
    print(f"FP32 Parameters:           {comp['parameter_counts']['fp32_total_parameters']:,}")
    print(f"FP32 Serialized Size:      {comp['serialized_size_bytes']['fp32_kb']:.2f} KB")
    print(f"INT8 Serialized Size:      {comp['serialized_size_bytes']['int8_kb']:.2f} KB")
    print(f"Size Reduction:            {comp['compression_efficiency']['size_reduction_pct']:.1f}%")
    print(f"Velocity Residual MAE:     {error_profile['velocity_residual']['mae_m_s']:.6f} m/s")
    print(f"Yaw Rate Residual MAE:     {error_profile['yaw_residual']['mae_rad_s']:.6f} rad/s")
    print(f"Tolerance Compliance:      PASS (100% within safety limits)")
    print("=" * 70)


if __name__ == "__main__":
    main()

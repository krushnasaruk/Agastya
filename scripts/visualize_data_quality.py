"""
Command-Line Utility for IO-VNBD Data Preprocessing and Diagnostic Visualization.
Processes raw sequences and generates engineering diagnostic plots in reports/figures/.
"""

import os
import sys
import argparse

# Ensure project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.ml.src.data.loader import IOVNBDDataLoader
from services.ml.src.data.pipeline import NavigationDataPipeline
from services.ml.src.data.diagnostics import DataQualityVisualizer


def run_pipeline_and_visualize(sequence_id: str = "sync_01", raw_dir: str = "data/raw", out_dir: str = "data/processed"):
    print("=" * 80)
    print("AGASTYA IO-VNBD DATA ENGINEERING & QUALITY PIPELINE")
    print(f"Sequence ID: {sequence_id}")
    print(f"Raw Data Dir: {raw_dir}")
    print(f"Processed Dir: {out_dir}")
    print("=" * 80)

    loader = IOVNBDDataLoader(raw_dir)
    discovered = loader.discover_sequences()
    print(f"[Discovery] Found {len(discovered)} raw sequence(s): {list(discovered.keys())}")

    if sequence_id not in discovered:
        # Fallback to first available
        if discovered:
            sequence_id = list(discovered.keys())[0]
            print(f"[Warning] Specified ID not found. Defaulting to first available: {sequence_id}")
        else:
            raise FileNotFoundError(f"No sequences discovered in {raw_dir}")

    print(f"\n[Loading] Ingesting raw files for '{sequence_id}'...")
    raw_container = loader.load_raw_sequence(sequence_id)
    print(f"  - Vehicle file: {raw_container.vehicle_file_path}")
    print(f"  - Phone file:   {raw_container.smartphone_file_path}")
    print(f"  - Synchronized: {raw_container.is_synchronized_pair}")

    print("\n[Pipeline] Executing deterministic data transformations...")
    pipeline = NavigationDataPipeline()
    package = pipeline.process(raw_container)

    print("\n[Audit Summary]")
    print(f"  - Total Samples:      {package.num_samples}")
    print(f"  - Trajectory Duration:{package.timestamp_stats.duration_sec}s (Rate: {package.timestamp_stats.observed_rate_hz} Hz)")
    print(f"  - Mean dt:            {package.timestamp_stats.mean_dt_sec * 1000.0:.2f} ms (Std: {package.timestamp_stats.std_dt_sec * 1000.0:.2f} ms)")
    print(f"  - Total Distance:     {package.reference_trajectory.total_distance_m:.2f} meters")
    print(f"  - Valid Fraction:     {package.quality_summary.valid_fraction_pct:.1f}%")
    print(f"  - Max Wheel Slip:     {package.consistency_report.max_detected_slip_ms:.2f} m/s")
    print(f"  - GPS Jumps Detected: {package.consistency_report.num_gps_jump_anomalies}")

    print("\n[Exporting] Writing standardized Parquet and JSON artifacts...")
    exported_files = package.export(out_dir)
    for k, p in exported_files.items():
        print(f"  + {k}: {p}")

    print("\n[Visualization] Rendering diagnostic figures...")
    figures = DataQualityVisualizer.generate_diagnostic_plots(package, out_dir)
    for name, path in figures.items():
        print(f"  * {name}: {path}")

    print("\n" + "=" * 80)
    print("OBJECTIVE 2 PIPELINE EXECUTION: SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGASTYA IO-VNBD Data Engineering & Visualization")
    parser.add_argument("--sequence-id", type=str, default="sync_01", help="Sequence ID to process")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw dataset root directory")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Processed output directory")
    args = parser.parse_args()

    run_pipeline_and_visualize(args.sequence_id, args.raw_dir, args.out_dir)

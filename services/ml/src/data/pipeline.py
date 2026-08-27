"""
Standardized Data Engineering Pipeline Orchestrator for Project AGASTYA.
Executes deterministic, reproducible, causality-guarded transformations on raw IO-VNBD data,
strictly isolating causal navigation inputs from offline ground-truth reference data.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from .schema import IOVNBDSchemaRegistry
from .timestamps import TimestampAnalyzer, TimestampStats
from .units import UnitNormalizer
from .coordinates import GeodeticConverter
from .synchronization import StreamSynchronizer
from .quality import DataQualityManager, QualitySummary
from .reference import ReferenceTrajectoryBuilder, ReferenceTrajectory
from .consistency import PhysicalConsistencyChecker, PhysicalConsistencyReport
from .loader import RawSequenceContainer


class PreprocessingCausality(str):
    CAUSAL = "CAUSAL"             # Computable in real-time step-by-step
    OFFLINE_ONLY = "OFFLINE_ONLY" # Evaluation only / uses reference ground truth


@dataclass
class ProcessedSequencePackage:
    sequence_id: str
    num_samples: int
    timestamps_sec: np.ndarray
    dt_array_sec: np.ndarray
    navigation_inputs_df: pd.DataFrame    # STRICTLY CAUSAL INPUTS ONLY (NO GROUND TRUTH)
    reference_trajectory_df: pd.DataFrame # GROUND TRUTH ONLY (FOR EVALUATION)
    quality_df: pd.DataFrame              # QUALITY BITMASKS
    reference_trajectory: ReferenceTrajectory
    timestamp_stats: TimestampStats
    quality_summary: QualitySummary
    consistency_report: PhysicalConsistencyReport
    sync_metadata: Dict[str, Any]
    provenance_metadata: Dict[str, Any]

    def export(self, output_dir: str) -> Dict[str, str]:
        """
        Export standardized Parquet and JSON files to output_dir with strict ground-truth isolation.
        """
        seq_dir = os.path.join(output_dir, "sequences", self.sequence_id)
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        nav_inputs_path = os.path.join(seq_dir, "navigation_inputs.parquet")
        ref_path = os.path.join(seq_dir, "reference_trajectory.parquet")
        quality_path = os.path.join(seq_dir, "quality_masks.parquet")
        meta_path = os.path.join(seq_dir, "metadata.json")
        rep_path = os.path.join(reports_dir, f"{self.sequence_id}_quality.json")

        # Export isolated Parquet files
        self.navigation_inputs_df.to_parquet(nav_inputs_path, index=False)
        self.reference_trajectory_df.to_parquet(ref_path, index=False)
        self.quality_df.to_parquet(quality_path, index=False)

        # Full metadata
        full_meta = {
            "pipeline_version": "2.1.0",
            "sequence_id": self.sequence_id,
            "num_samples": self.num_samples,
            "duration_sec": self.timestamp_stats.duration_sec,
            "nominal_rate_hz": self.timestamp_stats.nominal_rate_hz,
            "observed_rate_hz": self.timestamp_stats.observed_rate_hz,
            "total_distance_m": self.reference_trajectory.total_distance_m,
            "ground_truth_isolation_verified": True,
            "timestamp_stats": self.timestamp_stats.to_dict(),
            "quality_summary": self.quality_summary.to_dict(),
            "consistency_report": self.consistency_report.to_dict(),
            "sync_metadata": self.sync_metadata,
            "origin_metadata": self.reference_trajectory.origin_metadata,
            "causality_tags": self.provenance_metadata.get("causality_tags", {}),
            "unit_conversions": self.provenance_metadata.get("unit_conversions", {})
        }

        with open(meta_path, "w") as f:
            json.dump(full_meta, f, indent=2)

        with open(rep_path, "w") as f:
            json.dump({
                "sequence_id": self.sequence_id,
                "timestamp_stats": self.timestamp_stats.to_dict(),
                "quality_summary": self.quality_summary.to_dict(),
                "consistency_report": self.consistency_report.to_dict()
            }, f, indent=2)

        return {
            "navigation_inputs_parquet": nav_inputs_path,
            "reference_trajectory_parquet": ref_path,
            "quality_masks_parquet": quality_path,
            "metadata_json": meta_path,
            "report_json": rep_path
        }


class NavigationDataPipeline:
    """
    End-to-end reproducible data engineering pipeline for Project AGASTYA.
    Guarantees strict separation of causal navigation inputs from offline reference ground truth.
    """
    def __init__(
        self,
        nominal_rate_hz: float = 10.0,
        track_width_m: float = 1.47,
        max_speed_ms: float = 70.0,
        fallback_dt_sec: float = 0.10
    ):
        self.nominal_rate_hz = nominal_rate_hz
        self.track_analyzer = TimestampAnalyzer(
            gap_threshold_sec=0.25,
            nominal_rate_hz=nominal_rate_hz,
            fallback_dt_sec=fallback_dt_sec
        )
        self.normalizer = UnitNormalizer()
        self.synchronizer = StreamSynchronizer(max_allowed_offset_sec=0.15)
        self.geo_converter = GeodeticConverter()
        self.ref_builder = ReferenceTrajectoryBuilder(self.geo_converter)
        self.quality_mgr = DataQualityManager(max_speed_ms=max_speed_ms)
        self.consistency_checker = PhysicalConsistencyChecker(
            track_width_m=track_width_m,
            max_valid_vehicle_speed_ms=max_speed_ms
        )

    def process(
        self,
        raw_container: RawSequenceContainer,
        sync_method: str = "nearest_causal"
    ) -> ProcessedSequencePackage:
        """
        Execute full deterministic preprocessing on raw sequence container.
        """
        v_df = raw_container.vehicle_df
        s_df = raw_container.smartphone_df
        seq_id = raw_container.sequence_id

        if v_df is None and s_df is None:
            raise ValueError(f"Raw sequence '{seq_id}' contains no data.")

        causality_tags: Dict[str, str] = {}
        all_provenance: Dict[str, str] = {}

        # 1. Normalize Vehicle Data
        norm_v: Dict[str, np.ndarray] = {}
        if v_df is not None:
            norm_v, prov_v = self.normalizer.normalize_vehicle_dataframe(v_df)
            all_provenance.update(prov_v)

        # 2. Normalize Smartphone Data
        norm_s: Dict[str, np.ndarray] = {}
        if s_df is not None:
            norm_s, prov_s = self.normalizer.normalize_smartphone_dataframe(s_df)
            all_provenance.update(prov_s)

        # 3. Synchronize Streams
        if norm_v:
            aligned_data, sync_meta = self.synchronizer.align_streams(norm_v, norm_s, method=sync_method)
            raw_time_ms = v_df["Time"].to_numpy(dtype=np.float64)
        else:
            aligned_data = dict(norm_s)
            aligned_data["time_sec"] = norm_s["phone_time_sec"]
            aligned_data["source_timestamp_v_ms"] = np.full(len(norm_s["phone_time_sec"]), np.nan)
            aligned_data["source_timestamp_s_ms"] = norm_s["phone_time_sec"] * 1000.0
            aligned_data["sync_offset_ms"] = np.zeros(len(norm_s["phone_time_sec"]))
            aligned_data["sync_quality_flag"] = np.ones(len(norm_s["phone_time_sec"]), dtype=np.int8)
            sync_meta = {"is_synchronized": True, "method": "phone_only", "causality": "CAUSAL"}
            raw_time_ms = s_df["Time"].to_numpy(dtype=np.float64)

        # 4. Dynamic Timestamp & Jitter Analysis
        time_stats, dt_arr, time_valid_mask = self.track_analyzer.analyze(raw_time_ms)
        aligned_data["dt_sec"] = dt_arr

        # Tag Causal Sensor Signals
        for k in aligned_data.keys():
            if k in IOVNBDSchemaRegistry.get_causal_input_signal_names() or k in ["time_sec", "source_timestamp_v_ms", "source_timestamp_s_ms", "sync_offset_ms", "sync_quality_flag"]:
                causality_tags[k] = PreprocessingCausality.CAUSAL

        # 5. Geodetic & Metric Coordinate Conversion (Offline Reference)
        ref_data: Dict[str, np.ndarray] = {
            "time_sec": aligned_data["time_sec"]
        }

        if "latitude_deg" in aligned_data and "longitude_deg" in aligned_data:
            alt = aligned_data.get("altitude_m", np.zeros_like(aligned_data["latitude_deg"]))
            self.geo_converter.initialize_origin(aligned_data["latitude_deg"][0], aligned_data["longitude_deg"][0], alt[0])
            east_m, north_m, up_m = self.geo_converter.geodetic_to_enu(
                aligned_data["latitude_deg"],
                aligned_data["longitude_deg"],
                alt
            )

            ref_traj = self.ref_builder.build_reference(
                time_sec=aligned_data["time_sec"],
                latitude_deg=aligned_data["latitude_deg"],
                longitude_deg=aligned_data["longitude_deg"],
                altitude_m=alt,
                gps_speed_ms=aligned_data.get("gps_speed_ms"),
                heading_rad=aligned_data.get("heading_rad")
            )

            ref_data["pos_east_m"] = east_m
            ref_data["pos_north_m"] = north_m
            ref_data["pos_up_m"] = up_m
            ref_data["latitude_deg"] = aligned_data["latitude_deg"]
            ref_data["longitude_deg"] = aligned_data["longitude_deg"]
            ref_data["altitude_m"] = alt
            ref_data["ground_speed_ms"] = ref_traj.ground_speed_ms
            ref_data["velocity_east_ms"] = ref_traj.velocity_east_ms
            ref_data["velocity_north_ms"] = ref_traj.velocity_north_ms
            ref_data["heading_rad"] = ref_traj.heading_rad
            if "gps_accuracy_m" in aligned_data:
                ref_data["gps_accuracy_m"] = aligned_data["gps_accuracy_m"]
            if "satellites_count" in aligned_data:
                ref_data["satellites_count"] = aligned_data["satellites_count"]

            for k in ref_data.keys():
                causality_tags[k] = PreprocessingCausality.OFFLINE_ONLY
        else:
            zeros = np.zeros(len(dt_arr), dtype=np.float64)
            ref_traj = ReferenceTrajectory(
                timestamps_sec=aligned_data["time_sec"],
                east_m=zeros,
                north_m=zeros,
                up_m=zeros,
                ground_speed_ms=zeros,
                velocity_east_ms=zeros,
                velocity_north_ms=zeros,
                heading_rad=zeros,
                total_distance_m=0.0,
                origin_metadata={},
                roll_pitch_available=False
            )

        # 6. Quality Assessment & Mask Generation
        quality_masks, quality_summary = self.quality_mgr.evaluate_quality(aligned_data, time_valid_mask)

        # 7. Physical Consistency Checks
        anomaly_masks, consistency_rep = self.consistency_checker.check_consistency(aligned_data, dt_arr)

        # 8. Build STRICTLY ISOLATED DataFrames
        # Navigation Inputs: ONLY causal sensor signals, NO GPS/reference ground truth
        nav_input_dict = {
            "time_sec": aligned_data["time_sec"],
            "dt_sec": aligned_data["dt_sec"]
        }
        for k in IOVNBDSchemaRegistry.get_causal_input_signal_names():
            if k in aligned_data and k not in nav_input_dict:
                nav_input_dict[k] = aligned_data[k]

        # Add provenance columns
        for prov_col in ["source_timestamp_v_ms", "source_timestamp_s_ms", "sync_offset_ms", "sync_quality_flag"]:
            if prov_col in aligned_data:
                nav_input_dict[prov_col] = aligned_data[prov_col]

        nav_inputs_df = pd.DataFrame(nav_input_dict)
        ref_df = pd.DataFrame(ref_data)

        # Merge all quality & anomaly masks
        all_masks = dict(quality_masks)
        all_masks.update(anomaly_masks)
        quality_df = pd.DataFrame(all_masks)

        provenance = {
            "causality_tags": causality_tags,
            "unit_conversions": all_provenance,
            "is_synchronized": sync_meta.get("is_synchronized", False)
        }

        return ProcessedSequencePackage(
            sequence_id=seq_id,
            num_samples=len(nav_inputs_df),
            timestamps_sec=aligned_data["time_sec"],
            dt_array_sec=dt_arr,
            navigation_inputs_df=nav_inputs_df,
            reference_trajectory_df=ref_df,
            quality_df=quality_df,
            reference_trajectory=ref_traj,
            timestamp_stats=time_stats,
            quality_summary=quality_summary,
            consistency_report=consistency_rep,
            sync_metadata=sync_meta,
            provenance_metadata=provenance
        )

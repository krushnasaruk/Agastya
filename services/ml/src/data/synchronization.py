"""
Temporal Stream Synchronization Engine for IO-VNBD Data Engineering.
Supports pairwise synchronized file binding, timestamp cross-matching,
and strict causal alignment with complete provenance metadata.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class StreamSynchronizer:
    """
    Synchronizes Vehicle CAN and Smartphone sensor streams while strictly maintaining causality.
    """
    def __init__(self, max_allowed_offset_sec: float = 0.15):
        self.max_allowed_offset_sec = max_allowed_offset_sec

    def align_streams(
        self,
        vehicle_data: Dict[str, np.ndarray],
        phone_data: Optional[Dict[str, np.ndarray]] = None,
        method: str = "nearest_causal"
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Align Vehicle and Smartphone streams to a unified reference timeline.

        Parameters:
            vehicle_data: Dict of normalized vehicle arrays (must contain 'time_sec')
            phone_data: Optional dict of normalized phone arrays (must contain 'phone_time_sec')
            method: 'nearest_causal' (strictly causal) or 'linear_offline' (offline interpolation)

        Returns:
            aligned_data: Merged dictionary of aligned arrays with provenance columns
            sync_metadata: Provenance and offset diagnostics
        """
        if "time_sec" not in vehicle_data:
            raise KeyError("Vehicle data dictionary must contain 'time_sec' for synchronization.")

        ref_time = vehicle_data["time_sec"]
        n_ref = len(ref_time)
        aligned: Dict[str, np.ndarray] = dict(vehicle_data)

        # Baseline vehicle provenance
        aligned["source_timestamp_v_ms"] = ref_time * 1000.0

        if phone_data is None or len(phone_data) == 0:
            aligned["source_timestamp_s_ms"] = np.full(n_ref, np.nan)
            aligned["sync_offset_ms"] = np.zeros(n_ref)
            aligned["sync_quality_flag"] = np.zeros(n_ref, dtype=np.int8)

            return aligned, {
                "is_synchronized": True,
                "phone_stream_present": False,
                "num_samples": n_ref,
                "method": "vehicle_only",
                "causality": "CAUSAL",
                "mean_temporal_offset_ms": 0.0,
                "max_temporal_offset_ms": 0.0
            }

        phone_time = phone_data.get("phone_time_sec")
        if phone_time is None:
            raise KeyError("Phone data dictionary must contain 'phone_time_sec' for synchronization.")

        n_phone = len(phone_time)

        # Check if direct 1-to-1 index matching applies (standard in IO-VNBD Synchronised folder)
        if n_ref == n_phone and np.allclose(ref_time, phone_time, atol=0.015):
            for k, v in phone_data.items():
                if k != "phone_time_sec":
                    aligned[k] = v.copy()

            offsets_ms = np.abs(ref_time - phone_time) * 1000.0
            aligned["source_timestamp_s_ms"] = phone_time * 1000.0
            aligned["sync_offset_ms"] = offsets_ms
            aligned["sync_quality_flag"] = (offsets_ms <= self.max_allowed_offset_sec * 1000.0).astype(np.int8)

            return aligned, {
                "is_synchronized": True,
                "phone_stream_present": True,
                "num_samples": n_ref,
                "method": "direct_verified_match",
                "causality": "CAUSAL",
                "mean_temporal_offset_ms": round(float(np.mean(offsets_ms)), 3),
                "max_temporal_offset_ms": round(float(np.max(offsets_ms)), 3)
            }

        # Causal Alignment (Strictly: latest phone sample with t_phone <= t_ref[k])
        matched_indices = []
        offsets_sec = []

        if method == "nearest_causal":
            phone_idx = 0
            for k in range(n_ref):
                t_target = ref_time[k]
                # Advance phone pointer strictly while phone_time <= t_target
                while phone_idx + 1 < n_phone and phone_time[phone_idx + 1] <= t_target:
                    phone_idx += 1

                offset = float(np.abs(t_target - phone_time[phone_idx]))
                matched_indices.append(phone_idx)
                offsets_sec.append(offset)

            matched_idx_arr = np.array(matched_indices, dtype=np.int32)
            for k, v in phone_data.items():
                if k != "phone_time_sec":
                    aligned[k] = v[matched_idx_arr].copy()

            aligned["source_timestamp_s_ms"] = phone_time[matched_idx_arr] * 1000.0
            causality_type = "CAUSAL"

        elif method == "linear_offline":
            # Offline interpolation (USES FUTURE SAMPLES - NOT CAUSAL)
            for k, v in phone_data.items():
                if k != "phone_time_sec" and np.issubdtype(v.dtype, np.floating):
                    aligned[k] = np.interp(ref_time, phone_time, v)

            offsets_sec = [0.0] * n_ref
            aligned["source_timestamp_s_ms"] = ref_time * 1000.0
            causality_type = "OFFLINE_ONLY"

        else:
            raise ValueError(f"Unknown synchronization method: {method}")

        offsets_arr = np.array(offsets_sec) * 1000.0
        aligned["sync_offset_ms"] = offsets_arr
        aligned["sync_quality_flag"] = (offsets_arr <= self.max_allowed_offset_sec * 1000.0).astype(np.int8)

        sync_meta = {
            "is_synchronized": bool(np.all(np.array(offsets_sec) <= self.max_allowed_offset_sec)),
            "phone_stream_present": True,
            "num_samples": n_ref,
            "method": method,
            "causality": causality_type,
            "mean_temporal_offset_ms": round(float(np.mean(offsets_arr)), 3),
            "max_temporal_offset_ms": round(float(np.max(offsets_arr)), 3)
        }

        return aligned, sync_meta

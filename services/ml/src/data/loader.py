"""
IO-VNBD Raw Dataset Loader & Discovery Module for Project AGASTYA.
Discovers and loads raw CSV sequences across Vehicle, Smartphone, and
Synchronized dataset subdirectories without mutating raw files.
"""

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from .schema import IOVNBDSchemaRegistry


@dataclass
class RawSequenceContainer:
    sequence_id: str
    vehicle_df: Optional[pd.DataFrame] = None
    smartphone_df: Optional[pd.DataFrame] = None
    is_synchronized_pair: bool = False
    vehicle_file_path: Optional[str] = None
    smartphone_file_path: Optional[str] = None
    schema_validation: Dict[str, Any] = None


class IOVNBDDataLoader:
    """
    Discovers, inspects, and loads raw IO-VNBD dataset sequences.
    """
    def __init__(self, data_root_dir: str):
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.sync_dir = os.path.join(self.data_root_dir, "Synchronised V and S datasets")
        self.v_dir = os.path.join(self.data_root_dir, "V-datasets")
        self.s_dir = os.path.join(self.data_root_dir, "S-datasets")

    def discover_sequences(self) -> Dict[str, Dict[str, str]]:
        """
        Scan directory hierarchy and return mapping of sequence_id -> file paths.
        """
        sequences: Dict[str, Dict[str, str]] = {}

        # 1. Scan Synchronised Folder (Priority for multimodal DR)
        if os.path.exists(self.sync_dir):
            v_sync_files = glob.glob(os.path.join(self.sync_dir, "V_dataset_*.csv")) + glob.glob(os.path.join(self.sync_dir, "V_*.csv"))
            for v_path in v_sync_files:
                base_name = os.path.basename(v_path)
                # Extract ID (e.g. V_dataset_01.csv -> 01)
                seq_id = base_name.replace("V_dataset_", "").replace("V_", "").replace(".csv", "")
                s_name = f"S_dataset_{seq_id}.csv"
                s_path = os.path.join(self.sync_dir, s_name)
                if not os.path.exists(s_path):
                    s_path = os.path.join(self.sync_dir, f"S_{seq_id}.csv")

                sequences[f"sync_{seq_id}"] = {
                    "vehicle": v_path,
                    "smartphone": s_path if os.path.exists(s_path) else None,
                    "type": "synchronized_pair"
                }

        # 2. Scan Standalone V-datasets
        if os.path.exists(self.v_dir):
            v_files = glob.glob(os.path.join(self.v_dir, "*.csv"))
            for v_path in v_files:
                base_name = os.path.basename(v_path)
                seq_id = base_name.replace("V_dataset_", "").replace("V_", "").replace(".csv", "")
                key = f"v_standalone_{seq_id}"
                if key not in sequences:
                    sequences[key] = {
                        "vehicle": v_path,
                        "smartphone": None,
                        "type": "vehicle_standalone"
                    }

        # 3. Scan Standalone S-datasets
        if os.path.exists(self.s_dir):
            s_files = glob.glob(os.path.join(self.s_dir, "*.csv"))
            for s_path in s_files:
                base_name = os.path.basename(s_path)
                seq_id = base_name.replace("S_dataset_", "").replace("S_", "").replace(".csv", "")
                key = f"s_standalone_{seq_id}"
                if key not in sequences:
                    sequences[key] = {
                        "vehicle": None,
                        "smartphone": s_path,
                        "type": "smartphone_standalone"
                    }

        return sequences

    def load_raw_sequence(self, sequence_id: str) -> RawSequenceContainer:
        """
        Load raw DataFrames for a given sequence_id without altering source files.
        """
        discovered = self.discover_sequences()
        if sequence_id not in discovered:
            # Fallback: check if direct path or simplified id
            matching = [k for k in discovered if sequence_id in k]
            if matching:
                sequence_id = matching[0]
            else:
                raise FileNotFoundError(f"Sequence ID '{sequence_id}' not found in {self.data_root_dir}. Available: {list(discovered.keys())}")

        info = discovered[sequence_id]
        v_df = None
        s_df = None
        v_path = info["vehicle"]
        s_path = info["smartphone"]

        schema_report: Dict[str, Any] = {}

        if v_path and os.path.exists(v_path):
            v_df = pd.read_csv(v_path)
            schema_report["vehicle"] = IOVNBDSchemaRegistry.validate_columns(list(v_df.columns), "vehicle")

        if s_path and os.path.exists(s_path):
            s_df = pd.read_csv(s_path)
            schema_report["smartphone"] = IOVNBDSchemaRegistry.validate_columns(list(s_df.columns), "smartphone")

        return RawSequenceContainer(
            sequence_id=sequence_id,
            vehicle_df=v_df,
            smartphone_df=s_df,
            is_synchronized_pair=(v_df is not None and s_df is not None),
            vehicle_file_path=v_path,
            smartphone_file_path=s_path,
            schema_validation=schema_report
        )

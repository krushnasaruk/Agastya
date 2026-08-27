"""
Sequence-Level Dataset Partitioning for Project AGASTYA (Objective 4).
Enforces trajectory-level dataset splits to eliminate temporal auto-correlation leakage.
Random timestep-level shuffling across sequences is strictly forbidden.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


@dataclass
class SequenceSplitConfig:
    train_sequences: List[str]
    val_sequences: List[str]
    test_sequences: List[str]
    split_strategy: str = "TRAJECTORY_LEVEL_DISJOINT"
    notes: str = "Held-out test sequences remain untouched until final blind evaluation."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetSplitManager:
    """
    Manages sequence-level partitioning for IO-VNBD datasets.
    """
    DEFAULT_SPLIT = SequenceSplitConfig(
        train_sequences=["sync_01"],
        val_sequences=["v_standalone_03"],
        test_sequences=["sync_02"],
        split_strategy="TRAJECTORY_LEVEL_DISJOINT",
        notes="Sequence sync_02 is held out as the blind evaluation benchmark."
    )

    @classmethod
    def get_default_split(cls) -> SequenceSplitConfig:
        return cls.DEFAULT_SPLIT

    @classmethod
    def validate_no_leakage(cls, split_config: SequenceSplitConfig) -> bool:
        """
        Verify that training, validation, and test sets are strictly disjoint.
        """
        train_set = set(split_config.train_sequences)
        val_set = set(split_config.val_sequences)
        test_set = set(split_config.test_sequences)

        if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
            raise ValueError("Data Leakage Detected: Sequence overlap between train, val, or test sets.")

        return True

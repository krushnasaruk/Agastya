"""
AI Residual Learning Subsystem for Project AGASTYA (Objective 5).
"""

from .feature_registry import CANONICAL_FEATURES, CANONICAL_FEATURE_NAMES, NUM_CANONICAL_FEATURES, validate_feature_matrix_columns
from .scaler import TrainOnlyScaler, TargetScaler
from .dataset import CausalWindowDataset
from .model import CausalResidualGRU
from .safety import SafetyGuard, GuardedResidual
from .trainer import ResidualModelTrainer, set_seed
from .evaluator import ResidualEvaluator, ResidualMetrics
from .rollout import AIRolloutEngine
from .outage_eval import OutageComparator
from .ablations import AblationRunner
from .diagnostics import Objective5Visualizer

__all__ = [
    "CANONICAL_FEATURES",
    "CANONICAL_FEATURE_NAMES",
    "NUM_CANONICAL_FEATURES",
    "validate_feature_matrix_columns",
    "TrainOnlyScaler",
    "TargetScaler",
    "CausalWindowDataset",
    "CausalResidualGRU",
    "SafetyGuard",
    "GuardedResidual",
    "ResidualModelTrainer",
    "set_seed",
    "ResidualEvaluator",
    "ResidualMetrics",
    "AIRolloutEngine",
    "OutageComparator",
    "AblationRunner",
    "Objective5Visualizer"
]

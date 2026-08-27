"""
Causal Maneuver Classifier for Objective 6.
Stratifies navigation timesteps into physical driving regimes using strictly causal onboard telemetry.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class CausalManeuverClassifier:
    """
    Classifies timesteps into distinct driving regimes without future lookahead.
    """
    STATIONARY = "stationary"
    STRAIGHT = "straight"
    MODERATE_TURN = "moderate_turn"
    AGGRESSIVE_TURN = "aggressive_turn"
    ACCELERATION = "acceleration"
    BRAKING = "braking"
    SLIP_LIKE = "slip_like"

    ALL_MANEUVERS = [
        STATIONARY,
        STRAIGHT,
        MODERATE_TURN,
        AGGRESSIVE_TURN,
        ACCELERATION,
        BRAKING,
        SLIP_LIKE
    ]

    @classmethod
    def classify_sequence(
        cls,
        speed_ms: np.ndarray,
        accel_x_ms2: np.ndarray,
        yaw_rate_rads: np.ndarray,
        slip_detected_flags: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Classify an entire sequence of causal telemetry into maneuver labels.
        """
        n = len(speed_ms)
        labels = np.empty(n, dtype=object)
        slips = slip_detected_flags if slip_detected_flags is not None else np.zeros(n, dtype=bool)

        for i in range(n):
            v = speed_ms[i]
            ax = accel_x_ms2[i]
            yr = abs(yaw_rate_rads[i])
            is_slip = bool(slips[i])

            if v < 0.08:
                labels[i] = cls.STATIONARY
            elif is_slip:
                labels[i] = cls.SLIP_LIKE
            elif yr >= 0.20:
                labels[i] = cls.AGGRESSIVE_TURN
            elif yr >= 0.05:
                labels[i] = cls.MODERATE_TURN
            elif ax >= 0.40:
                labels[i] = cls.ACCELERATION
            elif ax <= -0.40:
                labels[i] = cls.BRAKING
            else:
                labels[i] = cls.STRAIGHT

        return labels

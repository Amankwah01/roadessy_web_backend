# app/services/motion_stability_service.py

from collections import deque
from typing import Deque

import numpy as np


class MotionStabilityService:
    """Track recent motion scores and compute a bounded stability value.

    History keeps the last 10 scores. Stability is computed as
    1 / (1 + std(history)) so results are in (0, 1] and well-behaved.
    """

    def __init__(self) -> None:
        self.history: Deque[float] = deque(maxlen=10)

    def add_motion_score(self, score: float) -> None:
        """Add a new motion score.

        Args:
            score: numeric motion score.

        Raises:
            TypeError: if `score` cannot be converted to float.
        """
        try:
            f = float(score)
        except (TypeError, ValueError) as exc:
            raise TypeError("score must be a numeric value") from exc
        self.history.append(f)

    def get_stability(self) -> float:
        """Return a stability metric in (0, 1].

        Returns 1.0 when there are fewer than 3 samples.
        """
        if len(self.history) < 3:
            return 1.0
        std = float(np.std(list(self.history)))
        return float(1.0 / (1.0 + std))

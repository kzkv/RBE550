# Tom Kazakov
# RBE 550
# Assignment 4, Valet

import math
from typing import List, Tuple, Optional
import numpy as np
from world import Pos


def plan(origin: Pos, goal: Pos, obstacles: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Plan a kinematically feasible path from start to goal."""

    # Simple 3-point path
    path = [
        (origin.x, origin.y),
        ((origin.x + goal.x) / 2, (origin.y + goal.y) / 2),
        (goal.x, goal.y),
    ]

    return path

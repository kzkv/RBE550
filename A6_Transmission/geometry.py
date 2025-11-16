"""
Minimal transmission geometry uses stacked cylinders for shaft definitions
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CylinderSegment:
    diameter: float  # mm
    width: float  # mm


class Shaft:
    """
    Represents a shaft as a stack of cylinder segments
    Automatically computes positions from stacking
    """

    def __init__(self, segments: List[Tuple[float, float]], name: str = ""):
        """Initialize a shaft from a list of (diameter, width) tuples"""
        self.name = name
        self.segments = [CylinderSegment(d, w) for d, w in segments]
        self._compute_positions()

    def _compute_positions(self):
        """Compute center positions of each segment from stacking"""
        self.positions = []
        current_x = 0

        for seg in self.segments:
            # Center of this segment
            center_x = current_x + seg.width / 2
            self.positions.append(center_x)
            # Move to the end of this segment
            current_x += seg.width

        self.total_length = current_x

    def get_cylinders(self, offset: np.ndarray = np.array([0, 0, 0])) -> List[dict]:
        """Get cylinder data for rendering"""
        cylinders = []
        for seg, pos_x in zip(self.segments, self.positions):
            cylinders.append(
                {
                    "center": np.array([pos_x, 0, 0]) + offset,
                    "radius": seg.diameter / 2,
                    "length": seg.width,
                }
            )
        return cylinders


class Transmission:
    def __init__(self):
        """Initialize transmission with mainshaft, countershaft, and case"""

        self.mainshaft = Shaft(
            [  # diameter, width
                (36, 20),  # Front bearing journal
                (90, 8),  # Gear
                (120, 30),  # Clutch
                (90, 33),  # Two gears
                (106, 25),  # Gear
                (96, 8),  # Gear
                (120, 25),  # Gear
                (120, 30),  # Clutch
                (96, 8),  # Gear
                (130, 25),  # Gear
                (36, 118),  # Rear bearing journal
            ],
            name="mainshaft",
        )

        self.countershaft = Shaft(
            [  # diameter, width
                (36, 30),  # Front journal
                (140, 25),  # Gear
                (36, 91),  # Shaft segment  FIX lengh
                (140, 25),  # Gear
                (125, 25),  # Gear
                (36, 8),  # Shaft segment
                (80, 25),  # Gear
                (46, 38),  # Shaft segment
                (100, 25),  # Gear
                (36, 38),  # Rear journal
            ],
            name="countershaft",
        )

        # Countershaft vertical offset (below mainshaft)
        self.countershaft_offset = np.array([-25, 0, -50])

        self.case_inner_dims = {
            "length": 280,  # x
            "width": 160,  # y
            "height": 300,  # z
        }

        # Initial mainshaft position
        self.mainshaft_initial_pos = np.array([55, 0, 65])

    def get_mainshaft_at(self, position: np.ndarray) -> List[dict]:
        """Get mainshaft cylinders at a specified position"""
        return self.mainshaft.get_cylinders(position)

    def get_countershaft(self) -> List[dict]:
        """Get countershaft cylinders (stationary)"""
        return self.countershaft.get_cylinders(self.countershaft_offset)

    def get_case_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a case as two corner points"""

        min_corner = np.array(
            [0, -self.case_inner_dims["width"] / 2, -self.case_inner_dims["height"] / 2]
        )
        max_corner = np.array(
            [
                self.case_inner_dims["length"],
                self.case_inner_dims["width"] / 2,
                self.case_inner_dims["height"] / 2,
            ]
        )
        return min_corner, max_corner

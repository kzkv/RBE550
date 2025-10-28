# firetruck.py
import logging
import math
from typing import List, Tuple

import pygame

from world import Pos, World

logger = logging.getLogger(__name__)

INITIAL_HEADING = math.pi / 4


class Firetruck:
    """
    Mercedes Unimog firetruck robot with Ackerman steering.
    Basic stub implementation with pose management and rendering.
    """

    def __init__(self, world: 'World', preset_rows: tuple[int, int], preset_cols: tuple[int, int]):
        self.world = world

        location = self.world.field.initialize_position(preset_rows, preset_cols)
        self.pos = self.grid_to_pose(location, INITIAL_HEADING)

        # Truck specifications
        self.length = 4.9  # m
        self.width = 2.2  # m
        self.wheelbase = 3.0  # m
        self.min_turning_radius = 13.0  # m
        self.max_velocity = 10.0  # m/s

    def grid_to_pose(self, grid_pos: tuple[int, int], heading: float) -> Pos:
        """Convert grid position from field.initialize_position() to a Pos with heading"""
        row, col = grid_pos
        x, y = self.world.grid_to_world(row, col)
        return Pos(x=x, y=y, heading=heading)

    def set_pose(self, pos: Pos):
        """Set the firetruck pose"""
        self.pos = pos
        logger.debug(f"Firetruck pose set to {pos}")

    def update(self, dt: float = None):
        """Update the firetruck state"""
        # TODO: Implement kinematics and control
        # TODO: Make sure to use the world time
        pass

    def render(self):
        """Render vehicle at current or specified position"""
        FIRETRUCK_COLOR = (220, 50, 50)  # Red for firetruck
        FIRETRUCK_STRIPE_COLOR = (255, 255, 255)  # White stripe

        ppm = self.world.pixels_per_meter
        Lpx = math.floor(self.length * ppm) - 1  # Magical size optimized for the 7 PPM
        Wpx = math.floor(self.width * ppm)

        surf = pygame.Surface((Lpx, Wpx), pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            FIRETRUCK_COLOR,
            pygame.Rect(0, 0, Lpx, Wpx),
            border_radius=max(2, Wpx // 4),
        )
        # Indicate front
        stripe_w = 3  # px
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(surf, FIRETRUCK_STRIPE_COLOR, (stripe_x, 3, stripe_w, Wpx - 6), border_radius=2)

        # Rotate to the current heading
        rotated = pygame.transform.rotate(surf, -math.degrees(self.pos.heading))

        px = int(self.pos.x * ppm)
        py = int(self.pos.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        self.world.display.blit(rotated, rect.topleft)

    def render_route(self, route: List[Pos], color: Tuple[int, int, int]):
        if len(route) < 2:
            return  # Need at least 2 points to draw a line
        pts = [(int(pos.x * self.world.pixels_per_meter), int(pos.y * self.world.pixels_per_meter)) for pos in route]
        pygame.draw.lines(self.world.display, color, False, pts, 2)

# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for ideation

from dataclasses import dataclass
from typing import List, Tuple, Union
import math
import pygame

# Reuse helpers from your world module
from world import grid_to_world, world_to_grid, World

VEHICLE_BG_COLOR = (56, 67, 205)


@dataclass(frozen=True)
class VehicleSpec:
    length: float  # m
    width: float  # m
    wheelbase: Union[float, None]  # m
    cargo_manifest: str
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec):
        self.spec = spec
        self.x = 5.0  # m
        self.y = 5.0  # m
        self.theta = 1.2  # rad

    def render(self, world: World):
        ppm = world.pixels_per_meter
        Lpx = int(self.spec.length * ppm)
        Wpx = int(self.spec.width * ppm)

        surf = pygame.Surface((Lpx, Wpx), pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            self.spec.color,
            pygame.Rect(0, 0, Lpx, Wpx),
            border_radius=max(2, Wpx // 4),
        )

        # rotate to the current heading
        rotated = pygame.transform.rotate(surf, -math.degrees(self.theta))

        px = int(self.x * ppm)
        py = int(self.y * ppm)
        rect = rotated.get_rect(center=(px, py))

        world.screen.blit(rotated, rect.topleft)

# TODO: establish the goal position


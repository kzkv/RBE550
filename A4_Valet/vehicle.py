# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for ideation

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import math
import pygame

# Reuse helpers from your world module
from world import grid_to_world, world_to_grid, World

VEHICLE_BG_COLOR = (56, 67, 205)
VEHICLE_FRONT_STRIPE_COLOR = (255, 255, 255, 180)

@dataclass(frozen=True)
class VehicleSpec:
    length: float  # m
    width: float  # m
    wheelbase: Union[float, None]  # m
    cargo_manifest: str
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec, origin: Tuple[float, float] = (0.0, 0.0), heading: float = 0):
        self.spec = spec
        self.x, self.y = origin  # m
        self.heading = heading  # rad

        self.velocity = 0.0  # TODO: refactor into max_velocity or target_velocity?
        self._destination: Optional[Tuple[float, float]] = None

    def set_destination(self, destination: Tuple[float, float]):
        self._destination = destination

    def are_we_there_yet(self) -> bool:
        ERROR_THRESHOLD = 0.1  # m
        return self._destination is not None and self.distance_to(self._destination) < ERROR_THRESHOLD

    def distance_to(self, point: Tuple[float, float]) -> float:
        dx, dy = point
        return math.hypot(self.x - dx, self.y - dy)

    def drive(self, delta_time: float):
        # delta_time makes driving dependent on time, not on the compute/frame rendering
        if self._destination is None or delta_time <= 0.0:
            return

        if self.are_we_there_yet():
            return

        distance = self.distance_to(self._destination)
        step = self.velocity * delta_time
        tx, ty = self._destination
        dx, dy = tx - self.x, ty - self.y

        # TODO: consider a helper method to change location
        if step >= distance:  # don't overshoot the destination
            self.x, self.y = tx, ty
            self._destination = None
        else:
            ux, uy = dx / distance, dy / distance
            self.x += ux * step
            self.y += uy * step

        # TODO: make the heading functional
        # face the direction of travel
        self.heading = math.atan2(dy, dx)

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
        # indicate front
        stripe_w = 4  # px
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(surf, VEHICLE_FRONT_STRIPE_COLOR, (stripe_x, 4, stripe_w, Wpx - 8), border_radius=2)

        # rotate to the current heading
        rotated = pygame.transform.rotate(surf, -math.degrees(self.heading))

        px = int(self.x * ppm)
        py = int(self.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        world.screen.blit(rotated, rect.topleft)

        if self._destination is not None:
            tx, ty = self._destination
            tpx, tpy = int(tx * ppm), int(ty * ppm)  # in pixels
            pygame.draw.circle(world.screen, (0, 255, 0), (tpx, tpy), 5)
            pygame.draw.line(world.screen, (0, 255, 0), (px, py), (tpx, tpy), 2)
            # TODO: extract cosmetics to constants

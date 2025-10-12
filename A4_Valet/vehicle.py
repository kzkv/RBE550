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
    track_width: float
    cargo_manifest: str
    cruising_velocity: float
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR
    front_stripe_color: Tuple[int, int, int] = VEHICLE_FRONT_STRIPE_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec, origin: Tuple[float, float] = (0.0, 0.0), heading: float = 0):
        self.spec = spec
        self.x, self.y = origin  # m
        self.heading = heading  # rad, 0 is along x-axis, CCW is positive

        self._destination: Optional[Tuple[float, float]] = None

        # Prescribed linear and angular velocities
        self._v = 0.0
        self._w = 0.0

    def set_destination(self, destination: Tuple[float, float]):
        self._destination = destination

    def distance_to(self, point: Tuple[float, float]) -> float:
        dx, dy = point
        return math.hypot(self.x - dx, self.y - dy)

    def _distance_to_destination(self) -> float:
        return self.distance_to(self._destination)

    def are_we_there_yet(self) -> bool:
        ERROR_THRESHOLD = 0.1  # m
        return self._destination is not None and self.distance_to(self._destination) < ERROR_THRESHOLD

    @staticmethod
    def _wrap_angle(angle):
        """Wrap to [-pi, pi] to ensure the minimal signed turn"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def controller(self):
        """P-controller"""

        if self._destination is None or self.are_we_there_yet():
            self._v, self._w = 0.0, 0.0
            return

        tx, ty = self._destination
        dx, dy = tx - self.x, ty - self.y

        heading_des = math.atan2(dy, dx)
        heading_err = self._wrap_angle(heading_des - self.heading)

        # Gains
        k_v = 1.0
        k_w = 1.0

        # Forward velocity
        # TODO: refactor the code below for clarity?
        self._v = min(self.spec.cruising_velocity, k_v * self._distance_to_destination()) * max(0.0,
                                                                                                math.cos(heading_err))
        self._w = k_w * heading_err

    def drive(self, delta_time: float):
        if self._destination is not None:
            if self.are_we_there_yet():
                self._destination = None
                self._v, self._w = 0.0, 0.0
                return
            self.controller()

        v, w = self._v, self._w
        th = self.heading

        # TODO: refactor for clarity
        if abs(w) < 1e-8:
            # straight segment
            self.x += v * math.cos(th) * delta_time
            self.y += v * math.sin(th) * delta_time
        else:
            # closed-form unicycle integration (stable arcs)
            th_new = th + w * delta_time
            self.x += (v / w) * (math.sin(th_new) - math.sin(th))
            self.y += (v / w) * (-math.cos(th_new) + math.cos(th))
            self.heading = th_new

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
        pygame.draw.rect(surf, self.spec.front_stripe_color, (stripe_x, 4, stripe_w, Wpx - 8), border_radius=2)

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

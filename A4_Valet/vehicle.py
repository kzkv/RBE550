# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for ideation

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import math
import pygame

# Reuse helpers from your world module
from world import World, Pos

VEHICLE_BG_COLOR = (56, 67, 205)
VEHICLE_FRONT_STRIPE_COLOR = (255, 255, 255, 180)
DESTINATION_COLOR = (0, 255, 0)
BREADCRUMB_COLOR = (255, 0, 0)


@dataclass(frozen=True)
class VehicleSpec:
    length: float  # m
    width: float  # m
    wheelbase: Union[float, None]  # m
    track_width: float
    cargo_manifest: str
    cruising_velocity: float  # m/s
    w_max: float  # rad/s
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR
    front_stripe_color: Tuple[int, int, int] = VEHICLE_FRONT_STRIPE_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec, origin: Pos):
        self.spec = spec
        self.pos = origin

        self._destination: Optional[Tuple[float, float]] = None

        # Prescribed linear and angular velocities
        self._v = 0.0
        self._w = 0.0

        # Persistent breadcrumbs trail; stored in pixels to avoid recalculation in render.
        # Adds velocity to reflect how fast the robot was moving at each point.
        self.breadcrumbs: List[Tuple[Tuple[int, int], float]] = []

    def set_velocities(self, v: float, w: float):
        self._v, self._w = v, w

    @staticmethod
    def _wrap_angle(angle):
        """Wrap to [-pi, pi] to ensure the minimal signed turn"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def drive(self, delta_time: float, world: World):
        """Body commands (v, w) integrator"""
        v, w = self._v, self._w
        th = self.pos.heading

        if abs(w) < 1e-8:  # moving in a straight line
            self.pos.x += v * math.cos(th) * delta_time
            self.pos.y += v * math.sin(th) * delta_time
        else:  # unicycle update for constant v, w over delta_time
            th_new = th + w * delta_time
            self.pos.x += (v / w) * (math.sin(th_new) - math.sin(th))
            self.pos.y += (v / w) * (-math.cos(th_new) + math.cos(th))
            self.pos.heading = th_new

        self.pos.heading = self._wrap_angle(self.pos.heading)  # Keep heading in [-pi, pi]

        # Breadcrumbs update
        ppm = world.pixels_per_meter
        px, py = int(self.pos.x * ppm), int(self.pos.y * ppm)
        self.breadcrumbs.append(((px, py), self._v))

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
        rotated = pygame.transform.rotate(surf, -math.degrees(self.pos.heading))

        px = int(self.pos.x * ppm)
        py = int(self.pos.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        world.screen.blit(rotated, rect.topleft)

    def render_breadcrumbs(self, world: World):
        """max_velocity is not controlled for <=0"""
        for (px, py), v in self.breadcrumbs:
            pygame.draw.circle(world.screen, BREADCRUMB_COLOR, (px, py), max(1, int(v)))

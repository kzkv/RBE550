# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for ideation; Claude for rough code gen (deeply refactored)

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from enum import Enum
import math
import pygame

# Reuse helpers from your world module
from world import World, Pos

VEHICLE_BG_COLOR = (56, 67, 205)
VEHICLE_FRONT_STRIPE_COLOR = (255, 255, 255, 180)
DESTINATION_COLOR = (90, 200, 90, 100)
BREADCRUMB_COLOR = (255, 0, 0)


class KinematicModel(Enum):
    DIFF_DRIVE = "diff_drive"
    ACKERMANN = "ackermann"


@dataclass(frozen=True)
class VehicleSpec:
    length: float  # m
    width: float  # m
    wheelbase: Union[float, None]  # m; required for Ackermann, ignored for diff-drive
    track_width: float
    cargo_manifest: str
    cruising_velocity: float  # m/s
    w_max: float  # rad/s
    max_steering_angle: float  # rad
    origin: Pos
    destination: Pos
    safety_margin: float
    planned_xy_error: float
    planned_heading_error: float
    kinematic_model: KinematicModel
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR
    front_stripe_color: Tuple[int, int, int] = VEHICLE_FRONT_STRIPE_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec):
        self.spec = spec
        self.pos = spec.origin

        self._destination: Optional[Tuple[float, float]] = None

        # For diff-drive: linear and angular velocities
        # For Ackermann: linear velocity and steering angle
        self._v = 0.0
        self._w_or_delta = 0.0  # w for diff-drive, delta (steering angle) for Ackermann

        # Persistent breadcrumbs trail; stored in pixels to avoid recalculation in render.
        # Adds velocity to reflect how fast the robot was moving at each point.
        self.breadcrumbs: List[Tuple[Tuple[int, int], float]] = []

    def set_velocities(self, v: float, w: float):
        """
        Set control inputs for the vehicle.
        For diff-drive: v is linear velocity, w is angular velocity
        For Ackermann: v is linear velocity, w is treated as desired angular velocity
                       (converted to steering angle internally)
        """
        self._v = v

        if self.spec.kinematic_model == KinematicModel.ACKERMANN:
            # Convert desired angular velocity to steering angle
            # For Ackermann: w = v * tan(delta) / L
            # Therefore: delta = atan(w * L / v)
            if abs(v) > 1e-6:
                tan_delta = w * self.spec.wheelbase / v
                # Clamp to steering limits
                tan_steering = math.tan(self.spec.max_steering_angle)
                tan_delta = max(-tan_steering, min(tan_steering, tan_delta))
                self._w_or_delta = math.atan(tan_delta)
            else:
                self._w_or_delta = 0.0
        else:
            self._w_or_delta = w

    @staticmethod
    def _wrap_angle(angle):
        """Wrap to [-pi, pi] to ensure the minimal signed turn"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def drive(self, delta_time: float, world: World):
        """Integrate vehicle motion based on kinematic model"""
        v = self._v
        th = self.pos.heading

        # determine angular velocity based on the kinematic model
        if self.spec.kinematic_model == KinematicModel.ACKERMANN:
            # For Ackermann: compute w from steering angle
            # w = v * tan(delta) / L
            delta = self._w_or_delta
            if abs(delta) < 1e-8:
                w = 0.0
            else:
                w = v * math.tan(delta) / self.spec.wheelbase
        else:
            # For diff-drive: w is directly commanded
            w = self._w_or_delta

        # integrate motion using unicycle model, applies regardless of kinematic model
        if abs(w) < 1e-8:  # straight line motion
            new_x = self.pos.x + v * math.cos(th) * delta_time
            new_y = self.pos.y + v * math.sin(th) * delta_time
            self.pos = Pos(new_x, new_y, th)
        else:  # circular arc motion
            th_new = th + w * delta_time
            new_x = self.pos.x + (v / w) * (math.sin(th_new) - math.sin(th))
            new_y = self.pos.y + (v / w) * (-math.cos(th_new) + math.cos(th))
            th_new = self._wrap_angle(th_new)
            self.pos = Pos(new_x, new_y, th_new)

        self._update_breadcrumbs(world)

    def _update_breadcrumbs(self, world: World):
        """Update breadcrumbs trail"""
        ppm = world.pixels_per_meter
        px, py = int(self.pos.x * ppm), int(self.pos.y * ppm)
        self.breadcrumbs.append(((px, py), self._v))

    def render(self, world: World, pos: Optional[Pos] = None):
        """Render vehicle at current or specified position with optional color override"""
        ppm = world.pixels_per_meter
        Lpx = int(self.spec.length * ppm)
        Wpx = int(self.spec.width * ppm)

        # Use the provided position or current position
        if pos is not None:
            render_pos = pos if pos is not None else self.pos
            render_color = DESTINATION_COLOR
        else:
            render_pos = self.pos
            render_color = VEHICLE_BG_COLOR

        surf = pygame.Surface((Lpx, Wpx), pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            render_color,
            pygame.Rect(0, 0, Lpx, Wpx),
            border_radius=max(2, Wpx // 4),
        )
        # indicate front
        stripe_w = 4  # px
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(surf, self.spec.front_stripe_color, (stripe_x, 4, stripe_w, Wpx - 8), border_radius=2)

        # rotate to the current heading
        rotated = pygame.transform.rotate(surf, -math.degrees(render_pos.heading))

        px = int(render_pos.x * ppm)
        py = int(render_pos.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        world.screen.blit(rotated, rect.topleft)

    def render_breadcrumbs(self, world: World):
        """max_velocity is not controlled for <=0"""
        for (px, py), v in self.breadcrumbs:
            pygame.draw.circle(world.screen, BREADCRUMB_COLOR, (px, py), max(1, int(v)))

    def render_parking_zone(self, world):
        """
        Render an acceptable parking zone as a circle. I'm using a liberal interpretation of
        "the exact boundary of this goal box is at your discretion" from the assignment.
        I like the circular box because it's so much easier to render than a rectangle.
        """

        ppm = world.pixels_per_meter
        dest_px = int(self.spec.destination.x * ppm)
        dest_py = int(self.spec.destination.y * ppm)

        # Radius is xy_error + the furthest corner of the vehicle from its center
        vehicle_corner_distance = math.hypot(self.spec.length / 2, self.spec.width / 2)
        radius = self.spec.planned_xy_error + vehicle_corner_distance
        radius_px = int(radius * ppm)

        pygame.draw.circle(world.screen, DESTINATION_COLOR, (dest_px, dest_py), radius_px, width=1)

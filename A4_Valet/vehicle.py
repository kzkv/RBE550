# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for ideation; Claude for rough code gen (deeply refactored)

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from enum import Enum
import math
import pygame

from world import World, Pos

VEHICLE_BG_COLOR = (56, 67, 205)
VEHICLE_FRONT_STRIPE_COLOR = (255, 255, 255, 180)
DESTINATION_COLOR = (90, 200, 90, 100)
BREADCRUMB_COLOR = (255, 0, 0)
TRAILER_BREADCRUMB_COLOR = (100, 100, 100)
HITCH_COLOR = (0, 0, 0)

EPSILON = 1e-8  # Numerical tolerance for floating-point comparisons


class KinematicModel(Enum):
    DIFF_DRIVE = "diff_drive"
    ACKERMANN = "ackermann"


@dataclass(frozen=True)
class TrailerSpec:
    """Specification for a trailer attached to a vehicle"""
    length: float  # m
    width: float  # m
    hitch_length: float  # m, distance from vehicle rear to trailer axle (d1)
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR


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
    max_acceleration: float  # m/s²
    origin: Pos
    destination: Pos
    safety_margin: float
    planned_xy_error: float
    planned_heading_error: float
    kinematic_model: KinematicModel
    trailer: Optional[TrailerSpec] = None  # Optional trailer specification
    color: Tuple[int, int, int] = VEHICLE_BG_COLOR
    front_stripe_color: Tuple[int, int, int] = VEHICLE_FRONT_STRIPE_COLOR


class Vehicle:
    def __init__(self, spec: VehicleSpec):
        self.spec = spec
        self.pos = spec.origin

        self._destination: Optional[Tuple[float, float]] = None

        # For diff-drive: linear and angular velocities
        # For Ackermann: linear velocity and steering angle
        self._v_desired = 0.0  # Desired velocity from the controller
        self._v_actual = 0.0  # Actual velocity (affected by acceleration limits)
        self._w_or_delta = 0.0  # w for diff-drive, delta (steering angle) for Ackermann

        # Trailer state (if equipped)
        self.trailer_heading: Optional[float] = None
        self.trailer_pos: Optional[Pos] = None
        if self.spec.trailer is not None:
            self.trailer_heading = spec.origin.heading
            self.trailer_pos = self._compute_trailer_position(self.trailer_heading, self.spec, self.pos)

        # Persistent breadcrumbs trail
        self.breadcrumbs: List[Tuple[Tuple[int, int], float]] = []
        self.trailer_breadcrumbs: List[Tuple[Tuple[int, int], float]] = []

    def set_velocities(self, v: float, w: float):
        """
        Set control inputs for the vehicle.
        For diff-drive: v is linear velocity, w is angular velocity
        For Ackermann: v is linear velocity, w is treated as desired angular velocity
                       (converted to the steering angle internally)
        """
        self._v_desired = v

        if self.spec.kinematic_model == KinematicModel.ACKERMANN:
            # Convert desired angular velocity to steering angle
            # For Ackermann: w = v * tan(delta) / L
            # Therefore: delta = atan(w * L / v)
            if abs(v) > EPSILON:
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

    @staticmethod
    def _update_trailer_heading(
            truck_heading: float,
            trailer_heading: float,
            velocity: float,
            hitch_length: float,
            dt: float
    ) -> float:
        """Update trailer heading using kinematics. Returns the new trailer heading."""
        theta_0 = truck_heading
        phi = trailer_heading
        dphi_dt = (velocity / hitch_length) * math.sin(theta_0 - phi)
        new_phi = Vehicle._wrap_angle(phi + dphi_dt * dt)
        return new_phi

    @staticmethod
    def _compute_trailer_position(trailer_heading: float, vehicle_spec: VehicleSpec, vehicle_pos: Pos) -> Pos:
        """
        Compute the trailer position given a trailer heading.
        Uses the hitch constraint: trailer is d1 meters behind hitch point.
        """

        hitch_x = vehicle_pos.x - (vehicle_spec.length / 2) * math.cos(vehicle_pos.heading)
        hitch_y = vehicle_pos.y - (vehicle_spec.length / 2) * math.sin(vehicle_pos.heading)

        hitch_length = vehicle_spec.trailer.hitch_length
        trailer_x = hitch_x - hitch_length * math.cos(trailer_heading)
        trailer_y = hitch_y - hitch_length * math.sin(trailer_heading)
        return Pos(trailer_x, trailer_y, trailer_heading)

    def _update_trailer(self, v: float, delta_time: float):
        """Update trailer heading and position based on truck motion"""
        if self.spec.trailer is None or self.trailer_heading is None or self.trailer_pos is None:
            return

        # Update trailer heading using kinematic equation
        new_heading = self._update_trailer_heading(
            self.pos.heading,
            self.trailer_heading,
            v,
            self.spec.trailer.hitch_length,
            delta_time
        )

        # Update trailer position
        self.trailer_pos = self._compute_trailer_position(new_heading, self.spec, self.pos)
        self.trailer_heading = new_heading

    def drive(self, delta_time: float, world: World):
        """Integrate vehicle motion based on the kinematic model"""
        # Apply acceleration limits to reach the desired velocity
        v_error = self._v_desired - self._v_actual
        max_delta_v = self.spec.max_acceleration * delta_time

        if abs(v_error) <= max_delta_v:
            # Can reach the desired velocity in this time step
            self._v_actual = self._v_desired
        else:
            # Accelerate/decelerate at max rate
            self._v_actual += math.copysign(max_delta_v, v_error)

        v = self._v_actual
        th = self.pos.heading

        # Determine angular velocity based on the kinematic model
        if self.spec.kinematic_model == KinematicModel.ACKERMANN:
            # For Ackermann: compute w from steering angle
            # w = v * tan(delta) / L
            delta = self._w_or_delta
            if abs(delta) < EPSILON:
                w = 0.0
            else:
                w = v * math.tan(delta) / self.spec.wheelbase
        else:
            # For diff-drive: w is directly commanded
            w = self._w_or_delta

        # Integrate motion using a unicycle model; applies regardless of kinematic model
        if abs(w) < EPSILON:  # Straight-line motion
            new_x = self.pos.x + v * math.cos(th) * delta_time
            new_y = self.pos.y + v * math.sin(th) * delta_time
            self.pos = Pos(new_x, new_y, th)
        else:  # Circular-arc motion
            th_new = th + w * delta_time
            new_x = self.pos.x + (v / w) * (math.sin(th_new) - math.sin(th))
            new_y = self.pos.y + (v / w) * (-math.cos(th_new) + math.cos(th))
            th_new = self._wrap_angle(th_new)
            self.pos = Pos(new_x, new_y, th_new)

        # Update trailer kinematics if equipped
        if self.spec.trailer is not None and self.trailer_heading is not None:
            self._update_trailer(v, delta_time)

        self._update_breadcrumbs(world)

    def _update_breadcrumbs(self, world: World):
        """Update breadcrumbs trail"""
        ppm = world.pixels_per_meter
        px, py = int(self.pos.x * ppm), int(self.pos.y * ppm)
        self.breadcrumbs.append(((px, py), self._v_actual))

        # Add trailer breadcrumbs if equipped
        if self.trailer_pos is not None:
            trailer_px = int(self.trailer_pos.x * ppm)
            trailer_py = int(self.trailer_pos.y * ppm)
            self.trailer_breadcrumbs.append(((trailer_px, trailer_py), self._v_actual))

    def render(self, world: World, pos: Optional[Pos] = None):
        """Render vehicle at current or specified position"""
        if self.spec.trailer is not None and pos is None:
            if self.trailer_pos is not None:
                self._render_trailer(world, self.trailer_pos)

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
        # Indicate front
        stripe_w = 4  # px
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(surf, self.spec.front_stripe_color, (stripe_x, 4, stripe_w, Wpx - 8), border_radius=2)

        # Rotate to the current heading
        rotated = pygame.transform.rotate(surf, -math.degrees(render_pos.heading))

        px = int(render_pos.x * ppm)
        py = int(render_pos.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        world.screen.blit(rotated, rect.topleft)

    def _render_trailer(self, world: World, trailer_pos: Pos):
        """Render the trailer at the specified position"""
        if self.spec.trailer is None:
            return

        ppm = world.pixels_per_meter
        Lpx = int(self.spec.trailer.length * ppm)
        Wpx = int(self.spec.trailer.width * ppm)

        surf = pygame.Surface((Lpx, Wpx), pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            self.spec.trailer.color,
            pygame.Rect(0, 0, Lpx, Wpx),
            border_radius=max(2, Wpx // 4),
        )

        # Rotate to the trailer heading
        rotated = pygame.transform.rotate(surf, -math.degrees(trailer_pos.heading))

        px = int(trailer_pos.x * ppm)
        py = int(trailer_pos.y * ppm)
        rect = rotated.get_rect(center=(px, py))
        world.screen.blit(rotated, rect.topleft)

        # Draw hitch connection line
        self._render_hitch(world, trailer_pos)

    def _render_hitch(self, world: World, trailer_pos: Pos):
        """Draw a line showing the hitch connection between truck and trailer"""
        ppm = world.pixels_per_meter

        # Hitch point at the rear of the truck body
        truck_th = self.pos.heading
        hitch_x = self.pos.x - (self.spec.length / 2) * math.cos(truck_th)
        hitch_y = self.pos.y - (self.spec.length / 2) * math.sin(truck_th)

        # Convert to pixels
        hitch_px = int(hitch_x * ppm)
        hitch_py = int(hitch_y * ppm)
        trailer_px = int(trailer_pos.x * ppm)
        trailer_py = int(trailer_pos.y * ppm)

        # Draw hitch line
        pygame.draw.line(world.screen, HITCH_COLOR, (hitch_px, hitch_py), (trailer_px, trailer_py), 2)

    def render_breadcrumbs(self, world: World):
        """Render breadcrumbs for truck and trailer"""
        for (px, py), v in self.breadcrumbs:
            pygame.draw.circle(world.screen, BREADCRUMB_COLOR, (px, py), max(1, int(v)))

        for (px, py), v in self.trailer_breadcrumbs:
            pygame.draw.circle(world.screen, TRAILER_BREADCRUMB_COLOR, (px, py), max(1, int(v)))

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

    @staticmethod
    def integrate_trailer_along_path(
            truck_path: List[Pos],
            initial_trailer_heading: float,
            vehicle_spec: VehicleSpec,
    ) -> List[Pos]:
        """
        Integrate trailer kinematics along a truck path.
        Returns the trailer positions corresponding to each truck position.
        """
        if vehicle_spec.trailer is None:
            return []

        trailer_path = []
        trailer_heading = initial_trailer_heading
        d1 = vehicle_spec.trailer.hitch_length

        for i, truck_pos in enumerate(truck_path):
            # Compute hitch and trailer positions using helper methods
            trailer_pos = Vehicle._compute_trailer_position(trailer_heading, vehicle_spec, truck_pos)
            trailer_path.append(trailer_pos)

            # Update trailer heading for next step
            if i < len(truck_path) - 1:
                next_truck = truck_path[i + 1]
                dx = next_truck.x - truck_pos.x
                dy = next_truck.y - truck_pos.y
                dist = math.hypot(dx, dy)

                # Skip update if points are too close (avoid numerical issues)
                if dist < EPSILON:
                    continue

                # Estimate time step and velocity
                dt = dist / vehicle_spec.cruising_velocity
                if dt < EPSILON:
                    continue
                v = dist / dt

                # Update trailer heading using helper method
                trailer_heading = Vehicle._update_trailer_heading(
                    truck_pos.heading,
                    trailer_heading,
                    v,
                    d1,
                    dt
                )

        return trailer_path

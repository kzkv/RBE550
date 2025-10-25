# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Pure Pursuit path follower. Heavy refactoring of a ChatGPT-based proof of concept.

"""
Pure Pursuit allows following a polyline path with smooth acceleration and heading control.
The heading at the end of the segment is adjusted to the `lookahead` distance.
"""

import math
from bisect import bisect_left
from typing import List, Tuple

from vehicle import Vehicle, KinematicModel
from world import Pos

POS_ERROR_THRESHOLD = 0.1

EPSILON = 1e-8  # Numerical tolerance for floating-point comparisons


class PathFollower:
    def __init__(
            self,
            route: List[Pos],
            lookahead: float,
            vehicle: Vehicle,
    ):
        self.vehicle = vehicle
        self.route = route
        self.lookahead = lookahead
        self.wheel_speed_max = self.vehicle.spec.cruising_velocity * 1.5

        # Compute braking distance
        self.braking_distance = (self.vehicle.spec.cruising_velocity ** 2) / (2 * self.vehicle.spec.max_acceleration)

        # Compute acceleration distance
        self.acceleration_distance = self.braking_distance

        # Precompute cumulative arc lengths
        self.cumulative_arc_lengths = [0.0]
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            self.cumulative_arc_lengths.append(
                self.cumulative_arc_lengths[-1] + prev.distance_to(curr)
            )
        self.total_arc_length = self.cumulative_arc_lengths[-1]

        self.traveled = 0.0
        self.we_are_there = False
        self.destination = route[-1]

    def _interpolate_at(self, arclength: float) -> Tuple[float, float]:
        """Interpolate a point along the polyline at arclength"""
        arclength = min(self.total_arc_length, arclength)

        i = min(
            bisect_left(self.cumulative_arc_lengths, arclength),
            len(self.cumulative_arc_lengths) - 1
        )

        segment_prev, segment = self.cumulative_arc_lengths[i - 1], self.cumulative_arc_lengths[i]
        p0 = self.route[i - 1]
        p1 = self.route[i]

        if segment - segment_prev == 0:
            return p0.x, p0.y

        segment_fraction = (arclength - segment_prev) / (segment - segment_prev)
        return (p0.x + segment_fraction * (p1.x - p0.x),
                p0.y + segment_fraction * (p1.y - p0.y))

    def update(self):
        """Compute (v, w) and command the vehicle"""

        # Update traveled distance based on actual vehicle position along path
        min_dist = None
        closest_s = self.traveled

        # Search around current traveled position for closest point
        search_start = max(0.0, self.traveled - self.lookahead)
        search_end = min(self.total_arc_length, self.traveled + self.lookahead)

        # Sample points along the path
        samples = 20
        for i in range(samples + 1):
            s = search_start + (search_end - search_start) * i / samples
            px, py = self._interpolate_at(s)
            dist = math.hypot(px - self.vehicle.pos.x, py - self.vehicle.pos.y)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest_s = s

        self.traveled = closest_s

        # Lookahead target point (clamped)
        target_s = min(self.traveled + self.lookahead, self.total_arc_length)
        tx, ty = self._interpolate_at(target_s)

        # Distance to lookahead
        dx, dy = tx - self.vehicle.pos.x, ty - self.vehicle.pos.y

        # Transform into vehicle frame
        cos_t, sin_t = math.cos(self.vehicle.pos.heading), math.sin(self.vehicle.pos.heading)
        x_frame = cos_t * dx + sin_t * dy
        y_frame = -sin_t * dx + cos_t * dy

        # Pure Pursuit steering law (κ = 2*y_frame / d^2; thanks, ChatGPT)
        d_squared = x_frame * x_frame + y_frame * y_frame
        if d_squared == 0:
            kappa = 0  # Avoid division by zero.
        else:
            kappa = 2 * y_frame / d_squared  # Turn sharpness

        # Compute kinematic-specific limits
        if self.vehicle.spec.kinematic_model == KinematicModel.ACKERMANN:
            kappa_max = math.tan(self.vehicle.spec.max_steering_angle) / self.vehicle.spec.wheelbase
            kappa = max(-kappa_max, min(kappa_max, kappa))
            w_max_at_cruise = self.vehicle.spec.cruising_velocity * kappa_max
        else:
            # For diff-drive: use the specified w_max
            w_max_at_cruise = self.vehicle.spec.w_max

        # Clamp velocity: keep cruise unless curvature forces a limit
        # TODO: consider clamping centripetal acceleration too? IRL this can cause rolling over on uneven pavement
        v_limit_turn = w_max_at_cruise / max(EPSILON, abs(kappa))
        v = min(self.vehicle.spec.cruising_velocity, v_limit_turn)

        w = v * kappa

        # Apply kinematic-specific constraints
        if self.vehicle.spec.kinematic_model == KinematicModel.DIFF_DRIVE:
            # Clamp rotation based on the individual wheel speed for diff-drive
            # |w| <= 2/track_width * (wheel_speed_max - |v|); thanks again, ChatGPT
            w_cap = (2.0 / self.vehicle.spec.track_width) * max(0.0, self.wheel_speed_max - abs(v))
            if abs(w) > w_cap:
                w = math.copysign(w_cap, w)  # Keep the sign, but clamp
        else:
            # For Ackermann: constrain angular velocity based on current speed and max steering
            if abs(v) > EPSILON:
                w_max_current = abs(v) * math.tan(self.vehicle.spec.max_steering_angle) / self.vehicle.spec.wheelbase
                w = max(-w_max_current, min(w_max_current, w))

        # Acceleration ramp at the beginning of the path
        if self.traveled <= self.acceleration_distance:
            # Ensure minimum velocity to get started
            accel_scale = math.sqrt(self.traveled / self.acceleration_distance) if self.traveled > 0 else 0.1
            v *= max(accel_scale, 0.1)  # At least 10% speed to get started

        # Braking as we approach the end of the path
        distance_to_path_end = self.total_arc_length - self.traveled
        if distance_to_path_end <= self.braking_distance:
            brake_scale = math.sqrt(distance_to_path_end / self.braking_distance)
            v *= max(brake_scale, 0.05)  # Minimum 5% speed while braking

        # Stop when very close to the end of the trajectory
        if distance_to_path_end <= POS_ERROR_THRESHOLD:
            self.we_are_there = True
            v, w = 0.0, 0.0

        # TODO: the follower never commands going in reverse, but it could be useful
        self.vehicle.set_velocities(v, w)

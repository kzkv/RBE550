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

from vehicle import Vehicle
from world import Pos

BRAKING_DISTANCE = 1.5  # TODO: derive from cruising velocity?
POS_ERROR_THRESHOLD = 0.1


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
        self._v_last = 0.0
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

    def update(self, delta_time: float):
        """Compute (v, w) and command the vehicle"""
        # TODO: take into account track width

        # Advance nominal path progress; clamp to total arc length
        self.traveled = min(self.traveled + max(0.0, self._v_last) * delta_time, self.total_arc_length)

        # Lookahead target point (clamped)
        target_s = min(self.traveled + self.lookahead, self.total_arc_length)
        tx, ty = self._interpolate_at(target_s)

        # Distance to lookahead
        dx, dy = tx - self.vehicle.pos.x, ty - self.vehicle.pos.y

        # Transform into vehicle frame
        cos_t, sin_t = math.cos(self.vehicle.pos.heading), math.sin(self.vehicle.pos.heading)
        x_frame = cos_t * dx + sin_t * dy
        y_frame = -sin_t * dx + cos_t * dy
        # TODO: consider if the target is behind the vehicle

        # Pure Pursuit steering law (κ = 2*y_frame / d^2; thanks, ChatGPT)
        d_squared = x_frame * x_frame + y_frame * y_frame
        if d_squared == 0:
            kappa = 0  # avoid div by zero  TODO: why does this happen? probably zero-length segments
        else:
            kappa = 2 * y_frame / d_squared  # turn sharpness

        # Clamp velocity: keep cruise unless curvature forces a limit
        # TODO: consider clamping centripetal acceleration too? IRL this can cause rolling over on uneven pavement
        v_limit_turn = self.vehicle.spec.w_max / max(1e-6, abs(kappa))
        v = min(self.vehicle.spec.cruising_velocity, v_limit_turn)

        w = v * kappa

        # clamp rotation based on the individual wheel speed
        # |w| <= 2/track_width * (wheel_speed_max - |v|); thanks again, ChatGPT
        w_cap = (2.0 / self.vehicle.spec.track_width) * max(0.0, self.wheel_speed_max - abs(v))
        if abs(w) > w_cap:
            w = math.copysign(w_cap, w)  # keep the sign, but clamp
        # TODO: consider checking each wheels individually to avoid any corner cases

        # Are we there yet?
        left_to_go = self.vehicle.pos.distance_to(self.destination)
        if left_to_go <= BRAKING_DISTANCE:
            scale = max(0.0, left_to_go / BRAKING_DISTANCE)
            v *= scale

        if left_to_go <= POS_ERROR_THRESHOLD:
            self.we_are_there = True
            v, w = 0.0, 0.0

        # TODO: the follower never commands going in reverse, but it could be useful
        self.vehicle.set_velocities(v, w)
        self._v_last = v

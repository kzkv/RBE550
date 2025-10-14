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

BRAKING_DISTANCE = 1.5  # TODO: derive from cruising velocity?
POS_ERROR_THRESHOLD = 0.1


class PathFollower:
    def __init__(
            self,
            route: List[Tuple[float, float]],
            lookahead: float,
            v_cruise: float,
            w_max: float,
    ):
        # TODO: this relies on at least two waypoints. Write some units and isolate
        self.route = route
        self.lookahead = lookahead
        self.v_cruise = v_cruise
        self.w_max = w_max  # rad/s cap

        # Precompute cumulative arc lengths (segments) for easy interpolation
        self.cumulative_arc_lengths = [0.0]
        for i in range(1, len(route)):
            self.cumulative_arc_lengths.append(
                self.cumulative_arc_lengths[-1] + self.distance_between(route[i - 1], route[i])
            )
        self.total_arc_length = self.cumulative_arc_lengths[-1]
        self.traveled = 0.0  # Distance traveled along the concatenated route (m)
        self.we_are_there = False

        # Keep the last v to integrate progress smoothly
        self._v_last = 0.0

        self.destination = (self.route[-1][0], self.route[-1][1])
        # TODO: consider specifying heading for the last pos

    @staticmethod
    def distance_between(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
        ox, oy = origin
        dx, dy = destination
        return math.hypot(ox - dx, oy - dy)

    def _interpolate_at(self, arclength: float) -> Tuple[float, float]:
        """Interpolate a point along the polyline at arclength."""

        arclength = min(self.total_arc_length, arclength)  # if lookahead is beyond route, clamp to total length

        # Find the first segment index that cumulative_arc_lengths[i] >= arclength
        i = min(
            bisect_left(self.cumulative_arc_lengths, arclength),
            len(self.cumulative_arc_lengths) - 1
        )

        segment_prev, segment = self.cumulative_arc_lengths[i - 1], self.cumulative_arc_lengths[i]
        x0, y0 = self.route[i - 1]
        x1, y1 = self.route[i]

        # How far along the segment we are?
        segment_fraction = (arclength - segment_prev) / (segment - segment_prev)  # TODO: consider overlapping waypoints
        return x0 + segment_fraction * (x1 - x0), y0 + segment_fraction * (y1 - y0)

    def update(self, vehicle: Vehicle, delta_time: float):
        """Compute (v, w) and command the vehicle"""
        # TODO: take into account track width

        # Advance nominal path progress; clamp to total arc length
        self.traveled = min(self.traveled + max(0.0, self._v_last) * delta_time, self.total_arc_length)

        # Lookahead target point (clamped)
        target_s = min(self.traveled + self.lookahead, self.total_arc_length)
        tx, ty = self._interpolate_at(target_s)

        # Distance to lookahead
        dx, dy = tx - vehicle.x, ty - vehicle.y

        # Transform into vehicle frame
        cos_t, sin_t = math.cos(vehicle.heading), math.sin(vehicle.heading)
        x_frame = cos_t * dx + sin_t * dy
        y_frame = -sin_t * dx + cos_t * dy

        # Pure Pursuit steering law (κ = 2*y_frame / d^2; thanks, ChatGPT)
        d_squared = x_frame * x_frame + y_frame * y_frame
        kappa = 2 * y_frame / d_squared  # turn sharpness
        # TODO: consider the target being very close to the vehicle

        # Clamp velocity: keep cruise unless curvature forces a limit
        # TODO: consider clamping centripetal acceleration too?
        v_limit_turn = self.w_max / max(1e-6, abs(kappa))
        v = min(self.v_cruise, v_limit_turn)

        w = v * kappa

        # Are we there yet?
        left_to_go = self.distance_between((vehicle.x, vehicle.y), self.destination)
        if left_to_go <= BRAKING_DISTANCE:
            scale = max(0.0, left_to_go / BRAKING_DISTANCE)
            v *= scale

        if left_to_go <= POS_ERROR_THRESHOLD:
            self.we_are_there = True
            v, w = 0.0, 0.0

        vehicle.set_velocities(v, w)
        self._v_last = v

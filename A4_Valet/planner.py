# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: Claude for path planning algo; the generated code snippets refactored beyond recognition

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from itertools import product

from world import Pos, world_to_grid, GRID_DIMENSIONS

PLANNED_POS_ERROR_THRESHOLD = 0.25  # m
PLANNED_HEADING_ERROR_THRESHOLD = math.radians(5)  # rad (deg)


@dataclass
class MotionPrimitive:
    """A motion primitive (path segment)"""
    arc_length: float
    curvature: float
    num_steps: int = 20

    def apply(self, state: Pos) -> Tuple[Pos, List[Tuple[float, float]]]:
        """Apply this primitive starting from the given state."""
        path_points = []

        if abs(self.curvature) < 1e-6:  # Straight line
            for i in range(self.num_steps + 1):
                s = (i / self.num_steps) * self.arc_length
                x = state.x + s * math.cos(state.heading)
                y = state.y + s * math.sin(state.heading)
                path_points.append((x, y))

            end_state = Pos(
                x=state.x + self.arc_length * math.cos(state.heading),
                y=state.y + self.arc_length * math.sin(state.heading),
                heading=state.heading
            )

        else:  # Circular arc
            radius = 1.0 / self.curvature
            d_theta = self.curvature * self.arc_length

            cx = state.x - radius * math.sin(state.heading)
            cy = state.y + radius * math.cos(state.heading)

            for i in range(self.num_steps + 1):
                theta_i = state.heading + (i / self.num_steps) * d_theta
                x = cx + radius * math.sin(theta_i)
                y = cy - radius * math.cos(theta_i)
                path_points.append((x, y))

            end_heading = state.heading + d_theta
            end_heading = (end_heading + math.pi) % (2 * math.pi) - math.pi

            end_state = Pos(
                x=path_points[-1][0],
                y=path_points[-1][1],
                heading=end_heading
            )

        return end_state, path_points


@dataclass
class StateKey:
    """Discretized state for hashing"""
    x: float
    y: float
    theta: float

    @staticmethod
    def from_pos(pos: Pos, xy_resolution: float = 0.75, theta_resolution: float = 0.4) -> 'StateKey':
        """Create discretized state key"""
        return StateKey(
            x=round(pos.x / xy_resolution) * xy_resolution,
            y=round(pos.y / xy_resolution) * xy_resolution,
            theta=round(pos.heading / theta_resolution) * theta_resolution
        )

    def __hash__(self):
        return hash((self.x, self.y, self.theta))

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.theta == other.theta)


@dataclass(order=True)
class SearchNode:
    """Node in A* search"""
    f_score: float
    state: Pos = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['SearchNode'] = field(default=None, compare=False)
    path_segment: List[Tuple[float, float]] = field(default_factory=list, compare=False)


def create_motion_primitives() -> List[MotionPrimitive]:
    """Create motion primitives (forward only) - all combinations"""
    arc_lengths = [1.5, 3.0, 6.0]
    curvatures = [0.0, 0.15, -0.15, 0.3, -0.3, 0.5, -0.5, 0.8, -0.8, 1.0, -1.0]

    return [
        MotionPrimitive(arc_length=length, curvature=curvature)
        for length, curvature in product(arc_lengths, curvatures)
    ]


def is_collision_free(path_points: List[Tuple[float, float]],
                      obstacles: np.ndarray,
                      vehicle_width: float = 0.57) -> bool:
    """Check if path collides with obstacles, considering vehicle footprint."""
    world_size = GRID_DIMENSIONS * 3.0
    safety_radius = vehicle_width / 2.0 + 0.2

    for x, y in path_points:
        if (x < safety_radius or y < safety_radius or
                x >= world_size - safety_radius or y >= world_size - safety_radius):
            return False

        check_points = [
            (x, y),
            (x + safety_radius, y),
            (x - safety_radius, y),
            (x, y + safety_radius),
            (x, y - safety_radius),
        ]

        for cx, cy in check_points:
            row, col = world_to_grid(cx, cy)
            if row < 0 or row >= GRID_DIMENSIONS or col < 0 or col >= GRID_DIMENSIONS:
                return False
            if obstacles[row, col]:
                return False

    return True


def heuristic(state: Pos, goal: Pos) -> float:
    """Heuristic with strong heading awareness near goal."""
    dist = math.hypot(state.x - goal.x, state.y - goal.y)

    heading_error = abs(state.heading - goal.heading)
    if heading_error > math.pi:
        heading_error = 2 * math.pi - heading_error

    # Near goal: heading is critical
    if dist < 5.0:
        heading_penalty = heading_error * 8.0  # Strong penalty
    else:
        heading_penalty = heading_error * 2.0

    return dist + heading_penalty


def reconstruct_path(node: SearchNode) -> List[Tuple[float, float]]:
    """Reconstruct path by collecting all segments from goal back to start"""
    segments = []
    current = node

    # Walk back to start, collecting segments
    while current is not None:
        if current.path_segment and len(current.path_segment) > 0:
            segments.append(current.path_segment)
        current = current.parent

    # Reverse to get start→goal order
    segments.reverse()

    if not segments:
        return []

    # Flatten: take first segment entirely, then skip first point of subsequent segments
    path = list(segments[0])  # Start with entire first segment

    for segment in segments[1:]:
        for point in segment:
            # Skip if too close to last point (avoid duplicates)
            if path and math.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) < 0.05:
                continue
            path.append(point)

    return path


def plan(origin: Pos, goal: Pos, obstacles: np.ndarray,
         vehicle_width: float = 0.57) -> Optional[List[Tuple[float, float]]]:
    """A* path planning with vehicle footprint and goal heading awareness."""

    print(f"\nA* Path Planning")
    print(f"  Origin: ({origin.x:.2f}, {origin.y:.2f}, {math.degrees(origin.heading):.0f}°)")
    print(f"  Goal:  ({goal.x:.2f}, {goal.y:.2f}, {math.degrees(goal.heading):.0f}°)")

    primitives = create_motion_primitives()
    print(f"  Using {len(primitives)} motion primitives")

    open_set = []
    visited: Set[StateKey] = set()

    start_node = SearchNode(
        f_score=heuristic(origin, goal),
        state=origin,
        g_score=0.0,
        path_segment=[(origin.x, origin.y)]
    )

    heapq.heappush(open_set, start_node)

    nodes_expanded = 0
    max_iterations = 30000

    while open_set and nodes_expanded < max_iterations:
        current = heapq.heappop(open_set)

        current_key = StateKey.from_pos(current.state)
        if current_key in visited:
            continue

        visited.add(current_key)
        nodes_expanded += 1

        # Goal check - position AND heading
        dist_to_goal = math.hypot(current.state.x - goal.x, current.state.y - goal.y)
        heading_error = abs(current.state.heading - goal.heading)
        if heading_error > math.pi:
            heading_error = 2 * math.pi - heading_error

        # Success criteria
        position_tolerance = PLANNED_POS_ERROR_THRESHOLD
        heading_tolerance = PLANNED_HEADING_ERROR_THRESHOLD

        if dist_to_goal < position_tolerance and heading_error < heading_tolerance:
            print(f"\nPATH FOUND!")  # TODO: it would be very cool to display all of the paths tried by A*
            print(f"  Nodes expanded: {nodes_expanded}")
            print(f"  Path cost: {current.g_score:.1f}m")
            print(f"  Final position error: {dist_to_goal:.2f}m")
            print(f"  Final heading error: {math.degrees(heading_error):.1f}°")

            path = reconstruct_path(current)

            print(f"\nReconstructed {len(path)} points")
            if path:
                print(
                    f"  Path starts at: ({path[0][0]:.2f}, {path[0][1]:.2f}) - Expected: ({origin.x:.2f}, {origin.y:.2f})")
                print(
                    f"  Path ends at: ({path[-1][0]:.2f}, {path[-1][1]:.2f}) - Expected: ({goal.x:.2f}, {goal.y:.2f})")

                # Verify start
                start_error = math.hypot(path[0][0] - origin.x, path[0][1] - origin.y)
                if start_error > 0.1:
                    print(f"  WARNING: Path doesn't start at origin! Error: {start_error:.2f}m")

            # Ensure path ends at exact goal
            if path and math.hypot(path[-1][0] - goal.x, path[-1][1] - goal.y) > 0.1:
                path.append((goal.x, goal.y))

            print(f"  Path starts at: ({path[0][0]:.1f}, {path[0][1]:.1f})")
            print(f"  Path ends at: ({path[-1][0]:.1f}, {path[-1][1]:.1f})")
            print(f"  Total waypoints: {len(path)}")
            return path

        # Expand neighbors
        for primitive in primitives:
            new_state, path_segment = primitive.apply(current.state)

            new_key = StateKey.from_pos(new_state)
            if new_key in visited:
                continue

            if not is_collision_free(path_segment, obstacles, vehicle_width):
                continue

            g_score = current.g_score + abs(primitive.arc_length)
            f_score = g_score + heuristic(new_state, goal)

            neighbor = SearchNode(
                f_score=f_score,
                state=new_state,
                g_score=g_score,
                parent=current,
                path_segment=path_segment
            )

            heapq.heappush(open_set, neighbor)

        # Progress indicator
        if nodes_expanded % 1000 == 0:
            print(f"  {nodes_expanded} nodes, position err: {dist_to_goal:.1f}m, heading err: {math.degrees(heading_error):.0f}°")

    print(f"NO PATH FOUND AFTER EXPANDING {nodes_expanded} NODES")
    return None

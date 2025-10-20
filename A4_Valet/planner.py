# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: Claude for path planning algo; the generated code snippets refactored beyond recognition

"""
Lattice-based planner contructs the path out of precomputed motion primitives.
Takes into account target heading. Applies safety margins (radius of vehicle footprint).

Key design decisions:
1. Only driving forward.
2. Fixed (pre-generated) primitives set.
3. Collision-checking against safety margin (radius of vehicle footprint).
4. Collision-checking using cross-pattern for execution speed.
5. Aggressive weighting of the heading near the goal.
6. Limited number of explored nodes to avoid infinite loops.
7. Progress reporting for debugging.
8. Reconstructing a path from goal back to start.
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from itertools import product

from world import CELL_SIZE
from world import Pos, world_to_grid, GRID_DIMENSIONS

# Planning constants
PLANNED_POS_ERROR_THRESHOLD = 0.25  # m
PLANNED_HEADING_ERROR_THRESHOLD = math.radians(5)  # rad (deg)

# Motion primitive parameters
ARC_LENGTHS = [1.5, 3.0, 6.0]
CURVATURES = [0.0, 0.15, -0.15, 0.3, -0.3, 0.5, -0.5, 0.8, -0.8, 1.0, -1.0]
PRIMITIVE_STEPS = 10

# Discretization resolution; these are magic parameters that had to be tuned to achieve good performance.
# Too coarse or too fine is failing the path planning.
XY_RESOLUTION = CELL_SIZE / 4.0
THETA_RESOLUTION = 0.4  # ~23 deg

# Safety margins
VEHICLE_SAFETY_MARGIN = 0.2  # m

# A* search limits
MAX_ITERATIONS = 30000
PROGRESS_INTERVAL = 1000


@dataclass
class MotionPrimitive:
    """Motion primitive representing a curved or straight path segment"""
    arc_length: float
    curvature: float
    num_steps: int = PRIMITIVE_STEPS  # number of discrete steps along the arc

    def apply(self, state: Pos) -> Tuple[Pos, List[Tuple[float, float]]]:
        """Generate path points and end state for this primitive"""
        path_points = []

        if abs(self.curvature) < 1e-6:  # straight line, simple trig to move in the direction of the current heading
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

        else:  # circular arc, computes curvature center (pivot point)
            radius = 1.0 / self.curvature
            d_theta = self.curvature * self.arc_length

            # curvature center
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


def create_motion_primitives() -> List[MotionPrimitive]:
    """Generate all motion primitive combinations"""
    return [
        MotionPrimitive(arc_length=length, curvature=curvature)
        for length, curvature in product(ARC_LENGTHS, CURVATURES)
    ]


@dataclass
class StateKey:
    """
    Discretized state for hashing in a visited set. While in shape it's similar to Pos, it has different semantics:
    unlike Pos (a precise position in the world), Statekey is a "bucket" of similar poses.
    """
    x: float
    y: float
    theta: float

    @staticmethod
    def from_pos(pos: Pos) -> 'StateKey':
        """Create a discretized state key"""
        return StateKey(
            x=round(pos.x / XY_RESOLUTION) * XY_RESOLUTION,
            y=round(pos.y / XY_RESOLUTION) * XY_RESOLUTION,
            theta=round(pos.heading / THETA_RESOLUTION) * THETA_RESOLUTION
        )

    def __hash__(self):
        return hash((self.x, self.y, self.theta))

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.theta == other.theta)


@dataclass(order=True)
class SearchNode:
    """Node in A* search"""
    f_score: float
    state: Pos = field(compare=False)  # robot position at this node
    g_score: float = field(compare=False)  # actual distance from start to this node
    parent: Optional['SearchNode'] = field(default=None, compare=False)  # pointer for path reconstruction
    path_segment: List[Tuple[float, float]] = field(default_factory=list, compare=False)


def is_collision_free(
        path_points: List[Tuple[float, float]],
        obstacles: np.ndarray,
        vehicle_width: float
) -> bool:
    """Check if path collides with obstacles considering vehicle footprint"""
    world_size = GRID_DIMENSIONS * 3.0
    safety_radius = vehicle_width / 2.0 + VEHICLE_SAFETY_MARGIN

    for x, y in path_points:
        # do we fit inside the world?
        if (x < safety_radius or y < safety_radius or
                x >= world_size - safety_radius or y >= world_size - safety_radius):
            return False

        # do we collide with obstacles?
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

    heading_penalty = heading_error * 8.0 if dist < 5.0 else heading_error * 2.0

    return dist + heading_penalty


def reconstruct_path(node: SearchNode) -> List[Tuple[float, float]]:
    """Reconstruct path from goal back to start"""
    segments = []
    current = node

    while current is not None:
        if current.path_segment and len(current.path_segment) > 0:
            segments.append(current.path_segment)
        current = current.parent

    segments.reverse()

    if not segments:
        return []

    path = list(segments[0])

    for segment in segments[1:]:
        for point in segment:
            if path and math.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) < 0.1:
                continue
            path.append(point)

    return path


def plan(
        origin: Pos,
        goal: Pos,
        obstacles: np.ndarray,
        vehicle_width: float
) -> Optional[List[Tuple[float, float]]]:
    """A* path planning with vehicle footprint and goal heading awareness"""

    print(f"\nA* Path Planning")
    print(f"  Origin: ({origin.x:.2f}, {origin.y:.2f}, {math.degrees(origin.heading):.0f}°)")
    print(f"  Goal:   ({goal.x:.2f}, {goal.y:.2f}, {math.degrees(goal.heading):.0f}°)")

    primitives = create_motion_primitives()
    print(f"  Motion primitives: {len(primitives)}")

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

    while open_set and nodes_expanded < MAX_ITERATIONS:
        current = heapq.heappop(open_set)

        current_key = StateKey.from_pos(current.state)
        if current_key in visited:
            continue

        visited.add(current_key)
        nodes_expanded += 1

        dist_to_goal = math.hypot(current.state.x - goal.x, current.state.y - goal.y)
        heading_error = abs(current.state.heading - goal.heading)
        if heading_error > math.pi:
            heading_error = 2 * math.pi - heading_error

        if dist_to_goal < PLANNED_POS_ERROR_THRESHOLD and heading_error < PLANNED_HEADING_ERROR_THRESHOLD:
            # TODO: it would be cool to visualize all tried paths in A* for debugging
            print(f"\nPATH FOUND")
            print(f"  Nodes expanded: {nodes_expanded}")
            print(f"  Path cost: {current.g_score:.1f}m")
            print(f"  Final error: {dist_to_goal:.2f}m, {math.degrees(heading_error):.1f}°")

            path = reconstruct_path(current)

            print(f"  Reconstructed {len(path)} waypoints")
            print(f"  Path: ({path[0][0]:.1f}, {path[0][1]:.1f}) → ({path[-1][0]:.1f}, {path[-1][1]:.1f})")

            if path and math.hypot(path[-1][0] - goal.x, path[-1][1] - goal.y) > 0.1:
                path.append((goal.x, goal.y))

            return path

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

        if nodes_expanded % PROGRESS_INTERVAL == 0:
            print(f"  {nodes_expanded} nodes, err: {dist_to_goal:.1f}m / {math.degrees(heading_error):.0f}°")

    print(f"NO PATH FOUND after {nodes_expanded} nodes")
    return None

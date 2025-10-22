"""
Lattice-based planner constructs the path out of precomputed motion primitives.
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
import time
from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from itertools import product

from world import CELL_SIZE
from world import Pos, world_to_grid, GRID_DIMENSIONS
from vehicle import VehicleSpec

pi = math.pi

# Motion primitive parameters  # TODO: it would be great to smooth the trajectory
ARC_LENGTHS = [0.3, 0.75, 1.5, 3.0, 6.0]
CURVATURES = [0.0, pi / 24, -pi / 24, pi / 12, -pi / 12, pi / 6, -pi / 6, pi / 4, -pi / 4, pi / 2, -pi / 2, pi, -pi]
PRIMITIVE_STEPS = 10

# Prefer longer arcs; TODO: don't forget to highlight in the report how essential this proved to be
ARC_LENGTH_BIAS_WEIGHT = 2.0
MAX_ARC_LENGTH = max(ARC_LENGTHS)

# Discretization resolution; these are magic parameters that had to be tuned to achieve good performance.
# Too coarse or too fine is failing the path planning.
XY_RESOLUTION = 0.75
THETA_RESOLUTION = math.pi / 6  # ~30 deg

# Safety margins
VEHICLE_SAFETY_MARGIN = 0.1  # m

# A* search limits
MAX_ITERATIONS = 30000
PROGRESS_INTERVAL = 1000

PATH_EXTENSION = 0.25


def get_obb_corners(x: float, y: float, heading: float, length: float, width: float) -> List[Tuple[float, float]]:
    """Compute the four corners of an oriented bounding box"""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    half_length = length / 2
    half_width = width / 2

    # Corners in vehicle frame: front-right, front-left, back-left, back-right
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]

    # Transform to world frame
    world_corners = []
    for lx, ly in local_corners:
        wx = x + lx * cos_h - ly * sin_h
        wy = y + lx * sin_h + ly * cos_h
        world_corners.append((wx, wy))

    return world_corners


def check_obb_collision(corners: List[Tuple[float, float]], obstacles: np.ndarray) -> bool:
    """Check if any corner or edge of OBB collides with obstacles"""
    # Check all four corners
    for cx, cy in corners:
        row, col = world_to_grid(cx, cy)
        if row < 0 or row >= GRID_DIMENSIONS or col < 0 or col >= GRID_DIMENSIONS:
            return True  # Out of bounds
        if obstacles[row, col]:
            return True  # Corner in an obstacle cell

    # Check edges by sampling points between corners
    num_samples = 3  # Sample points per edge
    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]

        for j in range(1, num_samples):
            t = j / num_samples
            ex = c1[0] + t * (c2[0] - c1[0])
            ey = c1[1] + t * (c2[1] - c1[1])

            row, col = world_to_grid(ex, ey)
            if row < 0 or row >= GRID_DIMENSIONS or col < 0 or col >= GRID_DIMENSIONS:
                return True
            if obstacles[row, col]:
                return True

    return False


@dataclass
class MotionPrimitive:
    """Motion primitive representing a curved or straight path segment"""
    arc_length: float
    curvature: float
    num_steps: int = PRIMITIVE_STEPS

    def apply(self, state: Pos) -> Tuple[Pos, List[Pos]]:
        """Generate path points as Pos objects and end state for this primitive"""
        path_points = []

        if abs(self.curvature) < 1e-6:  # straight line
            for i in range(self.num_steps + 1):
                s = (i / self.num_steps) * self.arc_length
                x = state.x + s * math.cos(state.heading)
                y = state.y + s * math.sin(state.heading)
                path_points.append(Pos(x, y, state.heading))

            end_state = Pos(
                x=state.x + self.arc_length * math.cos(state.heading),
                y=state.y + self.arc_length * math.sin(state.heading),
                heading=state.heading
            )

        else:  # circular arc
            radius = 1.0 / self.curvature
            d_theta = self.curvature * self.arc_length

            cx = state.x - radius * math.sin(state.heading)
            cy = state.y + radius * math.cos(state.heading)

            for i in range(self.num_steps + 1):
                theta_i = state.heading + (i / self.num_steps) * d_theta
                x = cx + radius * math.sin(theta_i)
                y = cy - radius * math.cos(theta_i)
                path_points.append(Pos(x, y, theta_i))

            end_heading = state.heading + d_theta
            end_heading = (end_heading + math.pi) % (2 * math.pi) - math.pi

            end_state = Pos(
                x=path_points[-1].x,
                y=path_points[-1].y,
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
    unlike Pos (a precise position in the world), StateKey is a "bucket" of similar poses.
    """
    x: float
    y: float
    theta: float

    @staticmethod
    def from_pos(pos: Pos) -> 'StateKey':
        """Create a discretized state key"""
        # Wrap heading to [-pi, pi] before discretization to ensure
        # equivalent headings map to the same bucket
        wrapped_heading = (pos.heading + math.pi) % (2 * math.pi) - math.pi
        
        return StateKey(
            x=round(pos.x / XY_RESOLUTION) * XY_RESOLUTION,
            y=round(pos.y / XY_RESOLUTION) * XY_RESOLUTION,
            theta=round(wrapped_heading / THETA_RESOLUTION) * THETA_RESOLUTION
        )

    def __hash__(self):
        return hash((self.x, self.y, self.theta))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta


@dataclass(order=True)
class SearchNode:
    """Node in A* search"""
    f_score: float
    state: Pos = field(compare=False)  # robot position at this node
    g_score: float = field(compare=False)  # actual distance from start to this node
    parent: Optional['SearchNode'] = field(default=None, compare=False)  # pointer for path reconstruction
    path_segment: List[Pos] = field(default_factory=list, compare=False)  # Store Pos objects


def is_collision_free(
        path_points: List[Pos],
        obstacles: np.ndarray,
        vehicle_spec: VehicleSpec,
) -> bool:
    """Two-stage collision checking: cross-pattern for speed, OBB for accuracy"""
    world_size = GRID_DIMENSIONS * CELL_SIZE
    safety_radius = vehicle_spec.width / 2 + VEHICLE_SAFETY_MARGIN

    # Stage 1: Fast cross-pattern rejection on all points
    for pos in path_points:
        # Boundary check
        if (pos.x < safety_radius or pos.y < safety_radius or
                pos.x >= world_size - safety_radius or pos.y >= world_size - safety_radius):
            return False

        # Quick cross-pattern check (center + 4 cardinal points)
        check_points = [
            (pos.x, pos.y),
            (pos.x + safety_radius, pos.y),
            (pos.x - safety_radius, pos.y),
            (pos.x, pos.y + safety_radius),
            (pos.x, pos.y - safety_radius),
        ]

        for cx, cy in check_points:
            row, col = world_to_grid(cx, cy)
            if row < 0 or row >= GRID_DIMENSIONS or col < 0 or col >= GRID_DIMENSIONS:
                return False
            if obstacles[row, col]:
                return False

    # Stage 2: Precise OBB validation - check all points
    for pos in path_points:
        corners = get_obb_corners(pos.x, pos.y, pos.heading, vehicle_spec.length, vehicle_spec.width)
        if check_obb_collision(corners, obstacles):
            return False

    return True


def heuristic(state: Pos, goal: Pos) -> float:
    """Heuristic with strong heading awareness near goal."""
    dist = math.hypot(state.x - goal.x, state.y - goal.y)

    heading_error = abs(state.heading - goal.heading)
    if heading_error > math.pi:
        heading_error = 2 * math.pi - heading_error

    heading_penalty = heading_error * 8 if dist < 5.0 else heading_error * 2

    return dist + heading_penalty


def reconstruct_path(node: SearchNode) -> List[Pos]:
    """Reconstruct the path from goal back to start"""
    segments = []
    current = node

    while current is not None:
        if current.path_segment and len(current.path_segment) > 0:
            segments.append(current.path_segment)
        current = current.parent

    segments.reverse()

    if not segments:
        return []

    path = []

    # Add all segments, wrapping headings to ensure consistency
    for segment in segments:
        for pos in segment:
            # Wrap the heading to [-pi, pi] before adding to path
            wrapped_heading = (pos.heading + math.pi) % (2 * math.pi) - math.pi
            path.append(Pos(pos.x, pos.y, wrapped_heading))

    # Replace the last waypoint with the actual goal node state
    # This ensures the heading matches what the planner verified against the goal
    if path and node.state:
        path[-1] = Pos(node.state.x, node.state.y, node.state.heading)
    
    # Add a small extension segment in the direction of final heading
    # This helps Pure Pursuit track the final heading without instability
    # TODO: explain the shenanigans happening here in the report
    if path:
        final_waypoint = path[-1]
        extension_length = PATH_EXTENSION
        extended_x = final_waypoint.x + extension_length * math.cos(final_waypoint.heading)
        extended_y = final_waypoint.y + extension_length * math.sin(final_waypoint.heading)
        extension_point = Pos(extended_x, extended_y, final_waypoint.heading)
        path.append(extension_point)

    return path


def plan(
        origin: Pos,
        goal: Pos,
        obstacles: np.ndarray,
        vehicle_spec: VehicleSpec,
) -> Optional[List[Pos]]:
    """A* path planning with OBB collision checking"""

    start_time = time.perf_counter()  # Start timing

    print(f"\nA* Path Planning")
    print(f"  Origin: ({origin.x:.2f}, {origin.y:.2f}, {math.degrees(origin.heading):.0f}°)")
    print(f"  Goal:   ({goal.x:.2f}, {goal.y:.2f}, {math.degrees(goal.heading):.0f}°)")
    print(f"  Vehicle: {vehicle_spec.length:.2f}m long × {vehicle_spec.width:.2f}m wide")

    primitives = create_motion_primitives()
    print(f"  Motion primitives: {len(primitives)}")

    open_set = []
    visited: Set[StateKey] = set()

    start_node = SearchNode(
        f_score=heuristic(origin, goal),
        state=origin,
        g_score=0.0,
        path_segment=[origin]
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

        dist_to_goal = current.state.distance_to(goal)
        heading_error = current.state.heading_error_to(goal)

        if dist_to_goal < vehicle_spec.planned_xy_error and heading_error < vehicle_spec.planned_heading_error:
            elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time

            print(f"\nPATH FOUND")
            print(f"  Planning time: {elapsed_time:.3f}s")
            print(f"  Nodes expanded: {nodes_expanded}")
            print(f"  Nodes per second: {nodes_expanded / elapsed_time:.0f}")
            print(f"  Path cost: {current.g_score:.1f}m")
            print(f"  Final error: {dist_to_goal:.2f}m, {math.degrees(heading_error):.1f}°")

            path = reconstruct_path(current)
            print(f"  Reconstructed {len(path)} waypoints")
            
            # Debug: Show the discrepancy between goal and final waypoint
            if path:
                final_waypoint = path[-1]
                final_xy_error = final_waypoint.distance_to(goal)
                final_heading_error = final_waypoint.heading_error_to(goal)
                print(f"\n  Final waypoint vs goal:")
                print(f"    Goal heading: {math.degrees(goal.heading):.3f}°")
                print(f"    Final waypoint heading: {math.degrees(final_waypoint.heading):.3f}°")
                print(f"    XY error: {final_xy_error:.3f}m")
                print(f"    Heading error: {math.degrees(final_heading_error):.3f}°")
            
            return path

        # Expand neighbors
        for primitive in primitives:
            new_state, path_segment = primitive.apply(current.state)

            new_key = StateKey.from_pos(new_state)
            if new_key in visited:
                continue

            if not is_collision_free(path_segment, obstacles, vehicle_spec):
                continue

            # Bias towards longer arcs by penalizing shorter ones
            # Penalty is inversely proportional to arc length
            # Normalized by MAX_ARC_LENGTH, so the penalty is between 0 and ARC_LENGTH_BIAS_WEIGHT
            arc_bias_penalty = ARC_LENGTH_BIAS_WEIGHT * (1.0 - primitive.arc_length / MAX_ARC_LENGTH)

            g_score = current.g_score + abs(primitive.arc_length) + arc_bias_penalty
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
            print(f"  {nodes_expanded} nodes expanded")

    elapsed_time = time.perf_counter() - start_time
    print(f"NO PATH FOUND after {nodes_expanded} nodes in {elapsed_time:.3f}s")
    return None

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
from dataclasses import dataclass, field
from itertools import product

from collision import CollisionChecker
from world import Pos
from vehicle import VehicleSpec

pi = math.pi

# Motion primitive parameters  # TODO: it would be great to smooth the trajectory
ARC_LENGTHS = [0.3, 0.75, 1.5, 3.0]
CURVATURES = [0.0, pi / 24, -pi / 24, pi / 12, -pi / 12, pi / 6, -pi / 6]
PRIMITIVE_STEPS = 30

# Prefer longer arcs; TODO: don't forget to highlight in the report how essential this proved to be
ARC_LENGTH_BIAS_WEIGHT = 3.0
MAX_ARC_LENGTH = max(ARC_LENGTHS)

# Discretization resolution; these are magic parameters that had to be tuned to achieve good performance.
# Too coarse or too fine is failing the path planning.
XY_RESOLUTION = 0.75
THETA_RESOLUTION = math.radians(30.0)

# A* search limits
MAX_ITERATIONS = 30000
PROGRESS_INTERVAL = 1000

EPSILON = 1e-8  # Numerical tolerance for floating-point comparisons


@dataclass
class MotionPrimitive:
    """Motion primitive representing a curved or straight path segment"""
    arc_length: float
    curvature: float
    num_steps: int = PRIMITIVE_STEPS

    def apply(self, state: Pos) -> Tuple[Pos, List[Pos]]:
        """Generate path points as Pos objects and end state for this primitive"""
        path_points = []

        if abs(self.curvature) < EPSILON:  # Straight line
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

        else:  # Circular arc
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


def create_motion_primitives(vehicle_spec: VehicleSpec) -> List[MotionPrimitive]:
    """Generate all motion primitive combinations"""
    from vehicle import KinematicModel

    if vehicle_spec.kinematic_model == KinematicModel.ACKERMANN:
        # For Ackermann: filter curvatures to respect max steering angle
        kappa_max = math.tan(vehicle_spec.max_steering_angle) / vehicle_spec.wheelbase
        curvatures = [k for k in CURVATURES if abs(k) <= kappa_max]
    else:
        # For diff-drive: use the original curvature set
        curvatures = CURVATURES

    return [
        MotionPrimitive(arc_length=length, curvature=curvature)
        for length, curvature in product(ARC_LENGTHS, curvatures)
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


def heuristic(state: Pos, goal: Pos) -> float:
    """Heuristic with strong heading awareness near goal."""
    dist = math.hypot(state.x - goal.x, state.y - goal.y)

    heading_error = abs(state.heading - goal.heading)
    if heading_error > math.pi:
        heading_error = 2 * math.pi - heading_error

    heading_penalty = heading_error * 8 if dist < 5.0 else heading_error * 2

    return dist + heading_penalty


def reconstruct_path(node: SearchNode) -> List[Pos]:
    """Reconstruct the path from destination back to start"""
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
            # Wrap the heading to [-pi, pi] before adding to the path
            wrapped_heading = (pos.heading + math.pi) % (2 * math.pi) - math.pi
            path.append(Pos(pos.x, pos.y, wrapped_heading))

    return path


def plan(
        vehicle_spec: VehicleSpec,
        collision_checker: CollisionChecker,
) -> Optional[List[Pos]]:
    """A* path planning with fast overlay-based collision checking"""
    start_time = time.perf_counter()
    origin = vehicle_spec.origin
    destination = vehicle_spec.destination

    print(f"\nA* Path Planning")
    print(f"  Origin: ({origin.x:.2f}, {origin.y:.2f}, {math.degrees(origin.heading):.0f}°)")
    print(f"  Destination:   ({destination.x:.2f}, {destination.y:.2f}, {math.degrees(destination.heading):.0f}°)")
    print(f"  Vehicle: {vehicle_spec.length:.2f}m long × {vehicle_spec.width:.2f}m wide")

    primitives = create_motion_primitives(vehicle_spec)
    print(f"  Motion primitives: {len(primitives)}")

    open_set = []
    visited: Set[StateKey] = set()

    start_node = SearchNode(
        f_score=heuristic(origin, destination),
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

        dist_to_destination = current.state.distance_to(destination)
        heading_error = current.state.heading_error_to(destination)

        if dist_to_destination < vehicle_spec.planned_xy_error and heading_error < vehicle_spec.planned_heading_error:
            elapsed_time = time.perf_counter() - start_time

            print(f"\nPATH FOUND")
            print(f"  Planning time: {elapsed_time:.3f}s")
            print(f"  Nodes expanded: {nodes_expanded}")
            print(f"  Nodes per second: {nodes_expanded / elapsed_time:.0f}")
            print(f"  Path cost: {current.g_score:.1f}m")
            print(f"  Final error: {dist_to_destination:.2f}m, {math.degrees(heading_error):.1f}°")

            path = reconstruct_path(current)
            print(f"  Reconstructed {len(path)} waypoints")

            if path:
                final_waypoint = path[-1]
                final_xy_error = final_waypoint.distance_to(destination)
                final_heading_error = final_waypoint.heading_error_to(destination)
                print(f"\n  Final waypoint vs destination:")
                print(f"    Destination heading: {math.degrees(destination.heading):.3f}°")
                print(f"    Final waypoint heading: {math.degrees(final_waypoint.heading):.3f}°")
                print(f"    XY error: {final_xy_error:.3f}m")
                print(f"    Heading error: {math.degrees(final_heading_error):.3f}°")

            return path

        # Expand neighbors
        for primitive in primitives:
            new_state, path_segment = primitive.apply(current.state)

            # For Ackermann vehicles: reject spin-in-place primitives (zero arc length with curvature)
            from vehicle import KinematicModel
            if (vehicle_spec.kinematic_model == KinematicModel.ACKERMANN and
                    abs(primitive.arc_length) < 1.0 and abs(primitive.curvature) > 0.0):
                continue

            new_key = StateKey.from_pos(new_state)
            if new_key in visited:
                continue

            if not collision_checker.is_path_collision_free(path_segment):
                continue

            arc_bias_penalty = ARC_LENGTH_BIAS_WEIGHT * (1.0 - primitive.arc_length / MAX_ARC_LENGTH)

            g_score = current.g_score + abs(primitive.arc_length) + arc_bias_penalty
            f_score = g_score + heuristic(new_state, destination)

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

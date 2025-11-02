import logging
import math
import time
from scipy.ndimage import binary_dilation
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pygame

from world import Pos, World
from field import Cell
import reeds_shepp as rs
import networkx as nx

CONNECTOR_DENSITY = 0.1  # Fraction of empty cells to use as connectors

logger = logging.getLogger(__name__)

INITIAL_HEADING = math.pi / 2
FIREFIGHTING_DURATION = 5.0  # Time to suppress fire after arrival or ignition
MAX_POI_DISTANCE = 3

# Performance-critical constants for _integrate_path_step (called 26M+ times during roadmap building)
# Hardcoded to avoid repeated math.pi lookups and calculations
HALF_PI = 1.5707963267948966  # math.pi / 2
TWO_PI = 6.283185307179586  # 2 * math.pi
PI = 3.141592653589793  # math.pi

"""
Reeds-Shepp path segments are generated from cell center to cell center. This introduces a discrepancy, 
as an Ackermann steering pivots the car about the rear axle. The discrepancy will be corrected by a path follower.
"""


@dataclass(frozen=True)
class PathSegment:
    """A segment of a path with direction information"""

    start: Pos
    end: Pos
    is_forward: bool  # True for forward, False for reverse


class Firetruck:
    """
    Mercedes Unimog firetruck robot with Ackermann steering.
    Basic implementation with pose management, Reeds-Shepp path planning, and rendering.
    """

    def __init__(
        self, world: "World", preset_rows: tuple[int, int], preset_cols: tuple[int, int]
    ):
        self.world = world

        # Initialize position
        initial_location = self.world.field.initialize_location(
            preset_rows, preset_cols
        )
        self.pos = self._grid_to_pose(initial_location, INITIAL_HEADING)
        self.location = initial_location
        self.location_arrival = (
            0.0  # Time at which the firetruck arrived at the location
        )

        # Truck specifications
        self.length = 4.9  # m
        self.width = 2.2  # m
        self.wheelbase = 3.0  # m
        self.min_turning_radius = 13.0  # m
        self.max_velocity = 10.0  # m/s

        # Test path for visualization
        self.test_goal: Optional[Pos] = None
        self.test_path: Optional[List[Pos]] = None
        self.test_path_segments: Optional[List[PathSegment]] = (
            None  # Track direction info
        )

        # Generate points of interest for motion planning
        self.poi_locations = self._collect_poi_locations(origin=initial_location)
        self.poi_poses = self._collect_poi_poses()

        # Build roadmap
        self.roadmap = self._build_roadmap()

    def _grid_to_pose(self, grid_pos: tuple[int, int], heading: float) -> Pos:
        """Convert the grid location to a Pos with heading"""
        row, col = grid_pos
        x, y = self.world.grid_to_world(row, col)
        return Pos(x=x, y=y, heading=heading)

    def set_pose(self, pos: Pos):
        """Set the firetruck pose"""
        self.pos = pos
        old_location = self.location
        self.location = self.world.world_to_grid(pos.x, pos.y)

        # Reset presence timer if moved to a new cell
        if old_location != self.location:
            self.location_arrival = self.world.world_time

    def set_test_goal(self, goal: Pos):
        """Set a test goal and generate a Reeds-Shepp path for visualization"""
        self.test_goal = goal
        STEP_SIZE = 0.3  # meters between sampled waypoints

        try:
            # Generate the Reeds-Shepp path with direction information
            result = self._compute_reeds_shepp_path(
                start=self.pos,
                goal=goal,
                turning_radius=self.min_turning_radius,
                step_size=STEP_SIZE,
            )

            if result:
                self.test_path, self.test_path_segments = result
                logger.info(f"Generated RS path with {len(self.test_path)} waypoints")
            else:
                self.test_path = None
                self.test_path_segments = None
                logger.warning("No valid Reeds-Shepp path found")
        except Exception as e:
            self.test_path = None
            self.test_path_segments = None
            logger.error(f"Error generating Reeds-Shepp path: {e}")

    def _compute_reeds_shepp_path(
        self,
        start: Pos,
        goal: Pos,
        turning_radius: float,
        step_size: float,
    ) -> Optional[Tuple[List[Pos], List[PathSegment]]]:
        """Compute a Reeds-Shepp path from start to goal using standard Cartesian coordinates."""

        # Scale the problem by turning radius (the RS formulas assume unit turning radius)
        # The RS library expects angles in degrees
        scaled_start = (
            start.x / turning_radius,
            start.y / turning_radius,
            math.degrees(start.heading),
        )
        scaled_goal = (
            goal.x / turning_radius,
            goal.y / turning_radius,
            math.degrees(goal.heading),
        )

        # Get the optimal Reeds-Shepp path
        try:
            path_elements = rs.get_optimal_path(scaled_start, scaled_goal)
        except Exception as e:
            logger.error(f"Reeds-Shepp path generation failed: {e}")
            return None

        if not path_elements:
            return None

        # Sample the path at regular intervals
        return self._sample_path_elements(
            path_elements=path_elements,
            start=start,
            turning_radius=turning_radius,
            step_size=step_size,
        )

    def _sample_path_elements(
        self,
        path_elements: List[rs.PathElement],
        start: Pos,
        turning_radius: float,
        step_size: float,
    ) -> Tuple[List[Pos], List[PathSegment]]:
        """Sample a Reeds-Shepp path at regular intervals."""

        waypoints = [start]
        segments = []

        # Current state as we integrate along the path
        current_pos = start

        # Pre-compute to avoid repeated attribute lookups
        integrate = self._integrate_path_step

        for element in path_elements:
            # Calculate segment length and number of samples
            segment_length = element.param * turning_radius
            num_samples = max(1, int(segment_length / step_size))
            step_length = segment_length / num_samples

            is_forward = element.gear == rs.Gear.FORWARD

            # Cache element properties
            elem_steering = element.steering
            elem_gear = element.gear

            # Sample this segment
            for _ in range(num_samples):
                prev_pos = current_pos
                current_pos = integrate(
                    pos=current_pos,
                    step_length=step_length,
                    turning_radius=turning_radius,
                    steering=elem_steering,
                    gear=elem_gear,
                )
                waypoints.append(current_pos)
                segments.append(PathSegment(prev_pos, current_pos, is_forward))

        return waypoints, segments

    @staticmethod
    def _integrate_path_step(
        pos: Pos,
        step_length: float,
        turning_radius: float,
        steering: rs.Steering,
        gear: rs.Gear,
    ) -> Pos:
        """Integrate one step along a Reeds-Shepp path segment."""

        x, y, heading = pos.x, pos.y, pos.heading

        # Pre-compute gear value (avoid attribute lookup in tight loop)
        gear_val = gear.value

        if steering == rs.Steering.STRAIGHT:
            # Straight line motion
            step_gear = step_length * gear_val
            dx = step_gear * math.cos(heading)
            dy = step_gear * math.sin(heading)
            return Pos(x + dx, y + dy, heading)

        else:
            # Circular arc motion
            # Pre-compute steering value
            steering_val = steering.value

            # Angular change for this step
            delta_heading = -(step_length / turning_radius) * steering_val * gear_val

            # Compute the center of the arc using hardcoded HALF_PI constant
            center_angle = heading + HALF_PI * (-steering_val)
            cos_center = math.cos(center_angle)
            sin_center = math.sin(center_angle)
            cx = x + turning_radius * cos_center
            cy = y + turning_radius * sin_center

            # Update heading (normalize to [-pi, pi])
            new_heading = heading + delta_heading
            # Fast normalization using hardcoded constants
            if new_heading > PI:
                new_heading -= TWO_PI
            elif new_heading < -PI:
                new_heading += TWO_PI

            # Compute the new position on the circle
            new_center_angle = new_heading + HALF_PI * (-steering_val)
            new_x = cx - turning_radius * math.cos(new_center_angle)
            new_y = cy - turning_radius * math.sin(new_center_angle)

            return Pos(new_x, new_y, new_heading)

    def get_location(self) -> tuple[int, int]:
        """Get the current grid location as (row, col)"""
        return int(self.location[0]), int(self.location[1])

    def update(self):
        """Update the firetruck state"""
        # TODO: Implement kinematics and control
        self._suppress_fires()

    def _suppress_fires(self):
        """Suppress fires in adjacent cells after sufficient firefighting time"""
        current_time = self.world.world_time
        neighbors = self.world.field.get_cell_neighbors(self.location)

        # Filter for burning neighbors
        burning_neighbors = [
            pos
            for pos in neighbors
            if self.world.field.get_cell(pos[0], pos[1]) == Cell.BURNING
        ]

        # Suppress fires that have been burning for at least FIREFIGHTING_DURATION
        # since the latter of: truck arrival or fire ignition
        suppressed_count = 0
        for pos in burning_neighbors:
            ignition_time = self.world.field.ignition_times.get(pos, current_time)
            firefighting_start = max(self.location_arrival, ignition_time)

            if current_time >= firefighting_start + FIREFIGHTING_DURATION:
                if self.world.field.suppress(pos[0], pos[1]):
                    suppressed_count += 1

        if suppressed_count > 0:
            logger.debug(f"Firetruck suppressed {suppressed_count} fire(s)")

    def _collect_poi_locations(self, origin: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Collect all points of interest for motion planning.
        POIs are EMPTY cells adjacent to OBSTACLE cells in the initial field and connector locations.
        Always includes the firetruck's origin location.

        Returns a list of locations (row, col) for POIs, connectors, and origin
        """
        field = self.world.field

        # Find empty cells adjacent to obstacles
        obstacle_mask = field.cells == Cell.OBSTACLE
        dilated_obstacles = binary_dilation(obstacle_mask, structure=np.ones((3, 3)))
        empty_cells = field.cells == Cell.EMPTY

        # Valid POIs: empty cells adjacent to obstacles
        poi_mask = empty_cells & dilated_obstacles

        # Extract locations as (row, col) tuples
        poi_rows, poi_cols = np.where(poi_mask)
        poi_locations = list(zip(poi_rows, poi_cols))

        # Generate connector locations: random empty cells not already in POIs
        # These serve as intermediate waypoints for Reeds-Shepp maneuvering
        connector_mask = empty_cells & ~poi_mask
        connector_rows, connector_cols = np.where(connector_mask)

        if len(connector_rows) > 0:
            # Select a random subset of empty cells as connectors
            num_connectors = max(1, int(CONNECTOR_DENSITY * len(connector_rows)))
            connector_indices = field.rng.choice(
                len(connector_rows), size=num_connectors, replace=False
            )
            connector_locations = [
                (connector_rows[i], connector_cols[i]) for i in connector_indices
            ]
        else:
            connector_locations = []

        # Combine all locations, avoiding duplicates
        all_locations = poi_locations + connector_locations
        if origin not in all_locations:
            all_locations.append(origin)

        return all_locations

    def _collect_poi_poses(self):
        NUM_HEADINGS = 1
        # TODO: adjust the headings number back to 8
        headings = [i * (2 * math.pi / NUM_HEADINGS) for i in range(NUM_HEADINGS)]

        # Create Pos nodes
        poses = []
        for row, col in self.poi_locations:
            x, y = self.world.grid_to_world(row, col)
            for heading in headings:
                poses.append(Pos(x, y, heading))
        return poses

    @staticmethod
    def _should_poi_connect(
        location1: Tuple[int, int], location2: Tuple[int, int]
    ) -> bool:
        # Calculate Chebyshev distance between locations
        distance = max(
            abs(location1[0] - location2[0]), abs(location1[1] - location2[1])
        )
        return distance <= MAX_POI_DISTANCE

    def _build_roadmap(self) -> nx.DiGraph:
        """
        Build a directed graph roadmap connecting POI poses with Reeds-Shepp arcs.
        Returns a NetworkX DiGraph where:
        - Nodes are Pos (x, y, heading)
        - Edges contain path waypoints, segments, and cost
        """
        import networkx as nx

        start_time = time.time()
        logger.info(f"Building roadmap with {len(self.poi_poses)} poses...")

        G = nx.DiGraph()

        for pose in self.poi_poses:
            G.add_node(pose)

        # Group poses by location for efficient distance filtering
        location_to_poses = {}
        for pose in self.poi_poses:
            loc = self.world.world_to_grid(pose.x, pose.y)
            if loc not in location_to_poses:
                location_to_poses[loc] = []
            location_to_poses[loc].append(pose)

        # Generate edges between nearby poses
        edge_count = 0
        collision_count = 0

        for i, start_pose in enumerate(self.poi_poses):
            start_loc = self.world.world_to_grid(start_pose.x, start_pose.y)

            # Find nearby locations to connect to
            for goal_loc, goal_poses in location_to_poses.items():
                # Skip self-connections
                if start_loc == goal_loc:
                    continue

                # Distance filter in grid space
                if not self._should_poi_connect(start_loc, goal_loc):
                    continue

                # Try connecting to each heading at this location
                for goal_pose in goal_poses:
                    # Generate Reeds-Shepp path
                    result = self._compute_reeds_shepp_path(
                        start=start_pose,
                        goal=goal_pose,
                        turning_radius=self.min_turning_radius,
                        step_size=1.0,
                    )

                    if not result:
                        continue

                    waypoints, segments = result

                    # TODO: collision checks
                    # Quick collision check
                    # if self._check_path_collision(waypoints):
                    #     collision_count += 1
                    #     continue

                    # Compute path cost (total path length)
                    path_length = sum(
                        seg.start.distance_to(seg.end) for seg in segments
                    )

                    # Add edge to graph
                    G.add_edge(
                        start_pose,
                        goal_pose,
                        weight=path_length,
                        waypoints=waypoints,
                        segments=segments,
                    )
                    edge_count += 1

            # Progress logging
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.debug(
                    f"Processed {i + 1}/{len(self.poi_poses)} poses, "
                    f"{edge_count} edges, {collision_count} collisions "
                    f"(elapsed: {elapsed:.1f}s)"
                )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Roadmap built in {elapsed_time:.1f}s: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{collision_count} paths rejected due to collision"
        )

        return G

    # Rendering methods
    def render(self):
        """Render the firetruck at its current position"""
        FIRETRUCK_COLOR = (220, 50, 50)  # Red
        FIRETRUCK_STRIPE_COLOR = (255, 255, 255)  # White stripe

        ppm = self.world.pixels_per_meter
        Lpx = math.floor(self.length * ppm) - 1  # Adjusted for 7 PPM
        Wpx = math.floor(self.width * ppm)

        # Create a surface and draw truck body
        surf = pygame.Surface((Lpx, Wpx), pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            FIRETRUCK_COLOR,
            pygame.Rect(0, 0, Lpx, Wpx),
            border_radius=max(2, Wpx // 4),
        )

        # Draw the front indicator stripe
        stripe_w = 3  # pixels
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(
            surf,
            FIRETRUCK_STRIPE_COLOR,
            (stripe_x, 3, stripe_w, Wpx - 6),
            border_radius=2,
        )

        # Rotate for rendering
        rotated = pygame.transform.rotate(surf, math.degrees(self.pos.heading))

        # Convert to pixel coordinates and blit
        px, py = self.world.world_to_pixel(self.pos.x, self.pos.y)
        rect = rotated.get_rect(center=(px, py))
        self.world.display.blit(rotated, rect.topleft)

    def render_route(self, route: List[Pos], color: Tuple[int, int, int]):
        """Render a route as a polyline"""
        if len(route) < 2:
            return  # Need at least 2 points to draw a line

        pts = [self.world.world_to_pixel(pos.x, pos.y) for pos in route]
        pygame.draw.lines(self.world.display, color, False, pts, 2)

    def render_test_path(self):
        """Render the test Reeds-Shepp path if one exists"""
        if self.test_path is None or len(self.test_path) < 2:
            return

        FORWARD_COLOR = (0, 200, 255)  # Cyan for forward
        REVERSE_COLOR = (255, 100, 0)  # Orange for reverse
        GOAL_COLOR = (0, 255, 0)  # Green

        # Draw path segments with different colors for forward/reverse
        if self.test_path_segments:
            for segment in self.test_path_segments:
                color = FORWARD_COLOR if segment.is_forward else REVERSE_COLOR
                start_px, start_py = self.world.world_to_pixel(
                    segment.start.x, segment.start.y
                )
                end_px, end_py = self.world.world_to_pixel(segment.end.x, segment.end.y)
                pygame.draw.line(
                    self.world.display, color, (start_px, start_py), (end_px, end_py), 2
                )
        else:
            # Fallback: draw the entire path in cyan if segment info not available
            self.render_route(self.test_path, FORWARD_COLOR)

        # Draw goal pose indicator
        if self.test_goal:
            goal_px, goal_py = self.world.world_to_pixel(
                self.test_goal.x, self.test_goal.y
            )

            # Draw circle at goal
            pygame.draw.circle(self.world.display, GOAL_COLOR, (goal_px, goal_py), 8, 2)

            # Draw heading indicator
            # Negate heading for pygame's coordinate system
            arrow_len = 15
            end_x = goal_px + int(arrow_len * math.cos(-self.test_goal.heading))
            end_y = goal_py + int(arrow_len * math.sin(-self.test_goal.heading))
            pygame.draw.line(
                self.world.display, GOAL_COLOR, (goal_px, goal_py), (end_x, end_y), 2
            )

    def render_poi_locations(self):
        """Render POI location markers"""
        cell_dim = self.world.cell_dimensions
        LOCATION_COLOR = (220, 50, 50, 100)
        CIRCLE_RADIUS = cell_dim // 2 - 4

        surface = pygame.Surface(
            (self.world.display.get_width(), self.world.display.get_height()),
            pygame.SRCALPHA,
        )

        for row, col in self.poi_locations:
            center_x = int((col + 0.5) * cell_dim)
            center_y = int((row + 0.5) * cell_dim)
            pygame.draw.circle(
                surface,
                LOCATION_COLOR,
                (center_x, center_y),
                CIRCLE_RADIUS,
            )

        self.world.display.blit(surface, (0, 0))

    def render_roadmap(self):
        """
        Render the roadmap for visual validation.
        Shows a subset of edges to avoid clutter.
        """
        if self.roadmap is None:
            return

        EDGE_COLOR = (150, 150, 255, 100)
        NODE_COLOR = (100, 100, 200, 120)

        # Create transparent surface
        surface = pygame.Surface(
            (self.world.display.get_width(), self.world.display.get_height()),
            pygame.SRCALPHA,
        )

        # Draw edges
        edges = list(self.roadmap.edges(data=True))

        for start_pose, goal_pose, data in edges:
            waypoints = data.get("waypoints", [])
            if len(waypoints) < 2:
                continue

            # Draw the edge as a thin line
            points = [self.world.world_to_pixel(wp.x, wp.y) for wp in waypoints]
            pygame.draw.lines(surface, EDGE_COLOR, False, points, 1)

        # Draw nodes (just small circles at POI locations)
        drawn_locations = set()
        for pose in self.poi_poses:
            loc = self.world.world_to_grid(pose.x, pose.y)
            if loc in drawn_locations:
                continue
            drawn_locations.add(loc)

            px, py = self.world.world_to_pixel(pose.x, pose.y)
            pygame.draw.circle(surface, NODE_COLOR, (px, py), 3)

        self.world.display.blit(surface, (0, 0))

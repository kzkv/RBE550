import logging
import math
from scipy.ndimage import binary_dilation
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pygame

from world import Pos, World
from field import Cell
import reeds_shepp as rs

CONNECTOR_DENSITY = 0.1  # Fraction of empty cells to use as connectors

logger = logging.getLogger(__name__)

INITIAL_HEADING = math.pi / 2
FIREFIGHTING_DURATION = 5.0  # Time to suppress fire after arrival or ignition

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
        step_size = 0.3  # meters between sampled waypoints

        try:
            # Generate the Reeds-Shepp path with direction information
            result = self._compute_reeds_shepp_path(
                start=self.pos,
                goal=goal,
                turning_radius=self.min_turning_radius,
                step_size=step_size,
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

        waypoints = [start]  # Start with the initial pose
        segments = []

        # Current state as we integrate along the path
        current_pos = start

        for element in path_elements:
            # Calculate segment length and number of samples
            segment_length = element.param * turning_radius
            num_samples = max(1, int(segment_length / step_size))
            step_length = segment_length / num_samples

            is_forward = element.gear == rs.Gear.FORWARD

            # Sample this segment
            for _ in range(num_samples):
                prev_pos = current_pos
                current_pos = self._integrate_path_step(
                    pos=current_pos,
                    step_length=step_length,
                    turning_radius=turning_radius,
                    steering=element.steering,
                    gear=element.gear,
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

        if steering == rs.Steering.STRAIGHT:
            # Straight line motion
            dx = step_length * gear.value * math.cos(heading)
            dy = step_length * gear.value * math.sin(heading)
            return Pos(x + dx, y + dy, heading)

        else:
            # Circular arc motion; library interface:
            #   steering: LEFT=-1 (CCW), RIGHT=1 (CW)
            #   gear: FORWARD=1, BACKWARD=-1

            # Angular change for this step
            # Negative because: LEFT (steering=-1) should give positive angle change (CCW)
            delta_heading = (
                -(step_length / turning_radius) * steering.value * gear.value  # type: ignore
            )

            # Compute center of the arc
            # For LEFT turn (steering=-1): center is perpendicular left (heading + pi/2)
            # For RIGHT turn (steering=1): center is perpendicular right (heading - pi/2)
            center_angle = heading + (math.pi / 2) * (-steering.value)
            cx = x + turning_radius * math.cos(center_angle)
            cy = y + turning_radius * math.sin(center_angle)

            # Update heading
            new_heading = heading + delta_heading
            new_heading = (new_heading + math.pi) % (
                2 * math.pi
            ) - math.pi  # Normalize to [-pi, pi]

            # Compute the new position on the circle
            new_center_angle = new_heading + (math.pi / 2) * (-steering.value)
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

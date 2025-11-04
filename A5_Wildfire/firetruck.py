import hashlib
import logging
import math
import time
from pathlib import Path

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
import pygame

from world import Pos, World
from field import Cell
import reeds_shepp as rs
import networkx as nx

MAX_PATH_SEARCH_LENGTH = (
    500.0  # Maximum path length to consider (meters); simple safeguard
)

FORCE_REBUILD_ROADMAP = False  # Set to True to force roadmap rebuild

logger = logging.getLogger(__name__)

INITIAL_HEADING = math.pi / 2
FIREFIGHTING_DURATION = 5.0  # Time to suppress fire after arrival or ignition
MAX_POI_DISTANCE = 15  # Max distance in cells for the POIs to get connected
NUM_HEADINGS = 12  # Headings per POI

COVERAGE_RADIUS_METERS = 10.0  # meters
MAX_POI_COUNT = 120  # Maximum number of POIs to select
POI_EXCLUSION_RADIUS_FINE_CELLS = 50  # Fine cells - avoid clustering POIs too closely

TOP_CANDIDATES_COUNT = (
    20  # Number of POIs to select to be rendered (and considered for the goal)
)

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

        # Initialize position - self.pos now represents REAR AXLE
        initial_location = self.world.field.initialize_location(
            preset_rows, preset_cols
        )
        self.pos = self._grid_to_pose(initial_location, INITIAL_HEADING)
        self.location = initial_location
        self.location_arrival = 0.0

        # Truck specifications
        self.length = 4.9  # m
        self.width = 2.2  # m
        self.wheelbase = 3.0  # m
        self.min_turning_radius = 13.0  # m
        self.max_velocity = 10.0  # m/s
        self.max_velocity_reverse = 5.0  # m/s
        self.max_acceleration = 2.0  # m/s^2
        self.max_deceleration = 3.0  # m/s^2

        # Initialize vehicle controller
        self.controller = VehicleController(
            wheelbase=self.wheelbase,
            max_velocity_forward=self.max_velocity,
            max_velocity_reverse=self.max_velocity_reverse,
            max_acceleration=self.max_acceleration,
            max_deceleration=self.max_deceleration,
            min_turning_radius=self.min_turning_radius,
        )

        # Path following state
        self.current_segment_index = 0
        self.path_complete = True

        # Generate points of interest for motion planning
        self.poi_locations = self._collect_poi_locations(origin=initial_location)
        self.poi_poses = self._collect_poi_poses()

        # Precompute which fine-grid cells each POI can reach
        self.poi_coverage_map = self._precompute_poi_coverage()

        # Build location-indexed dictionary for O(1) lookup
        # This is a significant speed-up for the A* path finding
        # TODO: this will be removed, as POIs are no longer location-bound
        self.poses_by_location = {}
        for pose in self.poi_poses:
            loc = pose.location
            if loc not in self.poses_by_location:
                self.poses_by_location[loc] = []
            self.poses_by_location[loc].append(pose)

        # Build roadmap (with caching)
        self.roadmap = self._load_or_build_roadmap()

        # Pre-render the roadmap surface (immutable)
        self.roadmap_surface = self._create_roadmap_surface()

        # Path planning state
        self.planned_path_segments = []  # List of PathSegment for the current path

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
    ) -> Optional[Tuple[List[Pos], List[PathSegment]]]:
        """Sample a Reeds-Shepp path at regular intervals, with early collision detection.
        Returns None if collision is detected, otherwise returns (waypoints, segments).
        """

        waypoints = [start]
        segments = []

        # Current state as we integrate along the path
        current_pos = start

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
                current_pos = self._integrate_path_step(
                    pos=current_pos,
                    step_length=step_length,
                    turning_radius=turning_radius,
                    steering=elem_steering,
                    gear=elem_gear,
                )

                # Early collision check using the inflated overlay
                # This checks: obstacles (inflated) + borders (inflated) + out-of-bounds
                if self.world.field.check_collision_at_pos(current_pos):
                    return None

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
            delta_heading = (
                -(step_length / turning_radius) * steering_val * gear_val  # noqa
            )

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
        dt = self.world.dt_world

        if self.planned_path_segments and not self.path_complete:
            # Everything operates in the rear-axle frame now
            new_pose, new_index, complete = self.controller.update(
                dt=dt,
                current_pose=self.pos,  # Already rear axle
                path_segments=self.planned_path_segments,
                segment_index=self.current_segment_index,
            )

            self.set_pose(new_pose)
            self.current_segment_index = new_index
            self.path_complete = complete

            if complete:
                logger.info("Path following complete")

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
        Collect points of interest for motion planning using a fine-grid coverage heatmap.
        POIs are selected from passable fine-grid cells that cover the most obstacles,
        with an exclusion radius to avoid clustering.
        Continues until MAX_POI_COUNT unique POIs are collected.

        Returns a list of fine-grid locations (row, col) for POIs
        """
        field = self.world.field

        # Get coverage heatmap from the field (fine grid resolution)
        heatmap = field.compute_poi_coverage_heatmap(COVERAGE_RADIUS_METERS)

        # Create a working copy for exclusion zones
        heatmap_working = heatmap.copy()

        selected_fine_pois = []  # List of (fine_row, fine_col) tuples

        # Greedy selection: iteratively pick the best remaining position
        iterations = 0
        max_iterations = MAX_POI_COUNT * 10  # Safety limit to prevent infinite loops

        while len(selected_fine_pois) < MAX_POI_COUNT and iterations < max_iterations:
            iterations += 1

            # Find the cell with maximum coverage
            max_coverage = heatmap_working.max()

            if max_coverage == 0:
                break  # No more viable POIs

            # Get all cells with max coverage
            max_indices = np.argwhere(heatmap_working == max_coverage)

            if len(max_indices) == 0:
                break

            # Pick the first one
            fine_row, fine_col = max_indices[0]
            selected_fine_pois.append((fine_row, fine_col))

            # Apply an exclusion zone around this POI (in fine grid)
            for dr in range(
                -POI_EXCLUSION_RADIUS_FINE_CELLS, POI_EXCLUSION_RADIUS_FINE_CELLS + 1
            ):
                for dc in range(
                    -POI_EXCLUSION_RADIUS_FINE_CELLS,
                    POI_EXCLUSION_RADIUS_FINE_CELLS + 1,
                ):
                    excl_row = fine_row + dr
                    excl_col = fine_col + dc

                    # Check bounds and Euclidean distance
                    if (
                        0 <= excl_row < field.fine_grid_size
                        and 0 <= excl_col < field.fine_grid_size
                    ):
                        distance = np.sqrt(dr * dr + dc * dc)
                        if distance <= POI_EXCLUSION_RADIUS_FINE_CELLS:
                            heatmap_working[excl_row, excl_col] = 0

        logger.info(
            f"Selected {len(selected_fine_pois)} fine-grid POIs after {iterations} iterations"
        )

        # Convert fine-grid origin to fine-grid coordinates if provided
        if origin is not None:
            cells_per_coarse = int(
                field.world.cell_size / field.collision_discretization
            )
            origin_fine_row = origin[0] * cells_per_coarse + cells_per_coarse // 2
            origin_fine_col = origin[1] * cells_per_coarse + cells_per_coarse // 2
            origin_fine = (origin_fine_row, origin_fine_col)

            # Add origin if not already covered by a nearby POI
            if origin_fine not in selected_fine_pois:
                selected_fine_pois.append(origin_fine)

        logger.info(f"Final POI count: {len(selected_fine_pois)} fine-grid locations")

        return selected_fine_pois

    def _collect_poi_poses(self):
        """Generate poses with multiple headings for each fine-grid POI location"""
        headings = [i * (2 * math.pi / NUM_HEADINGS) for i in range(NUM_HEADINGS)]

        # Create Pos nodes with location information
        poses = []
        field = self.world.field

        for fine_row, fine_col in self.poi_locations:
            # Convert fine-grid to world coordinates
            x = (fine_col + 0.5) * field.collision_discretization
            y = (
                field.grid_dimensions * field.world.cell_size
                - (fine_row + 0.5) * field.collision_discretization
            )

            for heading in headings:
                poses.append(Pos(x, y, heading, location=(fine_row, fine_col)))

        return poses

    def _precompute_poi_coverage(self) -> dict:
        """
        Precompute which COARSE-grid cells each POI can reach.
        Returns dict mapping POI location -> set of coarse-grid cells within range.
        """
        coverage_map = {}
        field = self.world.field
        cells_per_coarse = int(field.world.cell_size / field.collision_discretization)
        coverage_radius_coarse = int(
            np.ceil(COVERAGE_RADIUS_METERS / field.world.cell_size)
        )

        for poi_fine_loc in self.poi_locations:
            # Convert POI to coarse grid
            poi_coarse_row = poi_fine_loc[0] // cells_per_coarse
            poi_coarse_col = poi_fine_loc[1] // cells_per_coarse

            covered_cells = set()

            # Check coarse cells within the bounding box
            for dr in range(-coverage_radius_coarse - 1, coverage_radius_coarse + 2):
                for dc in range(
                    -coverage_radius_coarse - 1, coverage_radius_coarse + 2
                ):
                    cell_row = poi_coarse_row + dr
                    cell_col = poi_coarse_col + dc

                    # Check bounds
                    if (
                        0 <= cell_row < field.grid_dimensions
                        and 0 <= cell_col < field.grid_dimensions
                    ):
                        # Check Euclidean distance in meters
                        dx = dc * field.world.cell_size
                        dy = dr * field.world.cell_size
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist <= COVERAGE_RADIUS_METERS:
                            covered_cells.add((cell_row, cell_col))

            coverage_map[poi_fine_loc] = covered_cells

        return coverage_map

    def rank_pois_by_priority(self) -> List[Tuple[Tuple[int, int], float, int]]:
        """Fast POI ranking using precomputed coverage and set intersection."""
        truck_loc = self.get_location()
        field = self.world.field

        # Truck position in fine grid
        cells_per_coarse = int(field.world.cell_size / field.collision_discretization)
        truck_fine = np.array(
            [
                truck_loc[0] * cells_per_coarse + cells_per_coarse // 2,
                truck_loc[1] * cells_per_coarse + cells_per_coarse // 2,
            ]
        )

        # Get burning cells (coarse grid)
        burning_coarse = np.argwhere(field.cells == Cell.BURNING)
        burning_set = set(map(tuple, burning_coarse))

        # Vectorized distances
        poi_array = np.array(self.poi_locations)
        distances = np.sqrt(np.sum((poi_array - truck_fine) ** 2, axis=1))

        # Count fires via fast-set intersection
        results = []
        for i, poi_loc in enumerate(self.poi_locations):
            covered_cells = self.poi_coverage_map[poi_loc]
            fire_count = len(
                covered_cells & burning_set
            )  # O(min(|covered|, |burning|))

            if fire_count > 0:
                results.append((poi_loc, distances[i], fire_count))

        results.sort(key=lambda x: (-x[2], x[1]))
        return results

    @staticmethod
    def _should_poi_connect(
        location1: Tuple[int, int], location2: Tuple[int, int]
    ) -> bool:
        # Calculate Chebyshev distance between locations
        distance = max(
            abs(location1[0] - location2[0]), abs(location1[1] - location2[1])
        )
        return distance <= MAX_POI_DISTANCE

    def _get_roadmap_cache_path(self) -> Path:
        """
        Generate a unique cache filename based on the field configuration.
        Uses a hash of the field state to ensure cache validity.
        """
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        # Create a hash from field configuration and relevant parameters
        hash_data = {
            "field_cells": self.world.field.cells.tobytes(),
            "poi_locations": tuple(sorted(self.poi_locations)),
            "min_turning_radius": self.min_turning_radius,
            "max_poi_distance": MAX_POI_DISTANCE,
            "num_headings": NUM_HEADINGS,
            "poi_exclusion_radius_fine_cells": POI_EXCLUSION_RADIUS_FINE_CELLS,
        }

        # Serialize and hash
        hash_str = hashlib.sha256(str(hash_data).encode()).hexdigest()[:16]

        logger.info(f"Roadmap cache hash: {hash_str}")

        return cache_dir / f"roadmap_{hash_str}.pkl"

    @staticmethod
    def _save_roadmap(roadmap: nx.DiGraph, cache_path: Path):
        """Save the roadmap to disk using pickle."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(roadmap, f, protocol=pickle.HIGHEST_PROTOCOL)  # noqa
            logger.info(f"Roadmap saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save roadmap cache: {e}")

    @staticmethod
    def _load_roadmap(cache_path: Path) -> Optional[nx.DiGraph]:
        """Load a cached roadmap from the disk."""
        try:
            with open(cache_path, "rb") as f:
                roadmap = pickle.load(f)
            logger.info(f"Loaded cached roadmap from {cache_path}")
            logger.info(
                f"Roadmap: {roadmap.number_of_nodes()} nodes, "
                f"{roadmap.number_of_edges()} edges"
            )
            return roadmap
        except FileNotFoundError:
            logger.debug(f"No cached roadmap found at {cache_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load roadmap cache: {e}")
            return None

    def _load_or_build_roadmap(self) -> nx.DiGraph:
        """
        Load roadmap from cache if available, otherwise build and cache it.
        """
        cache_path = self._get_roadmap_cache_path()

        # Check if force rebuild is enabled
        if FORCE_REBUILD_ROADMAP:
            logger.info("Force rebuild enabled - skipping cache lookup")
            roadmap = self._build_roadmap()
            self._save_roadmap(roadmap, cache_path)
            return roadmap

        # Try to load from the cache first
        roadmap = self._load_roadmap(cache_path)
        if roadmap is not None:
            return roadmap

        # Cache miss - build the roadmap
        logger.info("Building roadmap from scratch")
        roadmap = self._build_roadmap()

        # Save to cache for next time
        self._save_roadmap(roadmap, cache_path)

        return roadmap

    def _build_roadmap(self) -> nx.DiGraph:
        """
        Build a directed graph roadmap connecting POI poses with Reeds-Shepp arcs.
        Returns a NetworkX DiGraph where:
        - Nodes are Pos (x, y, heading)
        - Edges contain path waypoints, segments, and cost
        """
        start_time = time.time()
        logger.info(f"Building roadmap with {len(self.poi_poses)} poses")

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
                    # Generate Reeds-Shepp path with collision checking during sampling
                    result = self._compute_reeds_shepp_path(
                        start=start_pose,
                        goal=goal_pose,
                        turning_radius=self.min_turning_radius,
                        step_size=1.0,
                    )

                    if not result:
                        collision_count += 1
                        continue

                    waypoints, segments = result

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

    def find_nearest_poi_to_location(
        self, target_location: tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Find the nearest POI (fine-grid location) to the target coarse-grid location."""
        if not self.poi_locations:
            return None

        field = self.world.field
        cells_per_coarse = int(field.world.cell_size / field.collision_discretization)

        # Convert target coarse location to fine grid
        target_fine_row = target_location[0] * cells_per_coarse + cells_per_coarse // 2
        target_fine_col = target_location[1] * cells_per_coarse + cells_per_coarse // 2

        # Find the nearest POI using Euclidean distance
        min_distance = float("inf")
        nearest_poi = None

        for poi_loc in self.poi_locations:
            poi_fine_row, poi_fine_col = poi_loc

            # Calculate Euclidean distance in fine grid space
            distance = math.sqrt(
                (poi_fine_row - target_fine_row) ** 2
                + (poi_fine_col - target_fine_col) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                nearest_poi = poi_loc

        logger.debug(
            f"Found nearest POI at {nearest_poi} "
            f"(distance: {min_distance * field.collision_discretization:.2f}m)"
        )

        return nearest_poi

    def plan_path_to_poi(self, target_poi: Tuple[int, int]) -> bool:
        """
        Plan a path from the current pose to a target POI (fine-grid location).
        Finds the shortest path across all possible goal headings at that POI.
        """
        if not self.roadmap:
            logger.warning("No roadmap available for path planning")
            self.clear_planned_path()
            return False

        # Find the closest pose in the roadmap to our current position
        start_pose = self._find_closest_roadmap_pose(self.pos)
        if start_pose is None:
            logger.warning(
                f"Cannot find a roadmap pose near current position {self.pos}"
            )
            self.clear_planned_path()
            return False

        # Get all goal poses at the target POI location
        goal_poses = [pose for pose in self.poi_poses if pose.location == target_poi]

        if not goal_poses:
            logger.warning(f"Target POI {target_poi} has no poses in the roadmap")
            self.clear_planned_path()
            return False

        # Heuristic for A*: Euclidean distance between poses
        def length_heuristic(node_a, node_b):
            """Euclidean distance heuristic for A* in the roadmap"""
            return node_a.distance_to(node_b)

        # Bias factor for reverse motion (1.0 = no bias, >1.0 = penalize reverse)
        REVERSE_PENALTY_FACTOR = 1.1

        # Try to find paths to ALL goal poses and pick the shortest
        best_path = None
        best_cost = float("inf")

        for goal_pose in goal_poses:
            try:
                # Use A* to find the shortest path in the roadmap
                path = nx.astar_path(
                    self.roadmap,
                    source=start_pose,
                    target=goal_pose,
                    heuristic=length_heuristic,
                    weight="weight",
                    cutoff=MAX_PATH_SEARCH_LENGTH,
                )

                # Calculate biased path cost (penalize reverse segments)
                biased_cost = 0.0
                for i in range(len(path) - 1):
                    edge_data = self.roadmap[path[i]][path[i + 1]]
                    segments = edge_data.get("segments", [])

                    # Calculate cost with reverse penalty
                    for segment in segments:
                        segment_length = segment.start.distance_to(segment.end)
                        if segment.is_forward:
                            biased_cost += segment_length
                        else:
                            biased_cost += segment_length * REVERSE_PENALTY_FACTOR

                # Keep track of the path with the lowest biased cost
                if biased_cost < best_cost:
                    best_cost = biased_cost
                    best_path = path

            except nx.NetworkXNoPath:
                continue

        if not best_path:
            logger.warning(f"No path found to POI {target_poi}")
            self.clear_planned_path()
            return False

        # Extract segments from the best path
        self.planned_path_segments = []
        for i in range(len(best_path) - 1):
            edge_data = self.roadmap[best_path[i]][best_path[i + 1]]
            segments = edge_data.get("segments", [])
            self.planned_path_segments.extend(segments)

        # Reset path following state
        self.current_segment_index = 0
        self.path_complete = False

        # Calculate actual (unbiased) path length for logging
        actual_length = sum(
            seg.start.distance_to(seg.end) for seg in self.planned_path_segments
        )

        # Count forward vs. reverse segments
        forward_count = sum(1 for seg in self.planned_path_segments if seg.is_forward)
        reverse_count = len(self.planned_path_segments) - forward_count

        logger.info(
            f"Planned path to POI {target_poi}: "
            f"{len(self.planned_path_segments)} segments "
            f"({forward_count} forward, {reverse_count} reverse), "
            f"length {actual_length:.1f}m (biased cost: {best_cost:.1f})"
        )
        return True

    def _find_closest_roadmap_pose(self, pos: Pos) -> Optional[Pos]:
        """Find the closest pose in the roadmap to the given position."""
        if not self.roadmap or self.roadmap.number_of_nodes() == 0:
            logger.error("Roadmap is empty or not initialized")
            return None

        # Both current pose and roadmap poses are in rear-axle frame - direct comparison
        min_distance = float("inf")
        closest_pose = None

        for roadmap_pose in self.roadmap.nodes():
            # Calculate distance (position and heading difference)
            position_dist = pos.distance_to(roadmap_pose)
            heading_diff = pos.heading_error_to(roadmap_pose)

            # Weighted distance: prioritize position over heading
            total_dist = position_dist + 0.5 * heading_diff

            if total_dist < min_distance:
                min_distance = total_dist
                closest_pose = roadmap_pose

        if closest_pose is None:
            logger.error(f"Could not find any roadmap pose near {pos}")
        else:
            logger.debug(f"Found closest roadmap pose at distance {min_distance:.2f}m")

        return closest_pose

    def clear_planned_path(self):
        """Clear the planned path."""
        self.planned_path_segments = []
        self.current_segment_index = 0
        self.path_complete = True
        self.controller.stop()

    # Rendering methods
    def render(self):
        """Render the firetruck with body centered on visual and rear axle indicator"""
        FIRETRUCK_COLOR = (220, 50, 50)  # Red
        FIRETRUCK_STRIPE_COLOR = (255, 255, 255)  # White stripe
        AXLE_INDICATOR_COLOR = (255, 255, 255)  # Semi-transparent white
        AXLE_INDICATOR = 5  # pixels

        ppm = self.world.pixels_per_meter
        Lpx = math.floor(self.length * ppm)
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
        stripe_w = 3
        stripe_x = Lpx - stripe_w - 6
        pygame.draw.rect(
            surf,
            FIRETRUCK_STRIPE_COLOR,
            (stripe_x, 3, stripe_w, Wpx - 6),
            border_radius=2,
        )

        # Rotate for rendering
        rotated = pygame.transform.rotate(surf, math.degrees(self.pos.heading))

        # Calculate body center position from rear axle
        # a Body center is wheelbase/2 ahead of rear axle
        center_offset = self.wheelbase / 2.0
        body_center_x = self.pos.x + center_offset * math.cos(self.pos.heading)
        body_center_y = self.pos.y + center_offset * math.sin(self.pos.heading)

        # Convert to pixel coordinates and blit centered on the body
        px, py = self.world.world_to_pixel(body_center_x, body_center_y)
        rect = rotated.get_rect(center=(px, py))
        self.world.display.blit(rotated, rect.topleft)

        # Render rear axle indicator (semi-transparent white circle)
        axle_surface = pygame.Surface(
            (AXLE_INDICATOR * 2, AXLE_INDICATOR * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            axle_surface,
            AXLE_INDICATOR_COLOR,
            (AXLE_INDICATOR, AXLE_INDICATOR),
            AXLE_INDICATOR,
            width=2,
        )

        # Position at rear axle
        axle_px, axle_py = self.world.world_to_pixel(self.pos.x, self.pos.y)
        axle_rect = axle_surface.get_rect(center=(axle_px, axle_py))
        self.world.display.blit(axle_surface, axle_rect.topleft)

    def render_route(self, route: List[Pos], color: Tuple[int, int, int]):
        """Render a route as a polyline"""
        if len(route) < 2:
            return  # Need at least 2 points to draw a line

        pts = [self.world.world_to_pixel(pos.x, pos.y) for pos in route]
        pygame.draw.lines(self.world.display, color, False, pts, 2)

    def _create_roadmap_surface(self) -> pygame.Surface:
        """
        Pre-render the roadmap to a surface.
        This is called once during initialization since the roadmap is immutable.
        """
        if self.roadmap is None:
            return pygame.Surface((0, 0), pygame.SRCALPHA)

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

        logger.info("Roadmap surface pre-rendered")
        return surface

    def render_roadmap(self):
        """
        Render the pre-computed roadmap surface.
        Fast operation - just blits the cached surface.
        """
        if self.roadmap_surface is not None:
            self.world.display.blit(self.roadmap_surface, (0, 0))

    def render_planned_path(self):
        """
        Render the planned path with forward segments in one color and reverse in another.
        """
        if not self.planned_path_segments:
            return

        FORWARD_COLOR = (50, 220, 50)  # Green for forward
        REVERSE_COLOR = (220, 50, 220)  # Magenta for reverse
        LINE_WIDTH = 1

        # Group consecutive segments by direction for efficient rendering
        current_direction = None
        current_points = []

        for segment in self.planned_path_segments:
            # Start a new group if a direction changes
            if segment.is_forward != current_direction:
                # Draw the previous group
                if len(current_points) >= 2:
                    color = FORWARD_COLOR if current_direction else REVERSE_COLOR
                    pygame.draw.lines(
                        self.world.display, color, False, current_points, LINE_WIDTH
                    )

                # Start a new group
                current_direction = segment.is_forward
                current_points = [
                    self.world.world_to_pixel(segment.start.x, segment.start.y)
                ]

            # Add an end point to the current group
            current_points.append(
                self.world.world_to_pixel(segment.end.x, segment.end.y)
            )

        # Draw the last group
        if len(current_points) >= 2:
            color = FORWARD_COLOR if current_direction else REVERSE_COLOR
            pygame.draw.lines(
                self.world.display, color, False, current_points, LINE_WIDTH
            )

    def _render_coverage_circles(
        self, locations: List[Tuple[int, int]], draw_center_markers: bool = False
    ) -> None:
        """
        Render 10 m coverage circles around the given fine-grid locations.

        Args:
            locations: List of (fine_row, fine_col) tuples to draw circles around
            draw_center_markers: If True, draw a small marker at each location center
        """
        COVERAGE_FILL_COLOR = (100, 150, 255, 40)
        MARKER_COLOR = (100, 150, 255, 150)
        COVERAGE_RADIUS_PIXELS = int(
            COVERAGE_RADIUS_METERS * self.world.pixels_per_meter
        )
        MARKER_RADIUS = 3  # Small marker for fine-grid precision

        field = self.world.field

        # Create separate surfaces for circles and markers
        circle_surface = pygame.Surface(
            (self.world.display.get_width(), self.world.display.get_height()),
            pygame.SRCALPHA,
        )

        marker_surface = pygame.Surface(
            (self.world.display.get_width(), self.world.display.get_height()),
            pygame.SRCALPHA,
        )

        for fine_row, fine_col in locations:
            # Convert fine-grid to world coordinates
            x = (fine_col + 0.5) * field.collision_discretization
            y = (
                field.grid_dimensions * field.world.cell_size
                - (fine_row + 0.5) * field.collision_discretization
            )

            # Convert world to pixel coordinates
            center_px, center_py = self.world.world_to_pixel(x, y)

            # Draw a coverage circle on circle surface
            pygame.draw.circle(
                circle_surface,
                COVERAGE_FILL_COLOR,
                (center_px, center_py),
                COVERAGE_RADIUS_PIXELS,
            )

            # Draw center marker on marker surface (rendered on top)
            if draw_center_markers:
                pygame.draw.circle(
                    marker_surface,
                    MARKER_COLOR,
                    (center_px, center_py),
                    MARKER_RADIUS,
                )

        # Blit circles first, then markers on top
        self.world.display.blit(circle_surface, (0, 0))
        if draw_center_markers:
            self.world.display.blit(marker_surface, (0, 0))

    def render_coverage_radius(self):
        """Render the 10 m coverage radius around the firetruck when stationary"""
        # Only show radius when the truck has no planned path (is at rest)
        if self.planned_path_segments:
            return

        # Convert truck's coarse-grid location to fine-grid
        field = self.world.field
        cells_per_coarse = int(field.world.cell_size / field.collision_discretization)
        truck_row, truck_col = self.get_location()
        truck_fine_row = truck_row * cells_per_coarse + cells_per_coarse // 2
        truck_fine_col = truck_col * cells_per_coarse + cells_per_coarse // 2

        truck_fine_location = [(truck_fine_row, truck_fine_col)]
        self._render_coverage_circles(truck_fine_location, draw_center_markers=False)

    def render_poi_locations(self):
        """Render POI location markers with 10 m coverage circles"""
        self._render_coverage_circles(self.poi_locations, draw_center_markers=True)

    def render_top_priority_pois(self):
        """
        Render the top N priority POIs with marker size normalized by fire coverage.
        Marker size scales from small to cell_size based on relative fire count.
        """

        # Get ranked POIs
        ranked_pois = self.rank_pois_by_priority()

        if not ranked_pois:
            return

        # Take top N
        top_pois = ranked_pois[:TOP_CANDIDATES_COUNT]

        # Extract fire counts for normalization
        fire_counts = np.array([fire_count for _, _, fire_count in top_pois])

        min_count = fire_counts.min()
        max_count = fire_counts.max()

        # Define marker size range (in pixels)
        MIN_MARKER_RADIUS = 3  # pixels
        MAX_MARKER_RADIUS = int(
            self.world.cell_size * self.world.pixels_per_meter
        )  # cell size in pixels

        # Normalize fire counts to marker sizes
        if max_count > min_count:
            # Linear interpolation
            normalized = (fire_counts - min_count) / (max_count - min_count)
            marker_radii = (
                MIN_MARKER_RADIUS + normalized * (MAX_MARKER_RADIUS - MIN_MARKER_RADIUS)
            ).astype(int)
        else:
            # All have the same count-use mid-size
            marker_radii = np.full(
                len(top_pois), (MIN_MARKER_RADIUS + MAX_MARKER_RADIUS) // 2, dtype=int
            )

        # Color scheme for top POIs (red/orange gradient for urgency)
        MARKER_COLOR = (100, 150, 255, 150)

        field = self.world.field

        # Create a surface for rendering
        surface = pygame.Surface(
            (self.world.display.get_width(), self.world.display.get_height()),
            pygame.SRCALPHA,
        )

        # Render each POI with a normalized size
        for (poi_fine_loc, distance, fire_count), radius in zip(top_pois, marker_radii):
            fine_row, fine_col = poi_fine_loc

            # Convert fine-grid to world coordinates
            x = (fine_col + 0.5) * field.collision_discretization
            y = (
                field.grid_dimensions * field.world.cell_size
                - (fine_row + 0.5) * field.collision_discretization
            )

            # Convert to pixel coordinates
            px, py = self.world.world_to_pixel(x, y)

            # Draw a marker circle
            pygame.draw.circle(surface, MARKER_COLOR, (px, py), radius)

        # Blit to display
        self.world.display.blit(surface, (0, 0))


class VehicleController:
    """
    Simple trajectory executor for pre-planned kinematically possible paths.
    Just plays back the trajectory with velocity control - no steering correction needed.
    """

    def __init__(
        self,
        wheelbase: float,
        max_velocity_forward: float,
        max_velocity_reverse: float,
        max_acceleration: float,
        max_deceleration: float,
        min_turning_radius: float,
    ):
        self.wheelbase = wheelbase
        self.max_velocity_forward = max_velocity_forward
        self.max_velocity_reverse = max_velocity_reverse
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.min_turning_radius = min_turning_radius

        # Current state
        self.current_velocity = 0.0
        self.distance_along_path = 0.0  # Track progress along a path

    def stop(self):
        """Stop the vehicle"""
        self.current_velocity = 0.0

    def _calculate_target_velocity(
        self, remaining_distance: float, is_forward: bool
    ) -> float:
        """
        Calculate target velocity based on a direction and remaining distance.
        """
        # Base velocity for a direction
        max_vel = self.max_velocity_forward if is_forward else self.max_velocity_reverse

        # Deceleration zone near the end of a path
        stopping_distance = 5.0  # meters
        if remaining_distance < stopping_distance:
            min_vel = 0.5  # m/s
            target_vel = min_vel + (max_vel - min_vel) * (
                remaining_distance / stopping_distance
            )
            return max(min_vel, min(target_vel, max_vel))

        return max_vel

    def update(
        self,
        dt: float,
        current_pose: Pos,
        path_segments: List[PathSegment],
        segment_index: int,
    ) -> Tuple[Pos, int, bool]:
        """
        Execute the pre-planned path by interpolating along segments.

        Args:
            dt: Time step
            current_pose: Current vehicle pose (rear axle)
            path_segments: List of path segments to follow
            segment_index: Current segment index

        Returns:
            - new_pose: Updated vehicle pose
            - new_segment_index: Updated segment index
            - complete: True if a path following is complete
        """
        if not path_segments:
            return current_pose, 0, True

        if segment_index >= len(path_segments):
            return current_pose, segment_index, True

        # Get current segment
        current_segment = path_segments[segment_index]
        is_forward = current_segment.is_forward

        # Calculate the remaining path distance from the current segment onward
        remaining_distance = sum(
            seg.start.distance_to(seg.end) for seg in path_segments[segment_index:]
        )

        # Determine target velocity
        target_velocity = self._calculate_target_velocity(
            remaining_distance, is_forward
        )

        # Smooth velocity changes with acceleration limits
        velocity_error = target_velocity - abs(self.current_velocity)

        if velocity_error > 0:
            max_delta_v = self.max_acceleration * dt
            speed = abs(self.current_velocity) + min(velocity_error, max_delta_v)
        else:
            max_delta_v = self.max_deceleration * dt
            speed = abs(self.current_velocity) + max(velocity_error, -max_delta_v)

        # Apply direction
        self.current_velocity = speed if is_forward else -speed

        # Calculate distance to travel this timestep
        distance_to_travel = abs(self.current_velocity) * dt
        distance_remaining = distance_to_travel

        # Traverse segments until we've used up our distance budget
        new_segment_index = segment_index
        new_pose = current_pose

        while distance_remaining > 1e-6 and new_segment_index < len(path_segments):
            segment = path_segments[new_segment_index]

            # Distance from current pose to the segment end
            segment_remaining = new_pose.distance_to(segment.end)

            if distance_remaining >= segment_remaining:
                # Move to the end of this segment and continue to the next
                new_pose = segment.end
                distance_remaining -= segment_remaining
                new_segment_index += 1
            else:
                # Interpolate along the current segment
                if segment_remaining > 1e-6:
                    t = distance_remaining / segment_remaining

                    dx = segment.end.x - new_pose.x
                    dy = segment.end.y - new_pose.y
                    dh = segment.end.heading - new_pose.heading

                    # Normalize heading difference
                    while dh > math.pi:
                        dh -= 2 * math.pi
                    while dh < -math.pi:
                        dh += 2 * math.pi

                    new_pose = Pos(
                        x=new_pose.x + t * dx,
                        y=new_pose.y + t * dy,
                        heading=new_pose.heading + t * dh,
                    )

                break

        # Normalize the final heading
        final_heading = new_pose.heading
        while final_heading > math.pi:
            final_heading -= 2 * math.pi
        while final_heading < -math.pi:
            final_heading += 2 * math.pi

        new_pose = Pos(new_pose.x, new_pose.y, final_heading)

        # Check if a path is complete
        complete = new_segment_index >= len(path_segments)

        if not complete:
            # Also check if we're very close to the final goal and moving slowly
            final_distance = new_pose.distance_to(path_segments[-1].end)
            if final_distance < 0.1 and abs(self.current_velocity) < 0.5:
                complete = True

        return new_pose, new_segment_index, complete

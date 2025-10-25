# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Collision checker
# Gen AI usage: Claud for drafting the code and ideation; GPT-5 for caching the overlays

import math
import numpy as np
import pygame
from typing import List, Tuple

from vehicle import VehicleSpec
from world import World, Pos, world_to_grid

LOOSE_OVERLAY_COLOR = (255, 165, 0, 50)
TIGHT_OVERLAY_COLOR = (255, 0, 0, 50)
BOUNDARY_OVERLAY_COLOR = (0, 0, 0, 25)


class CollisionChecker:
    """
    The idea is to:
    1. Discretize the world at fine resolution (maybe 0.15m, 20 cells per obstacle cell)
    2. Store discretized obstacles and pre-compute inflated overlays for conservative/optimistic checking
    3. Pre-compute overlays for loose tolerance (half-length + margin): "worst case".
    4. Pre-compute for tight tolerance (half-width + margin): if the vehicle is aligned with the obstacle side.
    5. If the collision checker passes loose, we are 100% in the clear; instantly approve the path point.
    6. If the collision checker fails the loose, but passes tight, transition to the OBB checking (most expensive).
    7. If fails both, reject the path point.
    8. Provide O(1) lookup by (x, y) coordinates
    """

    def __init__(self, world: World, vehicle_spec: VehicleSpec):
        self.obstacles = world.obstacles
        self.world = world
        self.vehicle_spec = vehicle_spec
        self.discretization = 0.1  # 20 fine cells per coarse cell

        # Calculate fine grid dimensions
        self.fine_grid_size = int(np.ceil(self.world.grid_dimensions * self.world.cell_size / self.discretization))

        # Discretize obstacles to fine grid
        self.fine_obstacles = self._discretize_obstacles()

        # Produce the inflated obstacles
        loose_radius = self.vehicle_spec.length / 2 + self.vehicle_spec.safety_margin
        tight_radius = self.vehicle_spec.width / 2 + self.vehicle_spec.safety_margin
        self.loose_overlay = self._inflate_obstacles(loose_radius)
        self.tight_overlay = self._inflate_obstacles(tight_radius)

        # Produce boundary overlay
        self.boundary_overlay = self._create_boundary_overlay()

        # Prebuild overlay surfaces for fast rendering
        self._build_overlay_surfaces()

    def _create_boundary_overlay(self) -> np.ndarray:
        """
        Create an overlay marking the boundary zone where OBB checks are needed.
        This marks cells within buffer_distance of the world edges.

        Use half-diagonal (worst-case corner distance) plus safety margin.
        """
        vehicle_half_diagonal = math.hypot(self.vehicle_spec.length / 2, self.vehicle_spec.width / 2)
        buffer_distance = vehicle_half_diagonal + self.vehicle_spec.safety_margin

        overlay = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)
        buffer_cells = int(np.ceil(buffer_distance / self.discretization))

        overlay[:buffer_cells, :] = True  # Top edge
        overlay[-buffer_cells:, :] = True  # Bottom edge
        overlay[:, :buffer_cells] = True  # Left edge
        overlay[:, -buffer_cells:] = True  # Right edge

        return overlay

    def _world_to_fine_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates (x, y) to (row, col) indices in the fine grid."""
        col = int(x / self.discretization)
        row = int(y / self.discretization)
        return row, col

    def _discretize_obstacles(self) -> np.ndarray:
        """Convert coarse obstacle grid to fine-resolution grid."""
        fine_obstacles = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)

        # Each coarse cell corresponds to multiple fine cells
        cells_per_coarse = int(self.world.cell_size / self.discretization)

        # Fill fine grid cells
        for row in range(self.world.grid_dimensions):
            for col in range(self.world.grid_dimensions):
                if self.obstacles[row, col]:
                    fine_row_start = row * cells_per_coarse
                    fine_row_end = min((row + 1) * cells_per_coarse, self.fine_grid_size)
                    fine_col_start = col * cells_per_coarse
                    fine_col_end = min((col + 1) * cells_per_coarse, self.fine_grid_size)

                    fine_obstacles[fine_row_start:fine_row_end, fine_col_start:fine_col_end] = True

        return fine_obstacles

    @staticmethod
    def _create_circular_kernel(radius_fine_grid_cells: int) -> np.ndarray:
        """Create a circular structuring element for inflation of the cells by a specified radius."""
        size = 2 * radius_fine_grid_cells + 1  # Kernel size must be odd
        kernel = np.zeros((size, size), dtype=bool)
        center = radius_fine_grid_cells

        # Fill circle using distance formula
        for i in range(size):
            for j in range(size):
                distance_sq = (i - center) ** 2 + (j - center) ** 2
                if distance_sq <= radius_fine_grid_cells ** 2:
                    kernel[i, j] = True

        return kernel

    def _inflate_obstacles(self, inflation_radius: float) -> np.ndarray:
        """
        Inflate obstacles by the given radius using morphological dilation.
        This computes the Minkowski sum of obstacles with a circle of a given radius.
        """
        from scipy.ndimage import binary_dilation

        # radius to fine grid cells
        inflation_cells = int(np.ceil(inflation_radius / self.discretization))

        kernel = self._create_circular_kernel(inflation_cells)
        inflated = binary_dilation(self.fine_obstacles, structure=kernel)
        return inflated

    def _check_collision_at_xy(self, overlay, pos: Pos) -> bool:
        """Check if position collides with given overlay."""
        row, col = self._world_to_fine_grid(pos.x, pos.y)

        # Out of bounds check
        in_bounds = 0 <= row < self.fine_grid_size and 0 <= col < self.fine_grid_size
        if not in_bounds:
            return True  # Out of bounds is treated as a collision

        return overlay[row, col]

    def check_loose(self, pos: Pos) -> bool:
        return self._check_collision_at_xy(self.loose_overlay, pos)

    def check_tight(self, pos: Pos) -> bool:
        return self._check_collision_at_xy(self.tight_overlay, pos)

    def check_boundary(self, pos: Pos) -> bool:
        """Check if position is in the boundary danger zone"""
        return self._check_collision_at_xy(self.boundary_overlay, pos)

    @staticmethod
    def _get_obb_corners(pos: Pos, length: float, width: float) -> List[Tuple[float, float]]:
        """Compute the four corners of an oriented bounding box for a rectangular body."""
        cos_h = math.cos(pos.heading)
        sin_h = math.sin(pos.heading)

        half_length = length / 2
        half_width = width / 2

        # Corners in body frame: front-right, front-left, back-left, back-right
        local_corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]

        # Transform to world frame
        world_corners = []
        for lx, ly in local_corners:
            wx = pos.x + lx * cos_h - ly * sin_h
            wy = pos.y + lx * sin_h + ly * cos_h
            world_corners.append((wx, wy))

        return world_corners

    def check_obb_collision(self, pos: Pos, length: float = None, width: float = None) -> bool:
        """Check if an oriented bounding box at position collides with obstacles."""
        # Use vehicle dimensions as defaults
        if length is None:
            length = self.vehicle_spec.length
        if width is None:
            width = self.vehicle_spec.width
        
        corners = self._get_obb_corners(pos, length, width)

        # Check all four corners
        for cx, cy in corners:
            row, col = world_to_grid(cx, cy)
            if row < 0 or row >= self.world.grid_dimensions or col < 0 or col >= self.world.grid_dimensions:
                return True  # Out of bounds
            if self.obstacles[row, col]:
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
                if row < 0 or row >= self.world.grid_dimensions or col < 0 or col >= self.world.grid_dimensions:
                    return True
                if self.obstacles[row, col]:
                    return True

        return False

    def is_path_collision_free(self, path_points: List[Pos]) -> bool:
        """
        Four-tier collision checking for a path:
        1. Loose overlay (conservative) - instant approval if clear
        2. Boundary overlay - triggers OBB check near edges
        3. Tight overlay (optimistic) - instant rejection if collision detected
        4. OBB check - precise validation for ambiguous cases and boundary regions
        """
        obb_check_needed = []

        for pos in path_points:
            # Tier 1: Loose overlay check (conservative radius)
            if not self.check_loose(pos):
                # Definitely safe - no collision in the worst case
                # Tier 2. Check if near boundary (need OBB for rotated vehicle)
                if self.check_boundary(pos):
                    obb_check_needed.append(pos)
                continue

            # Tier 3: Tight overlay check (optimistic radius)
            if self.check_tight(pos):
                # Definitely collides - even best-case alignment hits the obstacle
                return False

            # Tier 4: Ambiguous - needs precise OBB check
            obb_check_needed.append(pos)

        # Perform OBB checks only for ambiguous points and boundary regions
        for pos in obb_check_needed:
            if self.check_obb_collision(pos):
                return False

        return True

    def _build_overlay_surfaces(self) -> None:
        """Pre-render overlay boolean grids into Pygame Surfaces for fast blitting."""
        ppm = self.world.pixels_per_meter
        ppf = self.discretization * ppm  # pixels per fine cell (can be float)
        width = int(self.fine_grid_size * ppf)
        height = int(self.fine_grid_size * ppf)

        def build(overlay: np.ndarray, color: tuple[int, int, int, int]) -> pygame.Surface:
            surf = pygame.Surface((width, height), pygame.SRCALPHA)
            # Draw only once; use integers for rects as pygame prefers ints.
            for row in range(self.fine_grid_size):
                y = int(row * ppf)
                for col in range(self.fine_grid_size):
                    if overlay[row, col]:
                        x = int(col * ppf)
                        rect = pygame.Rect(x, y, int(ppf), int(ppf))
                        pygame.draw.rect(surf, color, rect)
            return surf

        self._loose_surface = build(self.loose_overlay, LOOSE_OVERLAY_COLOR)
        self._tight_surface = build(self.tight_overlay, TIGHT_OVERLAY_COLOR)
        self._boundary_surface = build(self.boundary_overlay, BOUNDARY_OVERLAY_COLOR)

    def render_loose_overlay(self):
        # Fast path: just blit the prebuilt surface
        self.world.screen.blit(self._loose_surface, (0, 0))

    def render_tight_overlay(self):
        self.world.screen.blit(self._tight_surface, (0, 0))

    def render_boundary_overlay(self):
        self.world.screen.blit(self._boundary_surface, (0, 0))

    def is_path_collision_free_with_trailer(
        self, 
        truck_path: List[Pos], 
        trailer_path: List[Pos]
    ) -> bool:
        """
        Four-tier collision checking for paths with trailer.
        Checks both truck and trailer at each corresponding point.
        """
        if len(truck_path) != len(trailer_path):
            raise ValueError(f"Path length mismatch: truck={len(truck_path)}, trailer={len(trailer_path)}")
        
        obb_check_needed = []

        for i, truck_pos in enumerate(truck_path):
            trailer_pos = trailer_path[i]
            
            # Tier 1: Loose overlay check for both
            truck_loose = self.check_loose(truck_pos)
            trailer_loose = self.check_loose(trailer_pos)
            
            if not truck_loose and not trailer_loose:
                # Both clear - check if near boundary
                if self.check_boundary(truck_pos) or self.check_boundary(trailer_pos):
                    obb_check_needed.append(i)
                continue

            # Tier 3: Tight overlay check
            if self.check_tight(truck_pos) or self.check_tight(trailer_pos):
                # At least one definitely collides
                return False

            # Tier 4: Ambiguous - needs OBB
            obb_check_needed.append(i)

        # Perform OBB checks for ambiguous points
        for i in obb_check_needed:
            truck_pos = truck_path[i]
            trailer_pos = trailer_path[i]
            
            # Check truck with its dimensions
            if self.check_obb_collision(truck_pos, self.vehicle_spec.length, self.vehicle_spec.width):
                return False
            
            # Check trailer with its dimensions
            if self.check_obb_collision(
                trailer_pos, 
                self.vehicle_spec.trailer.length, 
                self.vehicle_spec.trailer.width
            ):
                return False

        return True

# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Collision checker
# Gen AI usage: Claud for drafting the code and ideation


import numpy as np
import pygame

from vehicle import VehicleSpec
from world import World, Pos
from planner import VEHICLE_SAFETY_MARGIN

LOOSE_OVERLAY_COLOR = (255, 0, 0, 100)
TIGHT_OVERLAY_COLOR = (255, 165, 0, 100)


class CollisionChecker:
    """
    The idea is to:
    1. Discretize the world at fine resolution (maybe 0.15m, 20 cells per obstacle cell)
    2. Store discretized obstacles and pre-compute inflated overlays for conservative/optimistic checking
    3. Precompute overlays for loose tolerance (half-length + margin): "worst case".
    4. Precompute for tight tolerance (half-width + margin): if the vehicle is aligned with the obstacle side.
    5. If the collision checker passes loose, we are 100% in the clear; instantly approve the path point.
    6. If the collision checker fails the loose, but passes tight, transition to the OBB checking (most expensive).
    7. If fails both, reject the path point.
    8. Provide O(1) lookup by (x, y) coordinates
    """

    def __init__(self, world: World, vehicle_spec: VehicleSpec):
        self.obstacles = world.obstacles
        self.discretization = 0.1  # this probably doesn't need to be configurable; mathching safety margin
        self.world = world
        self.vehicle_spec = vehicle_spec

        # calculate fine grid dimensions
        self.fine_grid_size = int(np.ceil(self.world.grid_dimensions * self.world.cell_size / self.discretization))

        # discretize obstacles to fine grid
        self.fine_obstacles = self._discretize_obstacles()

        # prodice the inflated obstacles
        loose_radius = self.vehicle_spec.length / 2 + VEHICLE_SAFETY_MARGIN
        tight_radius = self.vehicle_spec.width / 2 + VEHICLE_SAFETY_MARGIN
        self.loose_overlay = self._inflate_obstacles(loose_radius)
        self.tight_overlay = self._inflate_obstacles(tight_radius)

    def world_to_fine_grid(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert world coordinates (x, y) to (row, col) indices in the fine grid.
        """
        col = int(x / self.discretization)
        row = int(y / self.discretization)
        return row, col

    def _discretize_obstacles(self) -> np.ndarray:
        """
        Convert coarse obstacle grid to fine-resolution grid.
        """
        fine_obstacles = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)

        # each coarse cell corresponds to multiple fine cells
        cells_per_coarse = int(self.world.cell_size / self.discretization)

        # fill fine grid cells
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
        """
        Create a circular structuring element for inflation of the cells by a specified radius.
        """

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

    def _check_collision_at_xy(self, overlay, pos: Pos):
        # TODO: this begs to be unit-tested (as does everything else)
        row, col = self.world_to_fine_grid(pos.x, pos.y)

        # TODO: make sure out of bounds check is not redundant
        in_bounds = 0 <= row < self.fine_grid_size and 0 <= col < self.fine_grid_size
        if not in_bounds:
            return True  # out of bounds is treated as an obstacle

        return overlay[row, col]

    def check_loose(self, pos: Pos):
        return self._check_collision_at_xy(self.loose_overlay, pos)

    def check_tight(self, pos: Pos):
        return self._check_collision_at_xy(self.tight_overlay, pos)

    def _render_overlay(self, overlay, color):
        """
        Render the overlay onto the world screen.
        """

        # scaling
        pixels_per_fine_cell = self.discretization * self.world.pixels_per_meter

        # occupied cell surface
        overlay_surface = pygame.Surface(
            (self.fine_grid_size * pixels_per_fine_cell,
             self.fine_grid_size * pixels_per_fine_cell),
            pygame.SRCALPHA
        )

        # draw each occupied cell as a small rectangle
        for row in range(self.fine_grid_size):
            for col in range(self.fine_grid_size):
                if overlay[row, col]:
                    rect = pygame.Rect(
                        col * pixels_per_fine_cell,
                        row * pixels_per_fine_cell,
                        pixels_per_fine_cell,
                        pixels_per_fine_cell
                    )
                    pygame.draw.rect(overlay_surface, (color), rect)

        # Blit onto world screen
        self.world.screen.blit(overlay_surface, (0, 0))

    def render_loose_overlay(self):
        self._render_overlay(self.loose_overlay, LOOSE_OVERLAY_COLOR)

    def render_tight_overlay(self):
        self._render_overlay(self.tight_overlay, TIGHT_OVERLAY_COLOR)

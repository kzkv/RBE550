# Tom Kazakov
# RBE 550, Assignment 5, Wildfire

from enum import IntEnum
from typing import Tuple, TYPE_CHECKING

import numpy as np
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from world import World


class Cell(IntEnum):
    """Cell states"""
    EMPTY = 0
    OBSTACLE = 1
    BURNING = 2
    BURNED = 3


CELL_COLORS = {
    Cell.EMPTY: (255, 255, 255),
    Cell.OBSTACLE: (50, 50, 50),
    Cell.BURNING: (255, 75, 0),
    Cell.BURNED: (150, 150, 150)
}


class Field:
    """Game field manager"""

    def __init__(self, seed, grid_dimensions, obstacle_density, world: 'World'):
        self.rng = np.random.default_rng(seed)
        self.grid_dimensions = grid_dimensions

        # Tetromino shapes
        self.canonical_shapes = (
            [[1, 1, 1, 1]],  # straight (or I)
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 1, 1], [1, 1, 0]],  # S-skew
            [[1, 1, 0], [0, 1, 1]],  # Z-skew
            [[1, 1, 1], [0, 1, 0]]  # T
        )

        self.cells = self._generate_obstacles(grid_dimensions, obstacle_density)

        # Track ignition times for burning cells
        self.ignition_times = {}  # (row, col) -> ignition_time
        self.burn_duration = 10.0  # seconds
        self.spread_radius = 30.0  # meters

        self.world = world

    def rotate(self, shape: np.ndarray):
        return np.rot90(shape, self.rng.choice([-1, 0, 1, 2]))

    @staticmethod
    def get_color(cell: Cell) -> Tuple[int, int, int]:
        return CELL_COLORS[cell]

    def in_bounds(self, row: int, col: int) -> bool:
        """Check if the given row and column are within the field bounds"""
        return 0 <= row < self.grid_dimensions and 0 <= col < self.grid_dimensions

    def get_cell(self, row: int, col: int) -> Cell:
        return Cell(self.cells[row, col])

    def set_cell(self, row: int, col: int, cell: Cell):
        """A generic setter for cell state"""
        if not self.in_bounds(row, col):
            return
        self.cells[row, col] = cell

    def ignite(self, row: int, col: int) -> bool:
        """Set a cell on fire if it's an obstacle."""
        if not self.in_bounds(row, col):
            return False
        if self.get_cell(row, col) == Cell.OBSTACLE:
            self.set_cell(row, col, Cell.BURNING)
            self.ignition_times[(row, col)] = self.world.world_time
            return True
        return False  # TODO: refactor to remove the return feedback if not needed

    # Generate obstacle field
    def _generate_obstacles(self, grid_dimensions: int, obstacle_density: float) -> np.ndarray:
        field = np.zeros((grid_dimensions, grid_dimensions), dtype=int)

        cell_target = int(obstacle_density * grid_dimensions * grid_dimensions)
        cell_coverage = 0
        while cell_coverage < cell_target:
            # Calculate batch size - place at least one tetromino
            placements_count = max(1, int((cell_target - cell_coverage) / 4))

            for _ in range(placements_count):
                # Select a random shape
                idx = self.rng.integers(len(self.canonical_shapes))
                shape = np.array(self.canonical_shapes[idx])

                # Rotate the shape randomly
                mask = self.rotate(shape)

                # Place the shape at a random location
                mask_height = mask.shape[0]
                mask_width = mask.shape[1]
                row = self.rng.integers(0, grid_dimensions - mask_height + 1)
                col = self.rng.integers(0, grid_dimensions - mask_width + 1)

                # Apply mask directly (overlapping allowed)
                field[row:row + mask_height, col:col + mask_width] = mask

            # Update coverage statistics
            cell_coverage = np.count_nonzero(field)

        return field

    def update_burning_cells(self):
        """Update burning cells and spread fire after burn duration."""
        cells_to_burnout = []
        cells_to_ignite = []

        # Check each burning cell
        for (row, col), ignition_time in self.ignition_times.items():
            time_burning = self.world.world_time - ignition_time

            if time_burning >= self.burn_duration:
                # This cell has been burning for 10 seconds
                cells_to_burnout.append((row, col))

                # Find all obstacles within a 30 m radius
                cell_size = self.world.cell_size
                center_x = (col + 0.5) * cell_size
                center_y = (row + 0.5) * cell_size

                # Calculate radius in grid cells
                radius_cells = int(np.ceil(self.spread_radius / cell_size))

                # Pre-calculate bounds
                r_min = max(0, row - radius_cells)
                r_max = min(self.grid_dimensions, row + radius_cells + 1)
                c_min = max(0, col - radius_cells)
                c_max = min(self.grid_dimensions, col + radius_cells + 1)
                r_range = np.arange(r_min, r_max)
                c_range = np.arange(c_min, c_max)
                r_grid, c_grid = np.meshgrid(r_range, c_range, indexing='ij')

                target_x = (c_grid + 0.5) * cell_size
                target_y = (r_grid + 0.5) * cell_size
                distances = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)

                # Find cells within the radius that are obstacles
                mask = (distances <= self.spread_radius) & (self.cells[r_min:r_max, c_min:c_max] == Cell.OBSTACLE)
                burning_r, burning_c = np.where(mask)

                # Convert back to absolute coordinates
                cells_to_ignite.extend(zip(burning_r + r_min, burning_c + c_min))

        # Burn out cells that have been burning for 10 seconds
        for row, col in cells_to_burnout:
            self.set_cell(row, col, Cell.BURNED)
            del self.ignition_times[(row, col)]

        # Ignite new cells
        for row, col in cells_to_ignite:
            self.ignite(row, col)

    def get_cell_neighbors(self, row, col) -> list[tuple[int, int]]:
        """Get up to 8 neighbors of a cell (fewer if on the edge)"""
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(row + dr, col + dc) for dr, dc in neighbor_offsets if self.in_bounds(row + dr, col + dc)]

    def has_all_empty_neighbors(self, row, col):
        """Check if all in-bounds neighbors are empty"""
        cell_neighbors = self.get_cell_neighbors(row, col)
        return all(self.cells[n_row][n_col] == 0 for n_row, n_col in cell_neighbors)

    def ignite_neighbors(self, row, col) -> int:
        """Ignite all obstacle neighbors of a cell"""
        ignited = sum(self.ignite(n_row, n_col) for n_row, n_col in self.get_cell_neighbors(row, col))
        logger.info(f"Ignited {ignited} neighbors of cell ({row}, {col})")
        return ignited

    def tally_cells(self):
        # Count cells by status; exclude empty, always include other statuses counts (even if not present)
        tally = {cell: 0 for cell in Cell.__members__.values() if cell != Cell.EMPTY}
        tally.update({Cell(s): int(c) for s, c in zip(*np.unique(self.cells, return_counts=True)) if s != Cell.EMPTY})
        return tally

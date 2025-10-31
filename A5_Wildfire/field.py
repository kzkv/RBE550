# Tom Kazakov
# RBE 550, Assignment 5, Wildfire

from enum import IntEnum
from typing import Tuple, TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_dilation
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from world import World

SPREAD_DURATION = 10.0  # s, time to spread fire after ignition
SPREAD_RADIUS = 30.0  # m
BURNOUT_DURATION = 30.0  # s, time to burn out after ignition

"""
Fire propagation logic:
- Fires ignite on OBSTACLE cells, transitioning them to BURNING state
- Each burning cell spreads fire once after SPREAD_DURATION seconds 
  to all OBSTACLE cells within SPREAD_RADIUS meters
- Fires burn out after BURNOUT_DURATION seconds, transitioning to BURNED state
- The 'has_spread' set tracks which fires have already propagated 
  to prevent repeated spreading
"""


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
        self.has_spread = set()  # Set of (row, col) cells that have already spread fire

        self.world = world

        # Pre-computed spread mask
        self.spread_radius_cells = int(np.ceil(SPREAD_RADIUS / self.world.cell_size))
        self.spread_mask = self._precompute_spread_mask()

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
        if self.get_cell(row, col) == Cell.OBSTACLE:
            self.set_cell(row, col, Cell.BURNING)
            self.ignition_times[(row, col)] = self.world.world_time
            
            # Award 1 point to Wumpus for igniting
            self.world.wumpus_score += 1
            logger.debug(f"Wumpus scored 1 point for igniting ({row}, {col})")
            
            return True
        return False

    def suppress(self, row: int, col: int) -> bool:
        """Put out fire in a cell if it's burning."""
        if self.get_cell(row, col) == Cell.BURNING:
            self.set_cell(row, col, Cell.OBSTACLE)
            if (row, col) in self.ignition_times:
                del self.ignition_times[(row, col)]
            self.has_spread.discard((row, col))
            
            # Award 2 points to Firetruck for suppressing
            self.world.firetruck_score += 2
            logger.debug(f"Firetruck scored 2 points for suppressing ({row}, {col})")
            
            return True
        return False

    def _precompute_spread_mask(self):
        """Pre-compute a reusable boolean mask for fire spread radius."""
        cell_size = self.world.cell_size
        spread_radius_cells = int(np.ceil(SPREAD_RADIUS / cell_size))

        # Create a mask centered at (0, 0) in relative coordinates
        r_range = np.arange(-spread_radius_cells, spread_radius_cells + 1)
        c_range = np.arange(-spread_radius_cells, spread_radius_cells + 1)
        r_grid, c_grid = np.meshgrid(r_range, c_range, indexing='ij')

        # Calculate distances in world coordinates
        target_x = (c_grid + 0.5) * cell_size
        target_y = (r_grid + 0.5) * cell_size
        center_x = 0.5 * cell_size
        center_y = 0.5 * cell_size
        distances = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)

        # Return the boolean mask
        return distances <= SPREAD_RADIUS

    def create_location_mask(self, row: int, col: int, radius: int) -> np.ndarray:
        """Create a dilated boolean mask around a location with the given radius"""
        mask = np.zeros((self.grid_dimensions, self.grid_dimensions), dtype=bool)
        
        if not self.in_bounds(row, col):
            return mask
            
        mask[row, col] = True
        
        if radius > 0:
            structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            mask = binary_dilation(mask, structure=structure)
        
        return mask

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

    def initialize_location(self, preset_rows, preset_cols) -> tuple | None:
        """
        Initialize wumpus or truck in a random cell within the preset area where all 8 neighbors are empty.
        Args: ranges of rows and columns of the preset area
        Returns tuple (row, col) if a valid location is found, None otherwise
        """
        min_row, max_row = preset_rows
        min_col, max_col = preset_cols

        # Find all valid cells in the preset area
        valid_cells = []

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if self.world.field.has_all_empty_neighbors((row, col)):
                    valid_cells.append((row, col))

        if not valid_cells:
            return None

        # Choose a random valid cell
        chosen_location = self.world.field.rng.choice(valid_cells)
        return tuple(chosen_location)

    def update_burning_cells(self):
        """Update burning cells and spread fire after spread duration."""
        cells_to_burnout = []
        cells_to_spread = []

        # Check each burning cell
        for (row, col), ignition_time in self.ignition_times.items():
            time_burning = self.world.world_time - ignition_time

            # Check if this fire should spread (only once)
            if time_burning >= SPREAD_DURATION and (row, col) not in self.has_spread:
                cells_to_spread.append((row, col))
                self.has_spread.add((row, col))

            # Check if this fire should burn out
            if time_burning >= BURNOUT_DURATION:
                cells_to_burnout.append((row, col))

        # Spread fires
        cells_to_ignite = []
        radius = self.spread_radius_cells
        for row, col in cells_to_spread:
            r_slice = slice(max(0, row - radius), min(self.grid_dimensions, row + radius + 1))
            c_slice = slice(max(0, col - radius), min(self.grid_dimensions, col + radius + 1))

            mask_slice = self.spread_mask[
                radius - (row - r_slice.start):radius + (r_slice.stop - row),
                radius - (col - c_slice.start):radius + (c_slice.stop - col)
            ]

            obstacle_mask = mask_slice & (self.cells[r_slice, c_slice] == Cell.OBSTACLE)
            burning_r, burning_c = np.where(obstacle_mask)
            cells_to_ignite.extend(zip(burning_r + r_slice.start, burning_c + c_slice.start))

        # Ignite new cells
        for row, col in cells_to_ignite:
            self.ignite(row, col)

        # Burn out cells that have been burning long enough
        for row, col in cells_to_burnout:
            self.set_cell(row, col, Cell.BURNED)
            del self.ignition_times[(row, col)]
            self.has_spread.discard((row, col))
            
            # Award 1 additional point to Wumpus for burning out an obstacle
            self.world.wumpus_score += 1
            logger.debug(f"Wumpus scored 1 point for burning out ({row}, {col})")

    def get_cell_neighbors(self, location: tuple[int, int]) -> list[tuple[int, int]]:
        """Get up to 8 neighbors of a cell (fewer if on the edge)"""
        row, col = location
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(row + dr, col + dc) for dr, dc in neighbor_offsets if self.in_bounds(row + dr, col + dc)]

    def has_all_empty_neighbors(self, location: tuple[int, int]):
        """Check if all in-bounds neighbors are empty"""
        cell_neighbors = self.get_cell_neighbors(location)
        return all(self.cells[n_row][n_col] == 0 for n_row, n_col in cell_neighbors)

    def ignite_neighbors(self, location: tuple[int, int]) -> int:
        """Ignite all obstacle neighbors of a cell"""
        ignited = sum(self.ignite(n_row, n_col) for n_row, n_col in self.get_cell_neighbors(location))
        return ignited

    def ignite_random_neighbor(self, location: tuple[int, int]) -> bool:
        """Ignite a random obstacle neighbor of a cell"""
        neighbors = self.get_cell_neighbors(location)
        obstacle_neighbors = [(row, col) for row, col in neighbors if self.get_cell(row, col) == Cell.OBSTACLE]
        return self.ignite(*self.rng.choice(obstacle_neighbors)) if obstacle_neighbors else False

    def suppress_neighbors(self, location: tuple[int, int]) -> int:
        suppressed = sum(self.suppress(n_row, n_col) for n_row, n_col in self.get_cell_neighbors(location))
        return suppressed

    def tally_cells(self):
        # Count cells by status; exclude empty, always include other statuses counts (even if not present)
        tally = {cell: 0 for cell in Cell.__members__.values() if cell != Cell.EMPTY}
        tally.update({Cell(s): int(c) for s, c in zip(*np.unique(self.cells, return_counts=True)) if s != Cell.EMPTY})
        return tally

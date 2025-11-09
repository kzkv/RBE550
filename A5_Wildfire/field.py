# Tom Kazakov
# RBE 550, Assignment 5, Wildfire

import logging
import math
import pygame
from enum import IntEnum
import numpy as np
from scipy.ndimage import binary_dilation
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from world import World, Pos

logger = logging.getLogger(__name__)

# Discretization parameter for collision checking
COLLISION_DISCRETIZATION = 0.2  # meters per fine grid cell

SPREAD_DURATION = 10.0  # s, time to spread fire after ignition
SPREAD_RADIUS = 30.0  # m
BURNOUT_DURATION = (
    720.0  # s, time to burn out after ignition; give Firetruck a chance to score points
)

"""
Field generation and environment model.

The Field class represents the static world layout (obstacles) and the dynamic fire states.
It operates in both coarse and fine resolutions to support different planning modules.

Responsibilities:
    - Generate randomized obstacle distributions using seeded tetromino shapes.
    - Initialize and manage the fine-grid mask for collision checking.
    - Track cell states: EMPTY, OBSTACLE, BURNING, BURNED.
    - Provide fire spread, burnout, and suppression mechanics.

Fire dynamics:
    1. Burning cells spread fire to nearby obstacles within SPREAD_RADIUS.
    2. Spread occurs once per cell, tracked via has_spread mask.
    3. Cells transition from BURNING → BURNED after BURNOUT_DURATION.
    4. Suppression immediately sets a cell to OBSTACLE.

Collision overlays:
    - Obstacles are inflated by a circular structuring element to approximate
      the Minkowski sum of the vehicle footprint and safety margin.
    - The inflation radius is computed from vehicle dimensions and stored as a constant.
    - Boolean overlays enable O(1) collision checks in planners.

Rendering:
    - The fine-grid overlay is pre-rendered as a pygame Surface for efficient drawing.

Field methods expose fast world-to-grid and grid-to-world conversions
for consistent handling across Wumpus, Firetruck, and World modules.
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
    Cell.BURNED: (150, 150, 150),
}

# Collision overlay rendering color
COLLISION_OVERLAY_COLOR = (0, 0, 0, 25)


class Field:
    """Game field manager"""

    def __init__(self, seed, grid_dimensions, obstacle_density, world: "World"):
        self.rng = np.random.default_rng(seed)
        self.grid_dimensions = grid_dimensions

        # Tetromino shapes
        self.canonical_shapes = (
            [[1, 1, 1, 1]],  # straight (or I)
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 1, 1], [1, 1, 0]],  # S-skew
            [[1, 1, 0], [0, 1, 1]],  # Z-skew
            [[1, 1, 1], [0, 1, 0]],  # T
        )

        self.cells = self._generate_obstacles(grid_dimensions, obstacle_density)

        # Track ignition times for burning cells
        self.ignition_times = {}  # (row, col) -> ignition_time
        self.has_spread = set()  # Set of (row, col) cells that have already spread fire

        self.world = world

        # Pre-computed spread mask
        self.spread_radius_cells = int(np.ceil(SPREAD_RADIUS / self.world.cell_size))
        self.spread_mask = self._precompute_spread_mask()

        # Initialize collision checking overlay
        self._init_collision_overlay()

    def _init_collision_overlay(self):
        """Initialize the fine-grid discretized obstacle overlay for fast collision checking."""
        # Calculate fine grid dimensions
        self.collision_discretization = COLLISION_DISCRETIZATION
        self.fine_grid_size = int(
            np.ceil(
                self.grid_dimensions
                * self.world.cell_size
                / self.collision_discretization
            )
        )

        # Discretize obstacles to fine grid
        self.fine_obstacles = self._discretize_obstacles_fine()

        # Create inflated obstacle overlay using half-diagonal of firetruck
        # TODO: break circular dependency and use encapsulated firetruck object
        firetruck_length = 4.9
        firetruck_width = 2.2
        firetruck_wheelbase = 3.0
        firetruck_half_diagonal = math.hypot(
            firetruck_length / 2 + firetruck_wheelbase / 2, firetruck_width / 2
        )

        # Inflate by half-diagonal (worst-case corner distance)
        inflation_radius = firetruck_half_diagonal
        inflated_obstacles = self._inflate_obstacles_fine(inflation_radius)

        # Create a border overlay with the same inflation radius
        border_overlay = self._create_border_overlay(inflation_radius)

        # Combine into a single collision overlay (obstacles OR borders)
        self.collision_overlay = inflated_obstacles | border_overlay

        # Pre-render overlay surface for fast rendering
        self._overlay_surface = self._prerender_overlay_surface()

    def _create_border_overlay(self, buffer_distance: float) -> np.ndarray:
        """
        Create an overlay marking the border zone as collision areas.
        Marks cells within buffer_distance of the world boundaries.

        Args:
            buffer_distance: Distance in meters from world edges to mark as collision zone

        Returns:
            Boolean array marking border collision zones
        """
        overlay = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)

        # Convert buffer distance to fine grid cells
        buffer_cells = int(np.ceil(buffer_distance / self.collision_discretization))

        # Mark cells near each edge
        overlay[:buffer_cells, :] = True  # Top edge
        overlay[-buffer_cells:, :] = True  # Bottom edge
        overlay[:, :buffer_cells] = True  # Left edge
        overlay[:, -buffer_cells:] = True  # Right edge

        return overlay

    def _prerender_overlay_surface(self) -> pygame.Surface:
        """Pre-render collision overlay to a pygame Surface for fast blitting."""
        ppm = self.world.pixels_per_meter
        ppf = self.collision_discretization * ppm  # pixels per fine cell
        width = int(self.fine_grid_size * ppf)
        height = int(self.fine_grid_size * ppf)

        surf = pygame.Surface((width, height), pygame.SRCALPHA)

        # Vectorized approach: find all collision cells at once
        collision_rows, collision_cols = np.where(self.collision_overlay)

        # Batch render collision cells
        ppf_int = int(ppf)
        for row, col in zip(collision_rows, collision_cols):
            x = int(col * ppf)
            y = int(row * ppf)
            rect = pygame.Rect(x, y, ppf_int, ppf_int)
            pygame.draw.rect(surf, COLLISION_OVERLAY_COLOR, rect)

        return surf

    def render_collision_overlay(self) -> None:
        """Render the collision overlay for visualization."""
        self.world.display.blit(self._overlay_surface, (0, 0))

    def _discretize_obstacles_fine(self) -> np.ndarray:
        """Convert coarse obstacle grid to fine-resolution grid."""
        fine_obstacles = np.zeros(
            (self.fine_grid_size, self.fine_grid_size), dtype=bool
        )

        # Each coarse cell corresponds to multiple fine cells
        cells_per_coarse = int(self.world.cell_size / self.collision_discretization)

        # Fill fine grid cells
        for row in range(self.grid_dimensions):
            for col in range(self.grid_dimensions):
                if self.cells[row, col] == Cell.OBSTACLE:
                    fine_row_start = row * cells_per_coarse
                    fine_row_end = min(
                        (row + 1) * cells_per_coarse, self.fine_grid_size
                    )
                    fine_col_start = col * cells_per_coarse
                    fine_col_end = min(
                        (col + 1) * cells_per_coarse, self.fine_grid_size
                    )

                    fine_obstacles[
                        fine_row_start:fine_row_end, fine_col_start:fine_col_end
                    ] = True

        return fine_obstacles

    @staticmethod
    def _create_circular_kernel(radius_fine_grid_cells: int) -> np.ndarray:
        """Create a circular structuring element for inflation."""
        size = 2 * radius_fine_grid_cells + 1
        kernel = np.zeros((size, size), dtype=bool)
        center = radius_fine_grid_cells

        for i in range(size):
            for j in range(size):
                distance_sq = (i - center) ** 2 + (j - center) ** 2
                if distance_sq <= radius_fine_grid_cells**2:
                    kernel[i, j] = True

        return kernel

    def _inflate_obstacles_fine(self, inflation_radius: float) -> np.ndarray:
        """Inflate obstacles by the given radius using morphological dilation."""
        # Convert radius to fine grid cells
        inflation_cells = int(np.ceil(inflation_radius / self.collision_discretization))

        kernel = self._create_circular_kernel(inflation_cells)
        inflated = binary_dilation(self.fine_obstacles, structure=kernel)
        return inflated

    def world_to_fine_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates (x, y) to (row, col) indices in the fine grid."""
        # Convert from Cartesian (y up) to grid coordinates (y down)
        # Match the coordinate system used in world_to_grid
        col = int(x / self.collision_discretization)

        # Flip Y: high y (Cartesian) -> low row (grid)
        # Total world height is grid_dimensions * cell_size
        world_height = self.grid_dimensions * self.world.cell_size
        row = int((world_height - y) / self.collision_discretization)

        return row, col

    def check_collision_at_pos(self, pos: "Pos") -> bool:
        """
        Check if a position collides with obstacles (using inflated overlay).
        Returns True if a collision is detected, False if clear.
        Uses pre-computed fine-grid overlay for O(1) lookup.
        """
        row, col = self.world_to_fine_grid(pos.x, pos.y)

        # Out of bounds check
        in_bounds = 0 <= row < self.fine_grid_size and 0 <= col < self.fine_grid_size
        if not in_bounds:
            return True  # Out of bounds is treated as a collision

        return self.collision_overlay[row, col]

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
        r_grid, c_grid = np.meshgrid(r_range, c_range, indexing="ij")

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
    def _generate_obstacles(
        self, grid_dimensions: int, obstacle_density: float
    ) -> np.ndarray:
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
                field[row : row + mask_height, col : col + mask_width] = mask

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
            r_slice = slice(
                max(0, row - radius), min(self.grid_dimensions, row + radius + 1)
            )
            c_slice = slice(
                max(0, col - radius), min(self.grid_dimensions, col + radius + 1)
            )

            mask_slice = self.spread_mask[
                radius - (row - r_slice.start) : radius + (r_slice.stop - row),
                radius - (col - c_slice.start) : radius + (c_slice.stop - col),
            ]

            obstacle_mask = mask_slice & (self.cells[r_slice, c_slice] == Cell.OBSTACLE)
            burning_r, burning_c = np.where(obstacle_mask)
            cells_to_ignite.extend(
                zip(burning_r + r_slice.start, burning_c + c_slice.start)
            )

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
        neighbor_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        return [
            (row + dr, col + dc)
            for dr, dc in neighbor_offsets
            if self.in_bounds(row + dr, col + dc)
        ]

    def has_all_empty_neighbors(self, location: tuple[int, int]):
        """Check if all in-bounds neighbors are empty"""
        cell_neighbors = self.get_cell_neighbors(location)
        return all(self.cells[n_row][n_col] == 0 for n_row, n_col in cell_neighbors)

    def ignite_neighbors(self, location: tuple[int, int]) -> int:
        """Ignite all obstacle neighbors of a cell"""
        ignited = sum(
            self.ignite(n_row, n_col)
            for n_row, n_col in self.get_cell_neighbors(location)
        )
        return ignited

    def ignite_random_neighbor(self, location: tuple[int, int]) -> bool:
        """Ignite a random obstacle neighbor of a cell"""
        neighbors = self.get_cell_neighbors(location)
        obstacle_neighbors = [
            (row, col)
            for row, col in neighbors
            if self.get_cell(row, col) == Cell.OBSTACLE
        ]
        return (
            self.ignite(*self.rng.choice(obstacle_neighbors))
            if obstacle_neighbors
            else False
        )

    def suppress_neighbors(self, location: tuple[int, int]) -> int:
        suppressed = sum(
            self.suppress(n_row, n_col)
            for n_row, n_col in self.get_cell_neighbors(location)
        )
        return suppressed

    def tally_cells(self):
        # Count cells by status; exclude empty, always include other statuses counts (even if not present)
        tally = {cell: 0 for cell in Cell.__members__.values() if cell != Cell.EMPTY}
        tally.update(
            {
                Cell(s): int(c)
                for s, c in zip(*np.unique(self.cells, return_counts=True))
                if s != Cell.EMPTY
            }
        )
        return tally

    def compute_poi_coverage_heatmap(self, coverage_radius_meters: float) -> np.ndarray:
        """
        Compute a fine-grid heatmap where each passable cell's value represents
        how many obstacles it can cover within the given radius.

        Args:
            coverage_radius_meters: Coverage radius in meters (e.g., 10 m for firetruck)

        Returns:
            Fine-grid array where each cell value = the number of obstacles coverable from that position
        """
        logger.info(
            f"Computing POI coverage heatmap with {coverage_radius_meters}m radius..."
        )

        # Initialize heatmap (fine grid resolution)
        heatmap = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=np.int32)

        # Convert coverage radius to fine grid cells
        coverage_radius_fine = int(
            np.floor(coverage_radius_meters / self.collision_discretization)
        )

        # Create circular kernel for coverage area
        kernel = self._create_circular_kernel(coverage_radius_fine)

        # For each obstacle cell in the coarse grid, inflate it on the fine grid
        cells_per_coarse = int(self.world.cell_size / self.collision_discretization)

        obstacle_rows, obstacle_cols = np.where(self.cells == Cell.OBSTACLE)

        for obs_row, obs_col in zip(obstacle_rows, obstacle_cols):
            # Convert an obstacle cell to a fine grid center
            fine_row_center = obs_row * cells_per_coarse + cells_per_coarse // 2
            fine_col_center = obs_col * cells_per_coarse + cells_per_coarse // 2

            # Apply kernel centered at this obstacle (increment all reachable fine cells)
            kernel_radius = coverage_radius_fine
            row_start = max(0, fine_row_center - kernel_radius)
            row_end = min(self.fine_grid_size, fine_row_center + kernel_radius + 1)
            col_start = max(0, fine_col_center - kernel_radius)
            col_end = min(self.fine_grid_size, fine_col_center + kernel_radius + 1)

            # Extract the valid portion of the kernel
            kernel_row_start = kernel_radius - (fine_row_center - row_start)
            kernel_row_end = kernel_row_start + (row_end - row_start)
            kernel_col_start = kernel_radius - (fine_col_center - col_start)
            kernel_col_end = kernel_col_start + (col_end - col_start)

            kernel_slice = kernel[
                kernel_row_start:kernel_row_end, kernel_col_start:kernel_col_end
            ]

            # Increment heatmap where kernel is True
            heatmap[row_start:row_end, col_start:col_end] += kernel_slice.astype(
                np.int32
            )

        # Mask out impassable areas (collision overlay)
        heatmap[self.collision_overlay] = 0

        logger.info(
            f"Heatmap computed: max coverage = {heatmap.max()}, "
            f"passable cells = {np.count_nonzero(heatmap)}"
        )

        return heatmap

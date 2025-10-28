# Tom Kazakov
# RBE 550
# Assignment 5, Wildfire

from dataclasses import dataclass
from typing import Tuple, List
from enum import IntEnum

import numpy as np
import pygame
import math

# Fundamental world constants
GRID_DIMENSIONS = 50
CELL_SIZE = 5.0  # 5 meters per cell
PIXELS_PER_METER = 7
OBSTACLE_DENSITY = 0.1


@dataclass(frozen=True)
class Pos:
    """Position in the world: x, y, heading"""
    x: float  # m
    y: float  # m
    heading: float  # rad, 0 is along x-axis, CCW is positive

    def distance_to(self, other: 'Pos') -> float:
        """Euclidean distance to another position"""
        return math.hypot(self.x - other.x, self.y - other.y)

    def heading_error_to(self, other: 'Pos') -> float:
        """Heading error to another position (wrapped to [-pi, pi])"""
        error = abs(self.heading - other.heading)
        if error > math.pi:
            error = 2 * math.pi - error
        return error

    def to_xy_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple for rendering"""
        return (self.x, self.y)


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

    # Generate obstacles field
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


class World:
    """World state and rendering"""

    def __init__(self, seed, time_speed: float):
        self.grid_dimensions = GRID_DIMENSIONS
        self.cell_size = CELL_SIZE
        self.pixels_per_meter = PIXELS_PER_METER
        self.cell_dimensions = CELL_SIZE * self.pixels_per_meter
        self.field_dimensions = self.grid_dimensions * self.cell_dimensions
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.hud_padding = 10
        self.hud_height = self.font.get_height() + self.hud_padding * 2

        self.clock = pygame.time.Clock()
        self.time_speed = time_speed
        self.world_time = 0.0  # World time in seconds

        self.field = Field(seed, self.grid_dimensions, OBSTACLE_DENSITY, world=self)

        self.display = pygame.display.set_mode((self.field_dimensions, self.field_dimensions + self.hud_height))

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Returns the (x, y) center of the cell in meters from the upper left corner at the given (row, col)"""
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y  # meters

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Returns the (row, col) of the cell containing the given point coordinates (x, y) in meters"""
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Mouse pixel -> world coordinates (in meters)"""
        return px / self.pixels_per_meter, py / self.pixels_per_meter

    def get_filtered_cells_coordinates(self, cell: Cell) -> list[tuple[int, int]]:
        """Return a list of (row, col) coordinates for all cells matching the given type."""
        rows, cols = np.where(self.field.cells == cell)
        return list(zip(rows, cols))

    def update(self):
        """Update world state with delta time adjusted for the multiplier"""
        dt = self.clock.tick(60) / 1000.0
        self.world_time += dt * self.time_speed

    # Rendering
    def clear(self):
        CELL_BG_COLOR = (255, 255, 255)
        self.display.fill(CELL_BG_COLOR)

    def render_grid(self):
        CELL_GRID_COLOR = (230, 230, 230)

        s = self.cell_dimensions
        for y in range(self.grid_dimensions):
            for x in range(self.grid_dimensions):
                pygame.draw.rect(
                    self.display,
                    CELL_GRID_COLOR,
                    pygame.Rect(x * s, y * s, s, s),
                    width=1,
                )

    def render_field(self):
        d = self.cell_dimensions
        for row in range(self.grid_dimensions):
            for col in range(self.grid_dimensions):
                cell = self.field.get_cell(row, col)
                if cell:
                    r = pygame.Rect(col * d, row * d, d, d)
                    pygame.draw.rect(self.display, self.field.get_color(cell), r)

    def render_spread(self):
        """Render the radius around burning cells."""
        surface = pygame.Surface((self.display.get_width(), self.display.get_height()), pygame.SRCALPHA)

        for (row, col) in self.get_filtered_cells_coordinates(Cell.BURNING):
            center_x = int((col + 0.5) * self.cell_dimensions)
            center_y = int((row + 0.5) * self.cell_dimensions)
            radius_pixels = int(self.field.spread_radius * self.pixels_per_meter)
            color = (*self.field.get_color(Cell.BURNING), 50)
            pygame.draw.circle(surface, color, (center_x, center_y), radius_pixels, radius_pixels)

        self.display.blit(surface, (0, 0))

    def render_hud(self, wumpus_pos: Pos = None, firetruck_pos: Pos = None, message: str = ""):
        HUD_BG_COLOR = (0, 0, 0)
        HUD_FONT_COLOR = (200, 200, 200)

        mx, my = pygame.mouse.get_pos()
        x, y = self.pixel_to_world(mx, my)
        row, col = self.world_to_grid(x, y)

        in_bounds = (0 <= row < self.grid_dimensions and 0 <= col < self.grid_dimensions)
        hud_rect = pygame.Rect(0, self.field_dimensions, self.field_dimensions, self.hud_height)
        pygame.draw.rect(self.display, HUD_BG_COLOR, hud_rect)

        # Format world time
        time_str = f"{self.world_time:4.0f}s"
        text = f"{time_str}    {message}"

        img = self.font.render(text, True, HUD_FONT_COLOR)

        if in_bounds:
            self.display.blit(img, (hud_rect.x + self.hud_padding, hud_rect.y + self.hud_padding))

    def render_route(self, route: List[Pos], color: Tuple[int, int, int]):
        if len(route) < 2:
            return  # Need at least 2 points to draw a line
        pts = [(int(pos.x * self.pixels_per_meter), int(pos.y * self.pixels_per_meter)) for pos in route]
        pygame.draw.lines(self.display, color, False, pts, 2)

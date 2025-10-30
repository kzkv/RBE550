# Tom Kazakov
# RBE 550, Assignment 5, Wildfire

import math
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wumpus import Wumpus
    from firetruck import Firetruck

import numpy as np
import pygame

from field import Field, Cell, SPREAD_RADIUS

# Fundamental world constants
GRID_DIMENSIONS = 50
CELL_SIZE = 5.0  # 5 meters per cell
PIXELS_PER_METER = 7
OBSTACLE_DENSITY = 0.1


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
        self.wumpus = None
        self.firetruck = None

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

    def get_filtered_cells_coordinates(self, cell: 'Cell') -> list[tuple[int, int]]:
        """Return a list of (row, col) coordinates for all cells matching the given type."""
        rows, cols = np.where(self.field.cells == cell)
        return list(zip(rows, cols))

    def set_wumpus(self, wumpus: 'Wumpus'):
        self.wumpus = wumpus

    def set_firetruck(self, firetruck: 'Firetruck'):
        self.firetruck = firetruck

    def update(self) -> float:
        """Update world state with delta time adjusted for the multiplier"""
        dt = self.clock.tick(60) / 1000.0
        self.world_time += dt * self.time_speed
        return dt * self.time_speed  # return time delta in world time

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
        """Render the radius around burning cells that haven't spread yet."""
        surface = pygame.Surface((self.display.get_width(), self.display.get_height()), pygame.SRCALPHA)

        for (row, col) in self.get_filtered_cells_coordinates(Cell.BURNING):
            # Only render radius for fires that haven't spread yet
            if (row, col) not in self.field.has_spread:
                center_x = int((col + 0.5) * self.cell_dimensions)
                center_y = int((row + 0.5) * self.cell_dimensions)
                radius_pixels = int(SPREAD_RADIUS * self.pixels_per_meter)
                color = (*self.field.get_color(Cell.BURNING), 50)
                pygame.draw.circle(surface, color, (center_x, center_y), radius_pixels, radius_pixels)

        self.display.blit(surface, (0, 0))

    def render_hud(self, message: str = ""):
        HUD_BG_COLOR = (0, 0, 0)
        HUD_FONT_COLOR = (200, 200, 200)

        hud_rect = pygame.Rect(0, self.field_dimensions, self.field_dimensions, self.hud_height)
        pygame.draw.rect(self.display, HUD_BG_COLOR, hud_rect)

        # Format world time
        time_str = f"{self.world_time:4.0f}s"

        # Cell states tally
        tally_str = "  ".join(f"{cell.name}: {count:<3d}" for cell, count in self.field.tally_cells().items())

        # Locations
        wumpus_location = self.wumpus.get_location()
        firetruck_location = self.firetruck.get_location()
        locations_str = f"Wumpus: {wumpus_location}   Firetruck: {firetruck_location}"

        text = f"{time_str}    {tally_str}    {locations_str}     {message}"

        img = self.font.render(text, True, HUD_FONT_COLOR)
        self.display.blit(img, (hud_rect.x + self.hud_padding, hud_rect.y + self.hud_padding))


@dataclass(frozen=True)
class Pos:
    """Position in the world: x, y, heading"""
    x: float  # m
    y: float  # m
    heading: float  # rad, 0 is along the x-axis, CCW is positive

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
        return self.x, self.y

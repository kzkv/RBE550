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
PIXELS_PER_METER = 5
OBSTACLE_DENSITY = 0.1

"""
World controller and simulation orchestrator.

The World class manages global simulation state, coordinates both agents (Wumpus and Firetruck),
and maintains a consistent notion of world time independent from real time.

Core responsibilities:
    1. Maintain the global clock and pause/resume logic.
    2. Step the physics of fire spread, burnout, and suppression.
    3. Update Wumpus and Firetruck agents sequentially each frame.
    4. Aggregate scores and expose summary statistics.
    5. Render the composite scene: field layers, agent overlays, HUD.

Simulation sequencing:
    1. Pause world time before any compute-heavy planning phase (A* for Wumpus and Firetruck).
    2. Allow agents to replan paths or select new goals.
    3. Resume world time and advance the simulation tick.
    4. Update fire states and apply scoring changes.
    5. Render the current frame to screen.

Scoring:
    +1 point for each ignition caused by the Wumpus.
    +1 point for each burned out cell awarder to the Wumpus.
    +2 points for each burning cell suppressed by the Firetruck.
    Burnout and suppression transitions are handled directly by the Field class.

HUD and telemetry:
    - Displays world time, agent scores, and active simulation state.

Design notes:
    - Deterministic simulation: world time pauses predictable renders (no "teleports") and clean time.
    - Clean separation between compute (planners) and render (pygame loop).
"""


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
        self.dt_world = 0.0  # Delta time in seconds
        self.is_paused = False  # Pause simulation during heavy computation

        self.field = Field(seed, self.grid_dimensions, OBSTACLE_DENSITY, world=self)
        self.wumpus = None
        self.firetruck = None

        # Scoring
        self.wumpus_score = 0
        self.firetruck_score = 0

        self.display = pygame.display.set_mode(
            (self.field_dimensions, self.field_dimensions + self.hud_height)
        )

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Returns the (x, y) center of the cell in meters from the upper left corner at the given (row, col)"""
        x = (col + 0.5) * self.cell_size

        # Flip row to Cartesian: row 0 (top) -> highest y
        y = (self.grid_dimensions - row - 0.5) * self.cell_size
        return x, y  # meters

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """
        Returns the (row, col) of the cell containing the given Cartesian coordinates.
        Cartesian: y=0 is at the bottom, y increases upward
        Grid: row 0 is at the top, row increases downward
        """
        col = int(x // self.cell_size)

        # Flip y to grid row: high y -> low row
        row = int((self.grid_dimensions * self.cell_size - y) // self.cell_size)
        return row, col

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates (Cartesian).
        Pygame Y-axis points down, but we want Cartesian Y pointing up.
        """
        x = px / self.pixels_per_meter

        # Flip Y: pygame y=0 is at the top, Cartesian y=0 is at bottom of field
        # Note: field starts at pygame y=0, so we don't need to subtract hud_height
        y = (self.field_dimensions - py) / self.pixels_per_meter
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert world coordinates (Cartesian) to pixel coordinates.
        Flip Y-axis for pygame rendering.
        """
        px = int(x * self.pixels_per_meter)

        # Flip Y: Cartesian y=0 (bottom) -> pygame y=field_dimensions (bottom of field)
        py = int(self.field_dimensions - y * self.pixels_per_meter)
        return px, py

    def get_filtered_cells_coordinates(self, cell: "Cell") -> list[tuple[int, int]]:
        """Return a list of (row, col) coordinates for all cells matching the given type."""
        rows, cols = np.where(self.field.cells == cell)
        return list(zip(rows, cols))

    def set_wumpus(self, wumpus: "Wumpus"):
        self.wumpus = wumpus

    def set_firetruck(self, firetruck: "Firetruck"):
        self.firetruck = firetruck

    def update(self):
        """Update world state with delta time adjusted for the multiplier"""
        dt_real = self.clock.tick(60) / 1000.0  # seconds (60 FPS)

        if self.is_paused:
            self.dt_world = 0.0
        else:
            # Cap dt to prevent huge jumps after resuming from pause
            # This prevents accumulated real-time from causing simulation glitches
            MAX_DT = 0.1  # Cap at 100 ms of real time (6 FPS minimum)
            dt_real = min(dt_real, MAX_DT)

            dt_world = dt_real * self.time_speed
            self.world_time += dt_world
            self.dt_world = dt_world  # Store for use by other components

    def pause_simulation(self):
        """Pause simulation time (during heavy computation)"""
        self.is_paused = True

    def resume_simulation(self):
        """Resume simulation time"""
        self.is_paused = False
        # Consume any accumulated time to prevent large dt on the next update
        self.clock.tick()

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
        surface = pygame.Surface(
            (self.display.get_width(), self.display.get_height()), pygame.SRCALPHA
        )

        for row, col in self.get_filtered_cells_coordinates(Cell.BURNING):
            # Only render radius for fires that haven't spread yet
            if (row, col) not in self.field.has_spread:
                center_x = int((col + 0.5) * self.cell_dimensions)
                center_y = int((row + 0.5) * self.cell_dimensions)
                radius_pixels = int(SPREAD_RADIUS * self.pixels_per_meter)
                color = (*self.field.get_color(Cell.BURNING), 50)
                pygame.draw.circle(
                    surface, color, (center_x, center_y), radius_pixels, radius_pixels
                )

        self.display.blit(surface, (0, 0))

    def render_hud(self, message: str = ""):
        HUD_BG_COLOR = (0, 0, 0)
        HUD_FONT_COLOR = (200, 200, 200)

        hud_rect = pygame.Rect(
            0, self.field_dimensions, self.field_dimensions, self.hud_height
        )
        pygame.draw.rect(self.display, HUD_BG_COLOR, hud_rect)

        # Format world time
        time_str = f"{self.world_time:4.1f}s"

        # Cell states tally
        tally_str = "  ".join(
            f"{cell.name}: {count:<3d}"
            for cell, count in self.field.tally_cells().items()
        )

        # Scores (color-coded)
        score_str = f"WU: {self.wumpus_score:<3d} FT: {self.firetruck_score:<3d}"

        # Pause indication
        pause_str = "PAUSED" if self.is_paused else ""

        text = " | ".join(
            [
                s
                for s in [
                    score_str,
                    time_str,
                    tally_str,
                    pause_str,
                    message,
                ]
                if s
            ]
        )

        img = self.font.render(text, True, HUD_FONT_COLOR)
        self.display.blit(
            img, (hud_rect.x + self.hud_padding, hud_rect.y + self.hud_padding)
        )


@dataclass(frozen=True)
class Pos:
    """Position in the world: x, y, heading"""

    x: float  # m
    y: float  # m
    heading: float  # rad, 0 is along the x-axis, CCW is positive
    location: Tuple[int, int] = None  # Optional (row, col) for roadmap poses
    fine_location: Tuple[int, int] = (
        None  # Optional fine grid location (row, col) for firetruck poses
    )

    def distance_to(self, other: "Pos") -> float:
        """Euclidean distance to another position"""
        return math.hypot(self.x - other.x, self.y - other.y)

    def heading_error_to(self, other: "Pos") -> float:
        """Heading error to another position (wrapped to [-pi, pi])"""
        error = abs(self.heading - other.heading)
        if error > math.pi:
            error = 2 * math.pi - error
        return error

    def to_xy_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple for rendering"""
        return self.x, self.y

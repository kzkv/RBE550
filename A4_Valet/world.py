# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Obstacle generator: parametric generation for the grid of specified size filling with tetrominoes
# Refactor/partial reuse of the A2 code as well as our Firebot course project code
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pygame
import math

rng = np.random.default_rng()

# Fundamental constants
GRID_DIMENSIONS = 12
CELL_SIZE = 3.0  # 3 meters per cell
PIXELS_PER_METER = 40

CELL_BG_COLOR = (255, 255, 255)
CELL_GRID_COLOR = (230, 230, 230)
OBSTACLE_BG_COLOR = (100, 100, 100)
HUD_BG_COLOR = (0, 0, 0)
HUD_FONT_COLOR = (200, 200, 200)
ROUTE_COLOR = (90, 200, 90)

# I started with a randomized field, but it was not presenting a consistent and interesting challenge
# Also, there was no reason to produce a randomized field to then persist it to eliminate run-to-run variation.
# Working with preset fields for this assignment.
PARKING_LOT_1 = np.array([  # the one from the assignment
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)

PARKING_LOT_2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)

PARKING_LOT_3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)

PARKING_LOT_4 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)

PARKING_LOT_5 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
], dtype=bool)

EMPTY_PARKING_LOT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)

EMPTY_PARKING_LOT_FOR_TRAILER = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
], dtype=bool)


@dataclass(frozen=True)
class Pos:
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


def grid_to_world(row, col):
    """Returns the (x, y) center of the cell in meters from the upper left corner at the given (row, col)"""
    x = (col + 0.5) * CELL_SIZE
    y = (row + 0.5) * CELL_SIZE
    return x, y  # meters


def world_to_grid(x, y):
    """Returns the (row, col) of the cell containing the given point coordinates (x, y) in meters"""
    col = int(x // CELL_SIZE)
    row = int(y // CELL_SIZE)
    return row, col


def pixel_to_world(px, py):
    """Mouse pixel -> world coordinates (in meters)"""
    return px / PIXELS_PER_METER, py / PIXELS_PER_METER


class World:
    def __init__(self, obstacles):
        self.grid_dimensions = GRID_DIMENSIONS
        self.cell_size = CELL_SIZE
        self.pixels_per_meter = PIXELS_PER_METER
        self.cell_dimensions = CELL_SIZE * self.pixels_per_meter
        self.field_dimensions = self.grid_dimensions * self.cell_dimensions
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.hud_padding = 10
        self.hud_height = self.font.get_height() + self.hud_padding * 2

        self.obstacles = obstacles

        self.screen = pygame.display.set_mode((self.field_dimensions, self.field_dimensions + self.hud_height))
        self.clock = pygame.time.Clock()

    def clear(self):
        self.screen.fill(CELL_BG_COLOR)

    def render_grid(self):
        s = self.cell_dimensions
        for y in range(self.grid_dimensions):
            for x in range(self.grid_dimensions):
                pygame.draw.rect(
                    self.screen,
                    CELL_GRID_COLOR,
                    pygame.Rect(x * s, y * s, s, s),
                    width=1,
                )

    def render_obstacles(self):
        s = self.cell_dimensions
        for y in range(self.grid_dimensions):
            for x in range(self.grid_dimensions):
                if self.obstacles[y, x]:
                    r = pygame.Rect(x * s, y * s, s, s)
                    pygame.draw.rect(self.screen, OBSTACLE_BG_COLOR, r)

    def render_hud(self, vehicle_location: Pos = None, destination: Pos = None, message: str = ""):
        text = message

        mx, my = pygame.mouse.get_pos()
        x, y = pixel_to_world(mx, my)
        row, col = world_to_grid(x, y)

        in_bounds = (0 <= row < self.grid_dimensions and 0 <= col < self.grid_dimensions)
        hud_rect = pygame.Rect(0, self.field_dimensions, self.field_dimensions, self.hud_height)
        pygame.draw.rect(self.screen, HUD_BG_COLOR, hud_rect)

        if vehicle_location:
            cursor_location_string = f"{x:04.1f}, {y:04.1f} (row/col {row:02d}, {col:02d})"
            vehicle_location_string = f"{vehicle_location.x:04.1f}, {vehicle_location.y:04.1f}"

            xy_error = vehicle_location.distance_to(destination)
            heading_error = vehicle_location.heading_error_to(destination)
            error_string = f"   Errors: XY {xy_error:.2f}m, heading {math.degrees(heading_error):.1f}°"

            text = f"Cursor: {cursor_location_string}   Vehicle: {vehicle_location_string}{error_string}   {message}"

        img = self.font.render(text, True, HUD_FONT_COLOR)

        if in_bounds:  # Assumes the cursor is always within bounds
            self.screen.blit(img, (hud_rect.x + self.hud_padding, hud_rect.y + self.hud_padding))

    def render_route(self, route: List[Pos]):
        if len(route) < 2:
            return  # Need at least 2 points to draw a line
        pts = [(int(pos.x * PIXELS_PER_METER), int(pos.y * PIXELS_PER_METER)) for pos in route]
        pygame.draw.lines(self.screen, ROUTE_COLOR, False, pts, 2)

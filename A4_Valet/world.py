# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Obstacle generator: parametric generation for the grid of specified size filling with tetrominoes
# Refactor/partial reuse of the A2 code as well as our Firebot course project code

import numpy as np
import pygame

rng = np.random.default_rng()

# Fundamental constants
GRID_DIMENSIONS = 12
CELL_SIZE = 3  # 3 meters per cell
PIXELS_PER_METER = 40

CELL_BG_COLOR = (255, 255, 255)
CELL_GRID_COLOR = (230, 230, 230)
OBSTACLE_BG_COLOR = (100, 100, 100)
HUD_BG_COLOR = (0, 0, 0)
HUD_FONT_COLOR = (200, 200, 200)

# There is no reason to produce a randomized field to only persist it later.
# Working with preset fields for this assignment.
PARKING_LOT = np.array([  # the one from the assignment
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=bool)


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
    def __init__(self):
        self.grid_dimensions = GRID_DIMENSIONS
        self.pixels_per_meter = PIXELS_PER_METER
        self.cell_dimensions = CELL_SIZE * self.pixels_per_meter
        self.field_dimensions = self.grid_dimensions * self.cell_dimensions
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.hud_padding = 10
        self.hud_height = self.font.get_height() + self.hud_padding * 2

        self.obstacles = PARKING_LOT

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

    def render_hud(self, vehicle_location):
        mx, my = pygame.mouse.get_pos()
        x, y = pixel_to_world(mx, my)
        row, col = world_to_grid(x, y)

        in_bounds = (0 <= row < self.grid_dimensions and 0 <= col < self.grid_dimensions)

        hud_rect = pygame.Rect(0, self.field_dimensions, self.field_dimensions, self.hud_height)
        pygame.draw.rect(self.screen, HUD_BG_COLOR, hud_rect)

        # assumes the cursor is always within bounds
        cursor_location_string = f"{x:04.1f}, {y:04.1f} (row/col {row:02d}, {col:02d})"
        vehicle_location_string = f"{vehicle_location[0]:04.1f}, {vehicle_location[1]:04.1f}"
        text = f"Cursor: {cursor_location_string}   Vehicle: {vehicle_location_string}"
        img = self.font.render(text, True, HUD_FONT_COLOR)
        if in_bounds:
            self.screen.blit(img, (hud_rect.x + self.hud_padding, hud_rect.y + self.hud_padding))

    def render_route(self, route):
        pts = [(int(x * PIXELS_PER_METER), int(y * PIXELS_PER_METER)) for (x, y) in route]
        pygame.draw.lines(self.screen, (90, 200, 90), False, pts, 2)

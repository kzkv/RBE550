# Tom Kazakov
# RBE 550
# Assignment 5, Valet
# Obstacle generator: parametric generation for the grid of specified size filling with tetrominoes
# Refactor/partial reuse of the A2 code as well as our Firebot course project code

import numpy as np
import pygame

rng = np.random.default_rng()

# Fundamental constants
GRID_DIMENSIONS = 12
CELL_SIZE = 3  # 3 meters per cell
PIXELS_PER_METER = 20

CELL_BG_COLOR = (255, 255, 255)
CELL_GRID_COLOR = (230, 230, 230)
OBSTACLE_BG_COLOR = (100, 100, 100)

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


class World:
    def __init__(self):
        self.grid_dimensions = GRID_DIMENSIONS
        self.pixels_per_meter = PIXELS_PER_METER
        self.cell_dimensions = CELL_SIZE * self.pixels_per_meter
        self.field_dimensions = self.grid_dimensions * self.cell_dimensions
        self.margin = self.cell_dimensions

        self.obstacles = PARKING_LOT

        self.screen = pygame.display.set_mode((self.field_dimensions, self.field_dimensions))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, self.cell_dimensions - 4)

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

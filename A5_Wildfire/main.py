# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

# TODO: Wumpus only ignites one cell
# TODO: don't calculate the radius, dilate and use the mask instead to ignite

# TODO: calculate cell priority for Wumpus,
#  add by number of unaffected cells in the radius,
#  penalty for already burning in the radius,
#  penalty for proximity to the truck,
#  penalty for distance of the planned route (in cells)

# TODO: visualize cell priority; validate with manually moving the truck

# TODO: make Wumpus move to the highest priority cell
# TODO: move the truck to see how Wumpus alters the plan

# TODO: scores for Wumpus and firetruck


import logging
import math

import numpy as np
import pygame

from wumpus import Wumpus
from firetruck import Firetruck
from world import World
from field import Cell

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

rng = np.random.default_rng()

pygame.init()
pygame.display.set_caption("Wildfire")

# SEED = 67
SEED = 41
TIME_SPEED = 2.0  # Time speed coefficient
PAR_TIME = 3600.0

WUMPUS_ROWS = (0, 10)
WUMPUS_COLS = (0, 10)
FIRETRUCK_ROWS = (40, 49)
FIRETRUCK_COLS = (40, 49)

world = World(SEED, time_speed=TIME_SPEED)
wumpus = Wumpus(world, WUMPUS_ROWS, WUMPUS_COLS)
firetruck = Firetruck(world, FIRETRUCK_ROWS, FIRETRUCK_COLS)
world.wumpus = wumpus
world.firetruck = firetruck

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Ignite on mouse click
            mx, my = pygame.mouse.get_pos()
            x, y = world.pixel_to_world(mx, my)
            row, col = world.world_to_grid(x, y)

            if world.field.get_cell(row, col) == Cell.OBSTACLE:
                world.field.ignite(row, col)
            elif world.field.get_cell(row, col) == Cell.EMPTY:
                heading = -math.pi / 2
                firetruck.set_pose(firetruck.grid_to_pose((row, col), heading))

    if world.world_time >= PAR_TIME:
        # TODO: consider what parts of the rendering should be done here
        world.render_hud(message="Time's up!")
        continue

    world.update()
    world.field.update_burning_cells()
    world.clear()
    world.render_field()
    world.render_spread()
    world.render_hud()

    wumpus.update()
    wumpus.render_path()
    wumpus.render()

    firetruck.update()
    firetruck.render()

    pygame.display.flip()

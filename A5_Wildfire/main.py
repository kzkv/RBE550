# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

import logging
import numpy as np
import pygame

from wumpus import Wumpus
from world import World

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

rng = np.random.default_rng()

pygame.init()
pygame.display.set_caption("Wildfire")

# SEED = 67
SEED = 41
TIME_SPEED = 10.0  # Time speed coefficient (1.0 = real time, 2.0 = 2x speed, etc.)
PAR_TIME = 3600.0
WUMPUS_ROWS = (0, 10)
WUMPUS_COLS = (0, 10)

world = World(SEED, time_speed=TIME_SPEED)
wumpus = Wumpus(world, WUMPUS_ROWS, WUMPUS_COLS)

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

            wumpus.set_goal(row, col)

    if world.world_time >= PAR_TIME:
        # TODO: consider what parts of the rendering should be done here
        world.render_hud(message="Time's up!")
        continue

    world.update()
    world.field.update_burning_cells()
    world.clear()
    world.render_field()
    world.render_spread()
    world.render_hud(message="HUD message")

    wumpus.update()
    wumpus.render_path()
    wumpus.render()

    pygame.display.flip()

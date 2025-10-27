# Tom Kazakov
# RBE 550
# Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

import numpy as np
import pygame

from world import World, Cell, Pos

rng = np.random.default_rng()

pygame.init()
pygame.display.set_caption("Wildfire")

# SEED = 67
SEED = 41

world = World(SEED)

runnig = True
while runnig:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            runnig = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Ignite on mouse click
            mx, my = pygame.mouse.get_pos()
            x, y = world.pixel_to_world(mx, my)
            row, col = world.world_to_grid(x, y)
            world.field.ignite(row, col)

    world.clear()
    world.render_field()
    world.render_hud(message="HUD message")

    pygame.display.flip()

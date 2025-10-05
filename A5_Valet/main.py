# Tom Kazakov
# RBE 550
# Assignment 5, Valet
# Gen AI usage: ChatGPT for project planning and ideation + some draft code gen

import pygame
import numpy as np
from world import World

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Valet")

world = World()

running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        else:
            pass  # TODO handle events
            # world.handle_event(e)

    world.clear()
    world.render_grid()
    world.render_obstacles()
    world.render_hud()

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()

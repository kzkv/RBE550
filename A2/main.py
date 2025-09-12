# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to ideate the pseudo-graphics implementation tech stack
# TODO: is there a better way to render than squishing the font to 0.7 line height?

from world import World
import logging
from render import render_grid, render_stats, render_game_over
from time import sleep

TICK_TIME = 0.5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

world = World(grid_size=64, rho=0.20)

while True:
    render_grid(world.grid)

    stats = world.calculate_stats()
    render_stats(stats)

    if stats[0] == 0:
        render_game_over()
        break

    world.move_enemies()
    sleep(TICK_TIME)

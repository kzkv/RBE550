# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to ideate the pseudo-graphics implementation tech stack
# TODO: is there a better way to render than squishing the font to 0.7 line height?

# TODO: reflect in the report one of the principal decisions: allow 4-connected or 8-connected movement.
"""
4-connected: less elegant movement, but avoids sqrt(2) vs equal-cost movement for diagonal compared to orthogonal.
Also simplifies the situation with corner-cutting.
Might simplify implementation with something like NetworkX (which connects grid cells orthogonally)
"""

from world import World
import logging
from render import render_grid, render_stats, render_game_over
from time import sleep
from planner import get_heros_journey

TICK_TIME = 0.5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

world = World()

# Init
render_grid(world.grid, path=[])
# sleep(TICK_TIME*5)  # Let the observer situate themselves with the map

# Main game loop
while True:
    render_grid(world.grid, path=get_heros_journey(world))

    stats = world.calculate_stats()
    render_stats(stats)

    if stats[0] == 0:
        render_game_over()
        break

    world.move_enemies()
    sleep(TICK_TIME)

# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to ideate the pseudo-graphics implementation tech stack

# TODO: is there a better way to render than squishing the font to 0.6 height 1.4 width?

from world import World
import logging
from render import render_grid, render_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

world = World(grid_size=64, rho=0.20)

render_grid(world.grid)
render_stats(world.calculate_stats())

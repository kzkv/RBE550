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
from render import render_grid, render_stats, render_game_over, render_great_success, render_stalemate
from time import sleep
from planner import get_heros_journey

TICK_TIME = 0.25

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
    heros_journey = get_heros_journey(world)

    render_grid(world.grid, path=heros_journey)
    render_stats(world)

    if not heros_journey and world.teleports > 0:
        world.teleport_hero()
        # Teleporting takes a tick to execute; this could be implemented differently,
        # but this way allows for visibility, however limited

        # Another potential application of teleportation mentioned in the assignment is
        # if the hero is "threatened". On the hundreds of test runs, it's redundant for the
        # target parameters (10 enemies, 0.2 rho), just planning the exclusion of one-cell
        # radius is enough. But the way to implement "threat" will be to establish if there
        # is a realistic intercept by an enemy, i.e. enemy is within 1 cell from the next
        # few cells of the path.

    if not heros_journey and world.count_enemies() == 0:
        # No available moves, but also no enemies left. This is a dead end.
        render_stalemate()
        break

    if not world.hero_alive:
        render_game_over()
        break

    if world.goal_reached:
        render_great_success()
        break

    world.move_enemies()
    if len(heros_journey) > 1:
        world.move_hero(heros_journey[1])

    sleep(TICK_TIME)

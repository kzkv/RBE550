# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to ideate the pseudo-graphics implementation tech stack
"""
4-connected: less elegant movement, but avoids sqrt(2) vs equal-cost movement for diagonal compared to orthogonal.
Also simplifies the situation with corner-cutting.
Might simplify implementation with something like NetworkX (which connects grid cells orthogonally)
"""
from world import World
import logging
from render import render_grid, render_stats, render_game_over, render_great_success, render_stalemate, render_stop
from planner import get_heros_journey
from blessed import Terminal

term = Terminal()  # For keystroke capture, it's important to have a terminal as a singleton, handed off to rendering.

TICK_TIME = 0.001

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

world = World()

# Init
render_grid(term, world.grid, path=[])

outcome = ""
with term.cbreak(), term.hidden_cursor():
    # Main game loop

    while True:
        heros_journey = get_heros_journey(world)

        render_grid(term, world.grid, path=heros_journey)
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
            outcome = "stalemate"
            render_stalemate(term)
            break

        if not world.hero_alive:
            outcome = "game_over"
            render_game_over(term)
            break

        if world.goal_reached:
            outcome = "great_success"
            render_great_success(term)
            break

        world.move_enemies()
        if len(heros_journey) > 1:
            world.move_hero(heros_journey[1])

        if term.inkey(timeout=TICK_TIME):
            outcome = "stop"
            render_stop(term)
            break

print(term.normal, end="")
world.tsv_out(outcome)

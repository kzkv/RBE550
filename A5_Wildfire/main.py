# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

# Motion-plan for firetruck
# TODO: tune the simulation

# Performance profiling
# TODO: run cpython profiling
# TODO: see if in-run profiling is necessary

# If I had more time
# TODO: consider singularity points (an obstacle trap present in SEED = 41)
# TODO: roadmap failsafe -> drive to the nearest POI if lost

# Report:
# Pos vs location (discrete vs continuous)

import logging

import numpy as np
import pygame

from firetruck import Firetruck
from world import World
from wumpus import Wumpus

DEBUG = False

logger = logging.getLogger(__name__)
(
    logging.basicConfig(level=logging.DEBUG)
    if DEBUG
    else logging.basicConfig(level=logging.INFO)
)

rng = np.random.default_rng()

pygame.init()
pygame.display.set_caption("Wildfire")

SEED = 67
# SEED = 41
TIME_SPEED = 10.0  # Time speed coefficient
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

# Reset the clock to disregard setup time
world.clock.tick()

logger.info("Starting main loop")

if DEBUG:
    connectivity = firetruck.analyze_roadmap_connectivity()
    logger.debug(connectivity)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # TODO: remove the manual controls when no longer needed
            mx, my = pygame.mouse.get_pos()
            x, y = world.pixel_to_world(mx, my)
            row, col = world.world_to_grid(x, y)

    if world.world_time >= PAR_TIME:
        # TODO: consider what parts of the rendering should be done here
        world.render_hud(message="Time's up!")
        continue

    # Compute changes, set goals, update actors
    world.pause_simulation()
    wumpus.update()
    wumpus.set_goal_auto()
    firetruck.update()
    world.resume_simulation()

    # Handle world state changes and rendering
    world.update()
    world.field.update_burning_cells()
    world.clear()
    world.render_field()
    if DEBUG:
        world.field.render_collision_overlay()
    world.render_spread()
    world.render_hud()

    # Handle Wumpus
    wumpus.render_priority_heatmap()
    wumpus.move()
    wumpus.render_path()
    wumpus.render()

    # Handle Firetruck
    if DEBUG:
        firetruck.render_roadmap()
    firetruck.render_top_priority_pois()
    firetruck.render_coverage_radius()
    firetruck.render_planned_path()
    firetruck.render()

    pygame.display.flip()

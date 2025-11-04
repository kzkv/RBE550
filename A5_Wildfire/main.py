# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

# Motion-plan for firetruck
# TODO: drive toward the max interest point
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

from field import Cell
from firetruck import Firetruck
from world import World
from wumpus import Wumpus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

rng = np.random.default_rng()

pygame.init()
pygame.display.set_caption("Wildfire")

# SEED = 67
SEED = 41
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

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # TODO: remove the manual controls when no longer needed
            # Ignite on mouse click
            mx, my = pygame.mouse.get_pos()
            x, y = world.pixel_to_world(mx, my)
            row, col = world.world_to_grid(x, y)

            if world.field.get_cell(row, col) == Cell.OBSTACLE:
                world.field.ignite(row, col)
                pass
            elif world.field.get_cell(row, col) == Cell.EMPTY:
                # Find the nearest POI to the clicked cell
                nearest_poi = firetruck.find_nearest_poi_to_location((row, col))

                if nearest_poi is None:
                    logger.warning(f"No POI found near {(row, col)}")
                else:
                    # Plan a path to the nearest POI
                    if firetruck.plan_path_to_poi(nearest_poi):
                        logger.info(f"Planned path to nearest POI at {nearest_poi}")
                    else:
                        logger.warning(f"Could not plan path to POI at {nearest_poi}")

    if world.world_time >= PAR_TIME:
        # TODO: consider what parts of the rendering should be done here
        world.render_hud(message="Time's up!")
        continue

    world.update()
    world.field.update_burning_cells()
    world.clear()
    world.render_field()
    world.field.render_collision_overlay()
    world.render_spread()
    world.render_hud()

    # firetruck.render_roadmap()

    # wumpus.update()
    # wumpus.render_priority_heatmap()
    # wumpus.move(dt_world)
    # wumpus.render_path()
    # wumpus.render()
    # wumpus.set_goal_auto()

    firetruck.render_poi_locations()
    firetruck.render_top_priority_pois()
    firetruck.update()
    firetruck.render_coverage_radius()
    firetruck.render_planned_path()
    firetruck.render()

    pygame.display.flip()

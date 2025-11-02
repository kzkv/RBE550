# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
# See Gen AI usage approach write-up in the report

# Motion-plan for firetruck
# TODO: persist roadmap for quick access between runs
# TODO: collision checking for the R-S curves
# TODO: drive toward the selected point; accel/decel
# TODO: drive in forward and in reverse with different speeds
# TODO: take into account the discrepancy between planning against body center and actual kinematics (pivot in the rear axle)
# TODO: establish points of interest for the firetruck
# TODO: drive toward the max interest point; take into account accel/decel
# TODO: consider trajectory follower to avoid pauses between segments
# TODO: tune the simulation

# Performance profiling
# TODO: run cpython profiling
# TODO: see if in-run profiling is necessary

# Report:
# Pos vs location (discrete vs continuous)

import logging

import numpy as np
import pygame
import math

from field import Cell
from firetruck import Firetruck
from world import World, Pos
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
                # world.field.ignite(row, col)
                pass
            elif world.field.get_cell(row, col) == Cell.EMPTY:
                # Plan a path to the clicked cell
                if firetruck.plan_path_to_location((row, col)):
                    final_pose = firetruck.planned_path_segments[-1].end
                    firetruck.set_pose(final_pose)
                    logger.info(f"Planned path to {(row, col)}")
                else:
                    logger.warning(f"Could not plan path to {(row, col)}")

    if world.world_time >= PAR_TIME:
        # TODO: consider what parts of the rendering should be done here
        world.render_hud(message="Time's up!")
        continue

    dt_world = world.update()
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

    # firetruck.render_poi_locations()
    firetruck.update()
    firetruck.render_planned_path()
    firetruck.render()

    pygame.display.flip()

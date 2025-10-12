# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for project planning and ideation + some draft code gen

import pygame
import numpy as np

from A4_Valet.vehicle import VehicleSpec, Vehicle
from world import World

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Valet")

world = World()

ROBOT = VehicleSpec(
    length=0.7, width=0.57, wheelbase=None, cargo_manifest="Burrito", cruising_velocity=1.0, track_width=0.57
)
robot = Vehicle(ROBOT, origin=(5.0, 5.0), heading=0.0)
robot.velocity = 1.0

# Scripted world-coordinate waypoints (meters)
waypoints = [
    (2.0, 2.0),
    (8.0, 2.0),
    (8.0, 6.0),
    (3.5, 6.0),
]

robot.set_destination(waypoints.pop(0))

running = True
while running:
    delta_time = world.clock.get_time() / 1000.0  # seconds

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        else:
            pass  # TODO handle events
            # world.handle_event(e)

    robot.drive(delta_time, world)

    if robot.are_we_there_yet() and waypoints:
        robot.set_destination(waypoints.pop(0))

    world.clear()
    world.render_grid()
    world.render_obstacles()
    world.render_hud()

    robot.render(world)
    robot.render_breadcrumbs(world)

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()

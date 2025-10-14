# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for project planning and ideation + some draft code gen

import pygame
import numpy as np
import math

from follower import PathFollower
from vehicle import VehicleSpec, Vehicle
from world import World

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Valet")

world = World()

ROBOT = VehicleSpec(
    length=0.7,
    width=0.57,
    wheelbase=None,
    cargo_manifest="Burrito",
    cruising_velocity=2.0,
    track_width=0.57  # Assumed the same as the vehicle
)
ORIGIN = (1.5, 1.5)
vehicle = Vehicle(ROBOT, origin=ORIGIN, heading=math.pi / 2)

# Scripted world-coordinate waypoints  TODO: replace with an output of the path planner
waypoints = [
    (7.0, 1.0),
    # (7.3, 2.3),
    (7.0, 13.0),
    (16.0, 13.0),
    (16.0, 24.0),
    (27.7, 34.5),
    (28.7, 34.5)
]
full_route = [ORIGIN] + waypoints

follower = PathFollower(
    full_route,
    lookahead=1.0,
    v_cruise=3.0,
    w_max=math.pi / 2,
)

running = True
while running:
    delta_time = world.clock.get_time() / 1000.0  # seconds

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        else:
            pass  # TODO handle events
            # world.handle_event(e)

    follower.update(vehicle, delta_time)
    vehicle.drive(delta_time, world)

    world.clear()
    world.render_grid()
    world.render_obstacles()
    world.render_route(full_route)
    world.render_hud((vehicle.x, vehicle.y))

    vehicle.render(world)
    vehicle.render_breadcrumbs(world)

    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()

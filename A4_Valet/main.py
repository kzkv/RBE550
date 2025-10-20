# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for project planning and ideation + some draft code gen

import pygame
import numpy as np
import math

from follower import PathFollower
from vehicle import VehicleSpec, Vehicle
from world import World, Pos
from planner import plan

# TODO: refactor prints into log statements

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Valet")

world = World()

ROBOT = VehicleSpec(
    length=0.7,
    width=0.57,
    wheelbase=None,
    cargo_manifest="Burrito",
    cruising_velocity=3.0,  # A speedy little burrito carrier
    w_max=math.pi / 4,
    track_width=0.57  # Assumed the same as the vehicle
)

ORIGIN = Pos(x=1.5, y=1.5, heading=math.pi / 2)
DESTINATION = Pos(x=28.7, y=34.5, heading=0.0)

vehicle = Vehicle(ROBOT, origin=ORIGIN)

route = plan(ORIGIN, DESTINATION, world.obstacles, vehicle_width=vehicle.spec.width)
if route is None:
    print("NO PATH FOUND!")
    route = [(ORIGIN.x, ORIGIN.y)]

full_route = route

follower = PathFollower(
    full_route,
    lookahead=0.5,
    vehicle=vehicle
)

# Render initial frame BEFORE starting the loop
world.clear()
world.render_grid()
world.render_obstacles()
world.render_route(full_route)
world.render_hud((vehicle.pos.x, vehicle.pos.y))
vehicle.render(world)
pygame.display.flip()

# Initialize the clock properly - this starts the timing
world.clock.tick(0)

# Reset clock right before loop to avoid large first delta_time
world.clock.tick()

running = True
while running:
    delta_time = world.clock.get_time() / 1000.0

    # To avoid the simulation running ahead during the initial planning phase,
    # clamp excessive delta_time (also happens during lag).
    delta_time = min(delta_time, 0.1)  # Max 100ms per frame
    
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
    
    follower.update(delta_time)
    vehicle.drive(delta_time, world)
    
    world.clear()
    world.render_grid()
    world.render_obstacles()
    world.render_route(full_route)
    world.render_hud((vehicle.pos.x, vehicle.pos.y))
    vehicle.render(world)
    vehicle.render_breadcrumbs(world)
    
    pygame.display.flip()
    world.clock.tick(60)

pygame.quit()

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
from world import PARKING_LOT_1, PARKING_LOT_2, PARKING_LOT_3, PARKING_LOT_4, EMPTY_PARKING_LOT, EMPTY_PARKING_LOT_FOR_TRAILER
from planner import plan

# TODO: refactor prints into log statements

rng = np.random.default_rng()
pygame.init()
pygame.display.set_caption("Valet")

ROBOT = VehicleSpec(
    length=0.7,
    width=0.57,
    wheelbase=None,
    cargo_manifest="Burrito",
    cruising_velocity=3.0,  # A speedy little burrito carrier
    w_max=math.pi / 2,
    track_width=0.57,  # Assumed the same as the vehicle
    origin=Pos(x=1.5, y=1.5, heading=math.pi / 2),
    destination=Pos(x=28.7, y=34.5, heading=0.0),
    planned_xy_error=0.5,
    planned_heading_error=math.radians(3)  # can be pretty precise for the tiny robot
)

CAR = VehicleSpec(
    length=5.2,
    width=1.8,
    wheelbase=2.8,
    cargo_manifest="Donuts",
    cruising_velocity=6.0,  # The donut carrier is faster than the burrito carrier
    w_max=math.pi / 4,  # Limited turning rate
    track_width=1.8,
    origin=Pos(x=1.5, y=3.0, heading=math.pi / 2),
    destination=Pos(x=27.0, y=34.5, heading=0.0),
    planned_xy_error=1.0,
    planned_heading_error=math.radians(10)  # a looser parked heading requirement
)

"""MODIFY THIS TO SET UP THE SIMULATION"""
# vehicle = Vehicle(ROBOT)
vehicle = Vehicle(CAR)

# world = World(PARKING_LOT_1)
# world = World(PARKING_LOT_2)
# world = World(PARKING_LOT_3)
# world = World(PARKING_LOT_4)
world = World(EMPTY_PARKING_LOT)
# world = World(EMPTY_PARKING_LOT_FOR_TRAILER)

# Render something to look at while the path planning is running
world.clear()
world.render_grid()
world.render_obstacles()
vehicle.render(world, pos=vehicle.spec.destination)  # TODO: mention using the vehicle render as the bounding box
vehicle.render_parking_zone(world)
vehicle.render(world)  # Render current position
world.render_hud(message="Planning route, please wait...")
pygame.display.flip()
pygame.event.pump()

route = plan(vehicle.spec.origin, vehicle.spec.destination, world.obstacles, vehicle.spec)

if route is None:
    print("NO PATH FOUND!")
    route = [vehicle.spec.origin]

full_route = route

follower = PathFollower(
    full_route,
    lookahead=0.5,
    vehicle=vehicle
)

# Reset clock right before loop to avoid large first delta_time
world.clock.tick()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    delta_time = world.clock.get_time() / 1000.0

    # To avoid the simulation running ahead during the initial planning phase,
    # clamp excessive delta_time (also happens during lag).
    delta_time = min(delta_time, 0.1)  # Max 100ms per frame

    world.clear()
    world.render_grid()
    world.render_obstacles()

    # Render destination
    vehicle.render(world, pos=vehicle.spec.destination)
    vehicle.render_parking_zone(world)

    if len(route) > 1:  # we actually have a route to follow
        follower.update(delta_time)
        vehicle.drive(delta_time, world)
        world.render_hud(vehicle_location=vehicle.pos, destination=vehicle.spec.destination)
        world.render_route(full_route)
    else:
        world.render_hud(message="NO PATH FOUND!")

    vehicle.render(world)  # Render current position
    vehicle.render_breadcrumbs(world)

    pygame.display.flip()
    world.clock.tick(60)

# pygame.quit()

# Tom Kazakov
# RBE 550
# Assignment 4, Valet
# Gen AI usage: ChatGPT for project planning and ideation + some draft code gen

import pygame
import numpy as np
import math

from collision import CollisionChecker
from follower import PathFollower
from vehicle import VehicleSpec, TrailerSpec, Vehicle, KinematicModel
from world import World, Pos, ROUTE_COLOR, EXPLORED_ROUTE_COLOR
from world import (
    PARKING_LOT_1,
    PARKING_LOT_2,
    PARKING_LOT_3,
    PARKING_LOT_4,
    PARKING_LOT_5,
    PARKING_LOT_6,
    EMPTY_PARKING_LOT,
    EMPTY_PARKING_LOT_FOR_TRAILER
)
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
    max_steering_angle=0.0,  # Irrelevant for a diff drive robot
    max_acceleration=2.0,  # m/s^2
    track_width=0.57,  # Assumed the same as the vehicle
    origin=Pos(x=1.5, y=1.5, heading=math.pi / 2),
    destination=Pos(x=28.7, y=34.5, heading=0.0),
    safety_margin=0.1,
    planned_xy_error=0.5,
    planned_heading_error=math.radians(3),  # can be pretty precise for the tiny robot
    kinematic_model=KinematicModel.DIFF_DRIVE
)

CAR = VehicleSpec(
    length=5.2,
    width=1.8,
    wheelbase=2.8,
    cargo_manifest="Donuts",
    cruising_velocity=6.0,  # The donut carrier is faster than the burrito carrier
    w_max=0.0,  # Constrained by steering
    max_steering_angle=math.radians(35),  # Typical steering for a passenger car
    max_acceleration=3.0,  # m/s^2
    track_width=1.8,
    origin=Pos(x=1.5, y=3.0, heading=math.pi / 2),
    destination=Pos(x=27.0, y=34.5, heading=0.0),
    safety_margin=0.1,
    planned_xy_error=1.2,
    planned_heading_error=math.radians(90),  # a looser parked heading requirement
    kinematic_model=KinematicModel.ACKERMANN
)

TRUCK = VehicleSpec(
    length=5.4,
    width=2.0,
    wheelbase=3.4,
    track_width=1.8,
    cargo_manifest="Old refrigerators and dimensional lumberjacks",
    cruising_velocity=3.0,  # A bit slower than a car
    w_max=0.0,
    max_steering_angle=math.radians(35),
    max_acceleration=1.5,  # m/s^2, also a bit slower than a car
    origin=Pos(x=3.5, y=10.5, heading=math.pi / 2),
    destination=Pos(x=27.0, y=34.5, heading=0.0),
    safety_margin=0.1,
    planned_xy_error=1.2,
    planned_heading_error=math.radians(120),
    kinematic_model=KinematicModel.ACKERMANN,
    trailer=TrailerSpec(
        length=4.5,
        width=2.0,
        hitch_length=5.0,  # d1 from the problem
    )
)

"""MODIFY THIS TO SET UP THE SIMULATION"""
# vehicle = Vehicle(ROBOT)
vehicle = Vehicle(CAR)
# vehicle = Vehicle(TRUCK)

# Worlds for the ROBOT
# world = World(PARKING_LOT_1)
# world = World(PARKING_LOT_2)

# Worlds for the CAR
world = World(PARKING_LOT_3)
# world = World(PARKING_LOT_4)

# Worlds for the TRUCK
# world = World(PARKING_LOT_5)
# world = World(PARKING_LOT_6)

# Empty parking lots
# world = World(EMPTY_PARKING_LOT)
# world = World(EMPTY_PARKING_LOT_FOR_TRAILER)

RENDER_OVERLAY = True
# RENDER_OVERLAY = False

# RENDER_EXPLORED_ROUTES = True
RENDER_EXPLORED_ROUTES = False
""""""

collision = CollisionChecker(world, vehicle.spec)

# Render something to look at while the path planning is running
world.clear()
world.render_grid()
world.render_obstacles()
vehicle.render(world, pos=vehicle.spec.destination)
vehicle.render_parking_zone(world)
vehicle.render(world)  # Render current position

if RENDER_OVERLAY:
    collision.render_loose_overlay()
    collision.render_boundary_overlay()
    collision.render_tight_overlay()

world.render_hud(message="Planning route, please wait...")

pygame.display.flip()
pygame.event.pump()

route, explored_segments = plan(vehicle.spec, collision)

if route is None or len(route) == 0:
    print("NO PATH FOUND!")
    route = [vehicle.spec.origin]

full_route = route

# Compute trailer route if vehicle has trailer
trailer_route = None
if vehicle.spec.trailer is not None and len(route) > 1:
    print("Computing trailer path from truck path...")
    trailer_route = Vehicle.integrate_trailer_along_path(
        route,
        vehicle.spec.origin.heading,  # Initial trailer heading (aligned with truck)
        vehicle.spec
    )
    print(f"  Trailer path: {len(trailer_route)} waypoints")

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

    if RENDER_EXPLORED_ROUTES:
        for segment in explored_segments:
            world.render_route(segment, EXPLORED_ROUTE_COLOR)
        world.render_route(route, ROUTE_COLOR)
        vehicle.render(world)
        world.render_hud(message=f"Explored {len(explored_segments)} segments")
        pygame.display.flip()
        continue

    if RENDER_OVERLAY:
        collision.render_loose_overlay()
        collision.render_boundary_overlay()
        collision.render_tight_overlay()

    if len(route) > 1:  # We actually have a route to follow
        from world import ROUTE_COLOR, TRAILER_ROUTE_COLOR

        follower.update()
        vehicle.drive(delta_time, world)
        world.render_hud(vehicle_location=vehicle.pos, destination=vehicle.spec.destination)
        world.render_route(full_route, ROUTE_COLOR)  # Truck route in green

        # Render trailer route in purple
        if trailer_route is not None:
            world.render_route(trailer_route, TRAILER_ROUTE_COLOR)
    else:
        world.render_hud(message="NO PATH FOUND!")

    vehicle.render(world)  # Render current position
    vehicle.render_breadcrumbs(world)

    pygame.display.flip()
    world.clock.tick(60)

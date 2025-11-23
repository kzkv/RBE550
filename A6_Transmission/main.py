# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import numpy as np
from transmission import Transmission
from rrt import RRT
import trimesh

# Configuration
RECALCULATE_PATH = False
SHOW_GOAL = False  # Show goal pose
SHOW_PATH = True  # View saved path
ANIMATE_PATH = False  # Animate path

# RRT parameters
STEP_SIZE = 5.0  # Step size in mm
ROTATION_STEP_SIZE = np.radians(3.0)  # Step size in radians
MAX_ITER = 20000  # Max iterations
GOAL_THRESHOLD = 5.0
GOAL_SAMPLE_RATE = 0.1  # Sample goal some portion of the time
OUTPUT_FILE = "path.npy"  # Path output file
SEED = 67

# Visualization parameters
PATH_FILE = "path.npy"  # Path file to visualize
CAMERA = {"angles": [np.radians(75), np.radians(0), np.radians(0)], "distance": 800}
ANIMATION_SPEED = 1.0  # Speed multiplier

if __name__ == "__main__":
    path_found = False

    transmission = Transmission()
    default_pos = transmission.primary_centroid.copy()

    start_config = np.concatenate([default_pos, [0.0, 0.0, 0.0]])

    X_OFFSET = -200  # mm, pull toward the case body
    Y_OFFSET = 0  # mm, lateral movement
    Z_OFFSET = 300  # mm, lift
    ROLL = 0.0  # degrees, shaft rotation
    PITCH = 90.0  # degrees, tilt vertical
    YAW = 0.0  # degrees, yaw rotation

    goal_config = np.array(
        [
            default_pos[0] + X_OFFSET,
            default_pos[1] + Y_OFFSET,
            default_pos[2] + Z_OFFSET,
            np.radians(ROLL),
            np.radians(PITCH),
            np.radians(YAW),
        ]
    )

    # Calculate a path
    if RECALCULATE_PATH:
        rrt = RRT(
            transmission,
            start_config,
            goal_config,
            step_size=STEP_SIZE,
            rotation_step_size=ROTATION_STEP_SIZE,
            max_iter=MAX_ITER,
            goal_threshold=GOAL_THRESHOLD,
            goal_sample_rate=GOAL_SAMPLE_RATE,
            seed=SEED,
        )
        path = rrt.plan()

        if path:
            print(f"\nSUCCESS: {len(path)} waypoints, {len(rrt.tree)} nodes")
            np.save(OUTPUT_FILE, np.array(path))
            path_found = True
        else:
            print("\nFAILED: No path found")

    # DRY setup for SHOW_GOAL and SHOW_PATH
    if SHOW_GOAL or SHOW_PATH:
        scene = trimesh.Scene()
        scene.add_geometry(transmission.case, node_name="case")
        scene.add_geometry(transmission.counter, node_name="counter")

        # Start (red)
        primary_start = transmission.set_primary_pose(
            start_config[:3], start_config[3:6]
        )
        primary_start.visual.face_colors = [255, 0, 0, 100]
        scene.add_geometry(primary_start, node_name="start")

    else:
        scene = None

    # Visualize goal pose
    if SHOW_GOAL:
        # Goal (green)
        primary_goal = transmission.set_primary_pose(goal_config[:3], goal_config[3:6])
        primary_goal.visual.face_colors = [0, 255, 0, 150]
        scene.add_geometry(primary_goal, node_name="goal")

        scene.set_camera(**CAMERA)
        scene.show()

    # Visualize the saved path
    if SHOW_PATH and (not RECALCULATE_PATH or path_found):
        path = np.load(PATH_FILE)
        print(f"Loaded path: {len(path)} waypoints")

        # End (green)
        primary_end = transmission.set_primary_pose(path[-1][:3], path[-1][3:6])
        primary_end.visual.face_colors = [0, 255, 0, 200]
        scene.add_geometry(primary_end, node_name="end")

        # Intermediate (blue)
        for i in range(1, len(path) - 1):
            if i % 10 == 0:
                primary_mid = transmission.set_primary_pose(path[i][:3], path[i][3:6])
                primary_mid.visual.face_colors = [0, 100, 255, 100]
                scene.add_geometry(primary_mid, node_name=f"mid_{i}")

        # Waypoint markers
        for i, config in enumerate(path):
            sphere = trimesh.creation.icosphere(radius=3.0)
            sphere.apply_translation(config[:3])
            sphere.visual.face_colors = [255, 0, 0, 255]
            scene.add_geometry(sphere, node_name=f"waypoint_{i}")

        scene.set_camera(**CAMERA)
        scene.show()

    # Animate the saved path
    if ANIMATE_PATH and (not RECALCULATE_PATH or path_found):
        transmission = Transmission()
        path = np.load(PATH_FILE)
        transmission.animate_path(
            path=path, camera_angle=CAMERA, speed=ANIMATION_SPEED, interpolate=True
        )

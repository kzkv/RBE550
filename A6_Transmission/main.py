# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import numpy as np
from transmission import (
    Transmission,
    COLOR_START,
    COLOR_GOAL,
    COLOR_END,
    COLOR_INTERMEDIATE,
)
from rrt import RRT

# Configuration
RECALCULATE_PATH = False
SHOW_GOAL = False  # Show goal pose
SHOW_PATH = True  # View saved path
ANIMATE_PATH = False  # Animate path

# RRT parameters
STEP_SIZE = 5.0  # Step size in mm
ROTATION_STEP_SIZE = np.radians(3.0)  # Step size in radians
MAX_ITER = 1000  # Max iterations
GOAL_THRESHOLD = 5.0
GOAL_SAMPLE_RATE = 0.1  # Sample goal some portion of the time
POS_MARGIN = 50.0  # mm, margin around start/goal for position bounds
ROT_MARGIN_DEG = 10.0  # Degree margin beyond goal for rotation bounds
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
            pos_margin=POS_MARGIN,
            rot_margin_deg=ROT_MARGIN_DEG,
            seed=SEED,
        )
        path = rrt.plan()

        if path:
            print(f"\nSUCCESS: {len(path)} waypoints, {len(rrt.tree)} nodes")
            np.save(OUTPUT_FILE, np.array(path))
            path_found = True
        else:
            print("\nFAILED: No path found")

    # Visualize goal pose
    if SHOW_GOAL:
        scene = transmission.create_base_scene()
        transmission.add_primary_to_scene(scene, start_config, COLOR_START, "start")
        transmission.add_primary_to_scene(scene, goal_config, COLOR_GOAL, "goal")
        scene.set_camera(**CAMERA)
        scene.show()

    # Visualize the saved path
    if SHOW_PATH and (not RECALCULATE_PATH or path_found):
        path = np.load(PATH_FILE)
        print(f"\nLoaded path: {len(path)} waypoints")

        scene = transmission.create_base_scene()
        transmission.add_primary_to_scene(scene, start_config, COLOR_START, "start")
        transmission.add_primary_to_scene(scene, path[-1], COLOR_END, "end")

        # Intermediate poses, display every 10th
        for i in range(10, len(path) - 1, 10):
            transmission.add_primary_to_scene(
                scene, path[i], COLOR_INTERMEDIATE, f"mid_{i}"
            )

        # Waypoint markers
        for i, config in enumerate(path):
            transmission.add_waypoint_sphere(
                scene, config[:3], node_name=f"waypoint_{i}"
            )

        scene.set_camera(**CAMERA)
        scene.show()

    # Animate the saved path
    if ANIMATE_PATH and (not RECALCULATE_PATH or path_found):
        transmission = Transmission()
        path = np.load(PATH_FILE)
        transmission.animate_path(
            path=path, camera_angle=CAMERA, speed=ANIMATION_SPEED, interpolate=True
        )

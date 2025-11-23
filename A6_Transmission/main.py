# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import numpy as np
from transmission import Transmission
from rrt import RRT

# CONFIGURATION
MODE = "plan_poc"  # Plan short escape path
# MODE = "plan_extract"  # Plan full extraction: lift and twist
RECALCULATE_PATH = True
SHOW_PATH = True  # View saved path
SHOW_GOAL = False

# RRT PARAMETERS
STEP_SIZE = 5.0  # Step size in mm
ROTATION_STEP_SIZE = np.radians(3.0)  # Step size in radians
MAX_ITER = 20000  # Max iterations
GOAL_THRESHOLD = 20.0
GOAL_SAMPLE_RATE = 0.1  # Sample goal some portion of the time
OUTPUT_FILE = "path.npy"  # Path output file
SEED = 67

# VISUALIZATION PARAMETERS
PATH_FILE = "path.npy"  # Path file to visualize
CAMERA = {"angles": [np.radians(75), np.radians(0), np.radians(0)], "distance": 600}

if __name__ == "__main__":
    path_found = False

    transmission = Transmission()
    default_pos = transmission._primary_centroid.copy()

    # TODO: switch from the trivial movement to full extraction
    # start_config = np.concatenate([default_pos, [0.0, 0.0, 0.0]])
    # goal_pos = default_pos.copy()
    # # goal_pos[0] += 20  # Pull into the case
    # goal_pos[2] += 50  # Lift, mm
    # goal_config = np.concatenate([goal_pos, [0.0, 0.0, 0.0]])

    start_config = np.concatenate([default_pos, [0.0, 0.0, 0.0]])

    goal_pos = default_pos.copy()
    goal_pos[0] -= 200
    goal_pos[2] += 300  # Lift 350mm
    goal_config = np.concatenate(
        [goal_pos, [0.0, np.radians(90), np.radians(0)]]
    )  # TODO: add twist to place the shaft over the case

    start_coll = transmission.check_collision(
        start_config[:3].tolist(), start_config[3:6].tolist()
    )
    goal_coll = transmission.check_collision(
        goal_config[:3].tolist(), goal_config[3:6].tolist()
    )

    print(f"Start: {start_config[:3]} {'[COLL]' if start_coll else '[FREE]'}")
    print(f"Goal:  {goal_config[:3]} {'[COLL]' if goal_coll else '[FREE]'}")
    print(f"Planning: step={STEP_SIZE}mm, iter={MAX_ITER}")

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
            dist = sum(
                np.linalg.norm(np.array(path[i + 1][:3]) - np.array(path[i][:3]))
                for i in range(len(path) - 1)
            )
            rot = sum(
                np.linalg.norm(np.array(path[i + 1][3:6]) - np.array(path[i][3:6]))
                for i in range(len(path) - 1)
            )

            print(f"\nSUCCESS: {len(path)} waypoints, {len(rrt.tree)} nodes")
            print(f"Distance: {dist:.1f}mm, Rotation: {np.degrees(rot):.1f}°")

            np.save(OUTPUT_FILE, np.array(path))
            path_found = True
            print(f"Saved: {OUTPUT_FILE}")
        else:
            print("\nFAILED: No path found")

    # MODE: Visualize goal pose
    if SHOW_GOAL:
        import trimesh

        print("Visualizing goal pose...")
        print("  Red = Start position")
        print("  Green = Goal position (check if it makes sense!)")
        print()

        scene = trimesh.Scene()
        scene.add_geometry(transmission.case, node_name="case")
        scene.add_geometry(transmission.counter, node_name="counter")

        # Start (red, semi-transparent)
        primary_start = transmission.set_primary_pose(
            start_config[:3], start_config[3:6]
        )
        primary_start.visual.face_colors = [255, 0, 0, 100]
        scene.add_geometry(primary_start, node_name="start")

        # Goal (green, semi-transparent)
        primary_goal = transmission.set_primary_pose(goal_config[:3], goal_config[3:6])
        primary_goal.visual.face_colors = [0, 255, 0, 150]
        scene.add_geometry(primary_goal, node_name="goal")

        # Add coordinate axes at goal to show orientation
        axis_length = 30
        # X-axis (red line)
        x_axis = trimesh.creation.cylinder(radius=1, height=axis_length)
        x_axis.visual.face_colors = [255, 0, 0, 255]
        # Y-axis (green line)
        y_axis = trimesh.creation.cylinder(radius=1, height=axis_length)
        y_axis.visual.face_colors = [0, 255, 0, 255]
        # Z-axis (blue line)
        z_axis = trimesh.creation.cylinder(radius=1, height=axis_length)
        z_axis.visual.face_colors = [0, 0, 255, 255]

        # Apply goal rotation and position to axes
        goal_transform = transmission._primary_transform(
            goal_config[:3], goal_config[3:6]
        )

        scene.set_camera(**CAMERA)
        scene.show()

    # MODE: Visualize saved path
    if SHOW_PATH and path_found:
        import trimesh

        transmission = Transmission()
        path = np.load(PATH_FILE)

        print(f"Loaded path: {len(path)} waypoints")

        scene = trimesh.Scene()
        scene.add_geometry(transmission.case, node_name="case")
        scene.add_geometry(transmission.counter, node_name="counter")

        # Start (red)
        primary_start = transmission.set_primary_pose(path[0][:3], path[0][3:6])
        primary_start.visual.face_colors = [255, 0, 0, 150]
        scene.add_geometry(primary_start, node_name="start")

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
            if i == 0:
                sphere.visual.face_colors = [255, 0, 0, 255]
            elif i == len(path) - 1:
                sphere.visual.face_colors = [0, 255, 0, 255]
            else:
                sphere.visual.face_colors = [255, 255, 0, 255]
            scene.add_geometry(sphere, node_name=f"waypoint_{i}")

        scene.set_camera(**CAMERA)
        scene.show()

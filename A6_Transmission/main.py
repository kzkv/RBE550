# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import numpy as np
from transmission import Transmission
from rrt import RRT

# CONFIGURATION
MODE = "plan_poc"  # Plan short escape path
# MODE = "plan_extract"  # Plan full extraction: lift and twist
SHOW_PATH = True  # View saved path

# RRT PARAMETERS
STEP_SIZE = 5.0  # Step size in mm
ROTATION_STEP_SIZE = np.radians(5.0)  # Step size in radians
MAX_ITER = 10000  # Max iterations
OUTPUT_FILE = "path.npy"  # Path output file

# VISUALIZATION PARAMETERS
PATH_FILE = "path.npy"  # Path file to visualize
CAMERA = {"angles": [np.radians(30), np.radians(45), 0], "distance": 500}

if __name__ == "__main__":

    # MODE: POC escape path (small lift)
    if MODE == "plan_poc":
        transmission = Transmission()
        default_pos = transmission._primary_centroid.copy()

        start_config = np.concatenate([default_pos, [0.0, 0.0, 0.0]])
        goal_pos = default_pos.copy()
        goal_pos[2] += 25  # Lift X mm
        goal_config = np.concatenate([goal_pos, [0.0, 0.0, 0.0]])

        start_coll = transmission.check_collision(
            start_config[:3].tolist(), start_config[3:6].tolist()
        )
        goal_coll = transmission.check_collision(
            goal_config[:3].tolist(), goal_config[3:6].tolist()
        )

        print(f"Start: {start_config[:3]} {'[COLL]' if start_coll else '[FREE]'}")
        print(f"Goal:  {goal_config[:3]} {'[COLL]' if goal_coll else '[FREE]'}")
        print(f"Planning: step={STEP_SIZE}mm, iter={MAX_ITER}")

        rrt = RRT(
            transmission,
            start_config,
            goal_config,
            STEP_SIZE,
            ROTATION_STEP_SIZE,
            MAX_ITER,
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
            print(f"Saved: {OUTPUT_FILE}")
        else:
            print("\nFAILED: No path found")

    # MODE: Plan full extraction (350mm lift)
    elif MODE == "plan_extract":
        transmission = Transmission()
        default_pos = transmission._primary_centroid.copy()

        start_config = np.concatenate([default_pos, [0.0, 0.0, 0.0]])
        goal_pos = default_pos.copy()
        goal_pos[2] += 350  # Lift 350mm
        goal_config = np.concatenate(
            [goal_pos, [0.0, 0.0, 0.0]]
        )  # TODO: add twist to place the shaft over the case

        start_coll = transmission.check_collision(
            start_config[:3].tolist(), start_config[3:6].tolist()
        )
        goal_coll = transmission.check_collision(
            goal_config[:3].tolist(), goal_config[3:6].tolist()
        )

        print(f"Start: {start_config[:3]} {'[COLL]' if start_coll else '[FREE]'}")
        print(f"Goal:  {goal_config[:3]} {'[COLL]' if goal_coll else '[FREE]'}")
        print(f"Planning: step={STEP_SIZE}mm, iter={MAX_ITER}, bias={GOAL_BIAS}\n")

        rrt = RRT(
            transmission, start_config, goal_config, STEP_SIZE, MAX_ITER, GOAL_BIAS
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
            print(f"Saved: {OUTPUT_FILE}")
        else:
            print("\nFAILED: No path found")

    # MODE: Visualize saved path
    if SHOW_PATH:
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

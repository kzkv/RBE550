import trimesh
import numpy as np
from trimesh.viewer import SceneViewer


class Transmission:
    """Minimal transmission with positioned meshes"""

    def __init__(self):
        """Load and position meshes"""

        # Load STL files
        # Shaft meshes have been pre-positioned before exporting from OpenSCAD
        self.case = trimesh.load_mesh("mesh/case.stl")
        self.primary = trimesh.load_mesh("mesh/primary_shaft.stl")
        self.counter = trimesh.load_mesh("mesh/counter_shaft.stl")

        # Assign colors
        self.case.visual.face_colors = [100, 100, 200, 150]  # Blue, transparent
        self.primary.visual.face_colors = [255, 165, 0, 255]  # Orange
        self.counter.visual.face_colors = [128, 128, 128, 255]  # Gray

        # Cache for convenience
        self._primary_centroid = self.primary.centroid

    def _primary_transform(self, position, rotation=None):
        """Compute 4x4 transform for the primary shaft for a given pose."""
        position = np.asarray(position, dtype=float)

        if rotation is None:
            # Pure translation: move centroid to `position`
            transform = np.eye(4)
            transform[:3, 3] = position - self._primary_centroid
            return transform

        rotation = np.asarray(rotation, dtype=float)

        # Move centroid to origin
        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -self._primary_centroid

        # Euler rotation matrix (roll, pitch, yaw)
        R = trimesh.transformations.euler_matrix(rotation[0], rotation[1], rotation[2])

        # Then move the rotated shaft so its centroid is at `position`
        T_pos = np.eye(4)
        T_pos[:3, 3] = position

        # Overall transform: translate to origin -> rotate -> translate to target
        return T_pos @ R @ T_to_origin

    def set_primary_pose(self, position, rotation=None):
        """Return a copy of the primary shaft with the given pose."""
        primary_copy = self.primary.copy()
        transform = self._primary_transform(position, rotation)
        primary_copy.apply_transform(transform)
        return primary_copy

    def _create_scene(self):
        """Create a scene with named geometries."""
        scene = trimesh.Scene()
        scene.add_geometry(self.case, node_name="case")
        scene.add_geometry(self.counter, node_name="counter")
        scene.add_geometry(self.primary, node_name="primary")
        return scene

    def show(self, camera_angle):
        """Show a static 3D viewer with current geometry."""
        scene = self._create_scene()

        scene.set_camera(
            angles=camera_angle.get("angles"),
            distance=camera_angle.get("distance"),
        )

        scene.show()

    def animate_path(self, path, camera_angle, speed=1.0, interpolate=True):
        """
        Animate the primary shaft following a path.

        Args:
            path: List of poses, where each pose is [x, y, z] or [x, y, z, roll, pitch, yaw]
            camera_angle: Camera settings dict with 'angles' and 'distance'
            speed: Animation speed multiplier (1.0 = normal, 2.0 = 2x faster)
            interpolate: If True, smoothly interpolate between waypoints

        The path will be the primary output of RRT planning.
        """
        if len(path) == 0:
            print("Warning: Empty path provided")
            return

        # Convert path to numpy array and ensure 6-DOF format
        waypoints = []
        for pose in path:
            pose = np.asarray(pose, dtype=float)
            if len(pose) == 3:
                # [x, y, z] -> [x, y, z, 0, 0, 0]
                pose = np.concatenate([pose, [0.0, 0.0, 0.0]])
            elif len(pose) == 6:
                pass  # Already in correct format
            else:
                raise ValueError(
                    f"Pose must be [x,y,z] or [x,y,z,roll,pitch,yaw], got {pose}"
                )
            waypoints.append(pose)

        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)

        print(f"Animating path with {num_waypoints} waypoints...")

        # Create scene
        scene = self._create_scene()
        scene.set_camera(
            angles=camera_angle.get("angles"),
            distance=camera_angle.get("distance"),
        )

        # Animation state
        state = {
            "t": 0.0,  # Current time parameter
            "speed": speed,
            "waypoints": waypoints,
            "num_waypoints": num_waypoints,
            "interpolate": interpolate,
        }

        def animation_callback(viewer):
            """Update primary shaft position along the path."""
            state["t"] += 0.01 * state["speed"]

            if state["interpolate"]:
                # Smooth interpolation between waypoints
                # Map t to waypoint index
                total_progress = state["t"] % 1.0  # Loop animation
                waypoint_progress = total_progress * (state["num_waypoints"] - 1)

                idx = int(waypoint_progress)
                local_t = waypoint_progress - idx

                if idx >= state["num_waypoints"] - 1:
                    # At the end
                    current_pose = state["waypoints"][-1]
                else:
                    # Interpolate between waypoints
                    pose_start = state["waypoints"][idx]
                    pose_end = state["waypoints"][idx + 1]
                    current_pose = (1 - local_t) * pose_start + local_t * pose_end
            else:
                # Step through waypoints (no interpolation)
                idx = int(state["t"] * state["num_waypoints"]) % state["num_waypoints"]
                current_pose = state["waypoints"][idx]

            # Extract position and rotation
            position = current_pose[:3]
            rotation = current_pose[3:6]

            # Update transform
            transform = self._primary_transform(position, rotation)
            scene.graph.update(frame_to="primary", matrix=transform)

        SceneViewer(
            scene,
            start_loop=True,
            callback=animation_callback,
        )

import trimesh
import numpy as np
from trimesh.viewer import SceneViewer

# Reduce mesh size for faster computation by this much
SHAFT_MESH_SIMPLIFICATION_PERCENT = 0.5
CASE_MESH_SIMPLIFICATION_PERCENT = 0.25

# Visualization colors [R, G, B, Alpha]
COLOR_CASE = [100, 100, 200, 100]
COLOR_COUNTER = [128, 128, 128, 255]
COLOR_PRIMARY = [255, 165, 0, 255]
COLOR_START = [255, 0, 0, 255]
COLOR_GOAL = [0, 255, 0, 255]
COLOR_END = [0, 255, 0, 255]
COLOR_INTERMEDIATE = [0, 100, 255, 100]
COLOR_WAYPOINT = [255, 0, 0, 255]


class Transmission:
    """Transmission assembly with positioned meshes and collision detection."""

    def __init__(self):
        self.case = simplify_mesh("mesh/case.stl", CASE_MESH_SIMPLIFICATION_PERCENT)
        self.primary = simplify_mesh(
            "mesh/primary_shaft.stl", SHAFT_MESH_SIMPLIFICATION_PERCENT
        )
        self.counter = simplify_mesh(
            "mesh/counter_shaft.stl", SHAFT_MESH_SIMPLIFICATION_PERCENT
        )

        self.case.visual.face_colors = COLOR_CASE
        self.primary.visual.face_colors = COLOR_PRIMARY
        self.counter.visual.face_colors = COLOR_COUNTER

        self.primary_centroid = self.primary.centroid

        # Pre-build collision manager for static objects
        self._collision_manager = trimesh.collision.CollisionManager()
        self._collision_manager.add_object("case", self.case)
        self._collision_manager.add_object("counter", self.counter)

    def _primary_transform(self, position, rotation=None):
        """Compute 4x4 transform matrix for primary shaft pose."""
        position = np.asarray(position, dtype=float)

        if rotation is None:
            transform = np.eye(4)
            transform[:3, 3] = position - self.primary_centroid
            return transform

        rotation = np.asarray(rotation, dtype=float)

        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -self.primary_centroid

        R = trimesh.transformations.euler_matrix(rotation[0], rotation[1], rotation[2])

        T_pos = np.eye(4)
        T_pos[:3, 3] = position

        return T_pos @ R @ T_to_origin

    def set_primary_pose(self, position, rotation=None):
        """Return copy of a primary shaft at specified pose."""
        primary_copy = self.primary.copy()
        transform = self._primary_transform(position, rotation)
        primary_copy.apply_transform(transform)
        return primary_copy

    def check_collision(self, position, rotation=None):
        """Check if a primary shaft collides with case or counter-shaft."""
        primary_test = self.set_primary_pose(position, rotation)
        return self._collision_manager.in_collision_single(primary_test)

    def create_base_scene(self):
        """Create a scene with case and counter-shaft only (no primary)."""
        scene = trimesh.Scene()
        scene.add_geometry(self.case, node_name="case")
        scene.add_geometry(self.counter, node_name="counter")
        return scene

    def add_primary_to_scene(self, scene, config, color, node_name="primary"):
        """Add a primary shaft to scene at given config with specified color."""
        primary_copy = self.set_primary_pose(config[:3], config[3:6])
        primary_copy.visual.face_colors = color
        scene.add_geometry(primary_copy, node_name=node_name)

    def add_waypoint_sphere(
        self, scene, position, radius=3.0, color=None, node_name=None
    ):
        """Add a waypoint sphere marker to scene."""
        if color is None:
            color = COLOR_WAYPOINT
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(position)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere, node_name=node_name)

    def _create_scene(self):
        """Create a scene with current geometry."""
        scene = trimesh.Scene()
        scene.add_geometry(self.case, node_name="case")
        scene.add_geometry(self.counter, node_name="counter")
        scene.add_geometry(self.primary, node_name="primary")
        return scene

    def show(self, camera_angle):
        """Display static 3D view."""
        scene = self._create_scene()
        scene.set_camera(
            angles=camera_angle.get("angles"), distance=camera_angle.get("distance")
        )
        scene.show()

    def animate_path(self, path, camera_angle, speed=1.0, interpolate=True):
        """Animate primary shaft along path."""
        if len(path) == 0:
            print("Empty path")
            return

        waypoints = []
        for pose in path:
            pose = np.asarray(pose, dtype=float)
            if len(pose) == 3:
                pose = np.concatenate([pose, [0.0, 0.0, 0.0]])
            elif len(pose) != 6:
                raise ValueError(f"Pose must be [x,y,z] or [x,y,z,r,p,y], got {pose}")
            waypoints.append(pose)

        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)

        scene = self._create_scene()
        scene.set_camera(
            angles=camera_angle.get("angles"), distance=camera_angle.get("distance")
        )

        state = {
            "t": 0.0,
            "speed": speed,
            "waypoints": waypoints,
            "num_waypoints": num_waypoints,
            "interpolate": interpolate,
        }

        def animation_callback(_):
            state["t"] += 0.01 * state["speed"]

            if state["interpolate"]:
                total_progress = state["t"] % 1.0
                waypoint_progress = total_progress * (state["num_waypoints"] - 1)
                idx = int(waypoint_progress)
                local_t = waypoint_progress - idx

                if idx >= state["num_waypoints"] - 1:
                    current_pose = state["waypoints"][-1]
                else:
                    pose_start = state["waypoints"][idx]
                    pose_end = state["waypoints"][idx + 1]
                    current_pose = (1 - local_t) * pose_start + local_t * pose_end
            else:
                idx = int(state["t"] * state["num_waypoints"]) % state["num_waypoints"]
                current_pose = state["waypoints"][idx]

            position = current_pose[:3]
            rotation = current_pose[3:6]
            transform = self._primary_transform(position, rotation)
            scene.graph.update(frame_to="primary", matrix=transform)

        SceneViewer(scene, start_loop=True, callback=animation_callback)


def simplify_mesh(input_path, target_percent=0.5):
    """Simplify a mesh to reduce triangle count."""
    return trimesh.load_mesh(input_path).simplify_quadric_decimation(target_percent)

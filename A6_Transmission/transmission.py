import trimesh
import numpy as np
from trimesh.viewer import SceneViewer

SHAFT_MESH_SIMPLIFICATION_PERCENT = 0.5
CASE_MESH_SIMPLIFICATION_PERCENT = 0.25

# Visualization colors [R, G, B, Alpha]
COLOR_CASE = [100, 100, 200, 100]
COLOR_COUNTER = [128, 128, 128, 150]
COLOR_PRIMARY = [255, 165, 0, 255]
COLOR_START = [255, 0, 0, 100]
COLOR_GOAL = [0, 255, 0, 100]
COLOR_END = [0, 255, 0, 100]
COLOR_INTERMEDIATE = [0, 100, 255, 50]
COLOR_WAYPOINT = [255, 0, 0, 255]
COLOR_TREE_EDGE = [150, 150, 150, 255]


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

    def _primary_transform(self, position, rotation):
        """Compute 4x4 transform matrix for primary shaft pose."""
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)

        T_to_origin = np.eye(4)
        T_to_origin[:3, 3] = -self.primary_centroid

        R = trimesh.transformations.euler_matrix(rotation[0], rotation[1], rotation[2])

        T_pos = np.eye(4)
        T_pos[:3, 3] = position

        return T_pos @ R @ T_to_origin

    def set_primary_pose(self, position, rotation):
        """Return copy of a primary shaft at specified pose."""
        primary_copy = self.primary.copy()
        transform = self._primary_transform(position, rotation)
        primary_copy.apply_transform(transform)
        return primary_copy

    def check_collision(self, position, rotation):
        """Check if a primary shaft collides with case or counter-shaft."""
        primary_test = self.set_primary_pose(position, rotation)
        return self._collision_manager.in_collision_single(primary_test)

    def add_primary_to_scene(self, scene, config, color, node_name="primary"):
        primary_copy = self.set_primary_pose(config[:3], config[3:6])
        primary_copy.visual.face_colors = color
        scene.add_geometry(primary_copy, node_name=node_name)

    @staticmethod
    def add_waypoint_sphere(scene, position, radius=3.0, color=None, node_name=None):
        if color is None:
            color = COLOR_WAYPOINT
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(position)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere, node_name=node_name)

    @staticmethod
    def add_tree_edges(
        scene, configs, parent_indices, color=None, node_name="tree_edges"
    ):
        """Add RRT tree edges to the scene."""
        if color is None:
            color = COLOR_TREE_EDGE

        edges = []
        for i, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0:
                child_pos = configs[i][:3]
                parent_pos = configs[parent_idx][:3]
                edges.append([parent_pos, child_pos])

        if edges:
            edge_geometry = trimesh.load_path(edges)
            for entity in edge_geometry.entities:
                entity.color = color
            scene.add_geometry(edge_geometry, node_name=node_name)

    def animate_path(self, path, camera_angle, speed=1.0, interpolate=True):
        waypoints = np.asarray(path, dtype=float)
        num_waypoints = len(waypoints)

        scene = trimesh.Scene()
        scene.add_geometry(self.counter, node_name="counter")
        scene.add_geometry(self.primary.copy(), node_name="primary")
        scene.add_geometry(self.case, node_name="case")
        scene.set_camera(
            angles=camera_angle.get("angles"), distance=camera_angle.get("distance")
        )

        t = 0.0

        def animation_callback(_):
            nonlocal t
            t += 0.01 * speed

            if interpolate:
                total_progress = t % 1.0
                waypoint_progress = total_progress * (num_waypoints - 1)
                idx = int(waypoint_progress)
                local_t = waypoint_progress - idx

                if idx >= num_waypoints - 1:
                    current_pose = waypoints[-1]
                else:
                    pose_start = waypoints[idx]
                    pose_end = waypoints[idx + 1]
                    current_pose = (1 - local_t) * pose_start + local_t * pose_end
            else:
                idx = int(t * num_waypoints) % num_waypoints
                current_pose = waypoints[idx]

            position = current_pose[:3]
            rotation = current_pose[3:6]
            transform = self._primary_transform(position, rotation)
            scene.graph.update(frame_to="primary", matrix=transform)

        SceneViewer(scene, start_loop=True, callback=animation_callback)


def simplify_mesh(input_path, target_percent=0.5):
    return trimesh.load_mesh(input_path).simplify_quadric_decimation(target_percent)

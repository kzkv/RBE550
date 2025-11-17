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

    def animate_primary(self, camera_angle):
        """Animate the primary shaft (simple proof of concept)."""
        scene = self._create_scene()

        # Set initial camera
        scene.set_camera(
            angles=camera_angle.get("angles"),
            distance=camera_angle.get("distance"),
        )

        # Animation parameter (mutable so inner function can update it)
        t = {"value": 0.0}

        def animation_callback(viewer):
            """
            Called every frame by SceneViewer.
            Updates the transform of the 'primary' node to animate it.
            """
            t["value"] += 0.01

            angle = t["value"] * 2.0
            x_offset = 100.0 * np.sin(t["value"])

            position = [x_offset, 0.0, 0.0]
            rotation = [0.0, angle, 0.0]

            transform = self._primary_transform(position, rotation)

            # Update the node transform in the scene graph
            scene.graph.update(frame_to="primary", matrix=transform)

        SceneViewer(
            scene,
            start_loop=True,
            callback=animation_callback,
        )

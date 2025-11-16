import trimesh


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

    def get_primary_at(self, position):
        """Get the primary shaft positioned at the given [x, y, z]"""
        primary_copy = self.primary.copy()
        offset = position - self.primary.centroid
        primary_copy.apply_translation(offset)
        return primary_copy

    def show(self, camera_angle):
        """Show interactive 3D viewer"""
        scene = trimesh.Scene([self.case, self.primary, self.counter])

        scene.set_camera(
            angles=camera_angle.get("angles"),
            distance=camera_angle.get("distance"),
        )

        scene.show()

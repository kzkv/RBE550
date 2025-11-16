"""
Minimal High-Quality Renderer
Includes all quality improvements:
- Closed cylinder caps
- Proper aspect ratio
- Smooth rendering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from geometry import Transmission


class Renderer:
    """Minimal renderer with quality settings"""

    def __init__(self, transmission: Transmission, resolution: int = 30):
        self.transmission = transmission
        self.resolution = resolution

    def _plot_cylinder(self, ax, center, radius, length, color="orange", alpha=0.8):
        """Plot a cylinder with closed ends along the X axis"""
        theta = np.linspace(0, 2 * np.pi, self.resolution)
        x_c, y_c, z_c = center

        # Cylinder body
        x = np.array([x_c - length / 2, x_c + length / 2])
        y = y_c + radius * np.cos(theta)
        z = z_c + radius * np.sin(theta)

        X = np.outer(x, np.ones(len(theta)))
        Y = np.outer(np.ones(2), y)
        Z = np.outer(np.ones(2), z)

        # Plot body
        ax.plot_surface(
            X,
            Y,
            Z,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        # Left cap
        Y_cap = y_c + np.outer(np.linspace(0, radius, 10), np.cos(theta))
        Z_cap = z_c + np.outer(np.linspace(0, radius, 10), np.sin(theta))
        X_cap_left = np.ones_like(Y_cap) * (x_c - length / 2)
        ax.plot_surface(
            X_cap_left,
            Y_cap,
            Z_cap,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Right cap
        X_cap_right = np.ones_like(Y_cap) * (x_c + length / 2)
        ax.plot_surface(
            X_cap_right,
            Y_cap,
            Z_cap,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

    @staticmethod
    def _plot_box_wireframe(ax, min_corner, max_corner, color="blue", linewidth=2):
        """Plot wireframe box"""

        # Create 8 vertices
        vertices = np.array(
            [
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]],
            ]
        )

        # Define 12 edges
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # verticals
        ]

        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, color=color, linewidth=linewidth)

    def _plot_bearing_bores(self, ax, color="blue", linewidth=1.5):
        """Plot bearing bore circles on front and back walls"""
        theta = np.linspace(0, 2 * np.pi, 40)
        radius = self.transmission.bearing_bore_diameter / 2
        z_center = self.transmission.bearing_bore_center_height

        # Front bearing bore (x=0)
        y_front = radius * np.cos(theta)
        z_front = z_center + radius * np.sin(theta)
        x_front = np.zeros_like(y_front)
        ax.plot3D(x_front, y_front, z_front, color=color, linewidth=linewidth)

        # Rear bearing bore (x=280)
        y_rear = radius * np.cos(theta)
        z_rear = z_center + radius * np.sin(theta)
        x_rear = np.ones_like(y_rear) * self.transmission.case_outer_dims["length"]
        ax.plot3D(x_rear, y_rear, z_rear, color=color, linewidth=linewidth)

    def _plot_pto_windows(self, ax, color="blue", linewidth=1.5):
        """Plot PTO window rectangles on side walls"""
        window_w = self.transmission.pto_window_dims["width"]
        window_h = self.transmission.pto_window_dims["height"]
        center_x = self.transmission.pto_window_center_x
        center_z = self.transmission.pto_window_center_z

        # Calculate window corners
        x_min = center_x - window_w / 2
        x_max = center_x + window_w / 2
        z_min = center_z - window_h / 2
        z_max = center_z + window_h / 2

        # Left wall window (y = -width/2)
        y_left = -self.transmission.case_outer_dims["width"] / 2
        window_left = np.array(
            [
                [x_min, y_left, z_min],
                [x_max, y_left, z_min],
                [x_max, y_left, z_max],
                [x_min, y_left, z_max],
                [x_min, y_left, z_min],  # Close the rectangle
            ]
        )
        ax.plot3D(*window_left.T, color=color, linewidth=linewidth)

        # Right wall window (y = +width/2)
        y_right = self.transmission.case_outer_dims["width"] / 2
        window_right = np.array(
            [
                [x_min, y_right, z_min],
                [x_max, y_right, z_min],
                [x_max, y_right, z_max],
                [x_min, y_right, z_max],
                [x_min, y_right, z_min],  # Close the rectangle
            ]
        )
        ax.plot3D(*window_right.T, color=color, linewidth=linewidth)

    def plot(
        self,
        ax,
        mainshaft_pos=None,
        show_mainshaft=True,
        show_countershaft=True,
        show_case=True,
        show_inner_case=True,
    ):
        """Plot the transmission"""
        # Plot case
        if show_case:
            # Outer case
            min_corner, max_corner = self.transmission.get_case_outer_bounds()
            self._plot_box_wireframe(
                ax, min_corner, max_corner, color="blue", linewidth=2
            )

            # Inner case bounds
            if show_inner_case:
                min_inner, max_inner = self.transmission.get_case_inner_bounds()
                self._plot_box_wireframe(
                    ax, min_inner, max_inner, color="blue", linewidth=1
                )

            # Bearing bores
            self._plot_bearing_bores(ax, color="darkblue", linewidth=1.5)

            # PTO windows
            self._plot_pto_windows(ax, color="darkblue", linewidth=1.5)

        # Plot countershaft (gray, stationary)
        if show_countershaft:
            for cyl in self.transmission.get_countershaft():
                self._plot_cylinder(
                    ax,
                    cyl["center"],
                    cyl["radius"],
                    cyl["length"],
                    color="gray",
                    alpha=1.0,
                )

        # Plot mainshaft (orange, movable)
        if show_mainshaft:
            if mainshaft_pos is None:
                mainshaft_pos = self.transmission.mainshaft_initial_pos

            for cyl in self.transmission.get_mainshaft_at(mainshaft_pos):
                self._plot_cylinder(
                    ax,
                    cyl["center"],
                    cyl["radius"],
                    cyl["length"],
                    color="orange",
                    alpha=1.0,
                )

        # Set labels
        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.set_zlabel("Z (mm)", fontsize=10)

        # Set limits
        ax.set_xlim([-50, 350])
        ax.set_ylim([-200, 200])
        ax.set_zlim([-200, 200])

        # Fixed aspect ratio to avoid ellipsoid renderings
        ax.set_box_aspect([1, 1, 1])

    def show(self, mainshaft_pos=None, title=None, view_angle=(15, -115)):
        """Show interactive plot"""
        matplotlib.use("macosx")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        self.plot(ax, mainshaft_pos=mainshaft_pos)

        if title:
            ax.set_title(title, fontsize=14)

        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        plt.tight_layout()
        plt.show()

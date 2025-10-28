# Tom Kazakov
# RBE 550, Assignment 5, Wildfire
import logging

import networkx as nx
import numpy as np
import pygame
from networkx.exception import NodeNotFound, NetworkXNoPath

from field import Cell
from world import World

logger = logging.getLogger(__name__)


class Wumpus:
    """Standard-issue Wumpus"""

    def __init__(self, world: 'World', preset_rows: tuple[int, int], preset_cols: tuple[int, int]):
        self.world = world
        self.location = None  # (row, col) current position
        self.goal = None  # (row, col) goal position
        self.path = []

        # Load and scale the wumpus image
        self.image = pygame.image.load('assets/wumpus.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (world.cell_dimensions, world.cell_dimensions))

        self._initialize_position(preset_rows, preset_cols)

    def _initialize_position(self, preset_rows, preset_cols) -> tuple | None:
        """
        Initialize wumpus in a random cell within the preset area where all 8 neighbors are empty.
        Args: ranges of rows and columns of the preset area
        Returns tuple (row, col) if a valid position is found, None otherwise
        """
        min_row, max_row = preset_rows
        min_col, max_col = preset_cols

        # Find all valid cells in the preset area
        valid_cells = []

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if self.world.field.has_all_empty_neighbors((row, col)):
                    valid_cells.append((row, col))

        if not valid_cells:
            return None

        # Choose a random valid cell
        chosen_location = self.world.field.rng.choice(valid_cells)
        self.location = tuple(chosen_location)
        return self.location

    def _graph_from_field(self) -> nx.Graph:
        """Build a graph from the field, removing impassable cells"""
        field = self.world.field
        G = nx.grid_2d_graph(field.grid_dimensions, field.grid_dimensions)

        # Remove obstacle, burning, and burned cells
        # TODO: remove/weigh area around firetruck
        impassable = np.isin(field.cells, [Cell.OBSTACLE, Cell.BURNING, Cell.BURNED])
        avoid = zip(*impassable.nonzero())
        G.remove_nodes_from(avoid)
        return G

    def plan_path_to(self, goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Plan a path from the current location to the goal using A*"""
        G = self._graph_from_field()

        try:
            # Euclidean distance heuristic for 8-connected grid  # TODO: is it actually 8-connected?
            euclidean_distance = lambda u, v: ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) ** 0.5
            path = nx.astar_path(G, source=self.location, target=goal, heuristic=euclidean_distance)
            logger.info(f"Path found: {len(path)} cells")
            return path
        except (NodeNotFound, NetworkXNoPath) as e:
            logger.warning(f"No path found from {self.location} to {goal}: {e}")
            return []

    def set_position(self, row: int, col: int):
        """Set the Wumpus's position"""
        if not self.world.field.in_bounds(row, col):
            return

        if self.world.field.get_cell(row, col) == Cell.EMPTY:
            self.location = (row, col)

    def set_goal(self, row: int, col: int):
        """Set a goal and plan a path to it"""
        if not self.world.field.in_bounds(row, col):
            logger.warning(f"Goal {row, col} out of bounds")
            return

        if self.world.field.get_cell(row, col) != Cell.EMPTY:
            logger.warning(f"Goal {row, col} is not empty")
            return

        self.goal = (row, col)
        logger.info(f"New goal set: {self.goal}")

        # Plan path
        self.path = self.plan_path_to(self.goal)

    def update(self):
        # Set neighbors on fire
        ignited = self.world.field.ignite_neighbors(self.location)
        if ignited:
            logger.info(f"Wumpus ignited {ignited} cells")

    def render(self):
        """Render the wumpus at its current position"""
        row, col = self.location
        x = col * self.world.cell_dimensions
        y = row * self.world.cell_dimensions
        self.world.display.blit(self.image, (x, y))

    def render_path(self):
        """Render the planned path"""
        if len(self.path) < 2:
            return

        PATH_COLOR = (200, 200, 200)
        GOAL_COLOR = (162, 32, 174)  # Wumpus color

        # Draw path lines
        cell_dim = self.world.cell_dimensions
        points = []
        for row, col in self.path:
            center_x = int((col + 0.5) * cell_dim)
            center_y = int((row + 0.5) * cell_dim)
            points.append((center_x, center_y))

        pygame.draw.lines(self.world.display, PATH_COLOR, False, points, 2)

        # Draw goal marker
        if self.goal:
            goal_row, goal_col = self.goal
            goal_x = int((goal_col + 0.5) * cell_dim)
            goal_y = int((goal_row + 0.5) * cell_dim)
            pygame.draw.circle(self.world.display, GOAL_COLOR, (goal_x, goal_y), 10, 2)

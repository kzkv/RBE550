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

EPSILON = 1e-6  # For floating-point comparisons and division-by-zero checks

# Priority calculation weights
WEIGHT_UNAFFECTED_OBSTACLES = 10.0  # Reward for potential damage
WEIGHT_BURNING_PENALTY = 5.0  # Penalty for redundant ignition
WEIGHT_PATH_DISTANCE = 5.0  # Penalty per grid cell of actual path distance
TRUCK_EXCLUSION_RADIUS = 10  # Cells - hard exclusion zone around truck

# Goal selection parameters
TOP_CANDIDATES_COUNT = 20  # Number of top cells to evaluate actual paths for (after rough filtering on Euclidean distance)
MIN_SCORE_IMPROVEMENT = 1.0  # Stop evaluating if no improvement by this threshold

"""
Automatic goal selection algorithm:
1. Calculate priorities using Euclidean distance
2. Compute actual A* path lengths for top candidates
3. Stop early if no improvement is found
4. Select the best candidate based on the sum of priorities and actual path lengths
5. Set the goal to this cell
6. Plan a path to it

Cell priorities:
    Higher score -> better target for Wumpus.
    
    Priority factors:
    + Number of unaffected obstacles in spread radius (potential damage)
    - Number of already burning cells in radius (wasted effort)
    - Proximity to firetruck (avoid getting suppressed) -> exclusion zone
    - Distance from current Wumpus location (path cost)
    
    Only empty cells that are adjacent to obstacles are considered valid targets.
    Cells near the ones burning are also excluded (there is no point in going there).
    
Goal selection:
    Uses a heuristic approach: 
    1. Calculate priorities using Euclidean distance
    2. Compute actual A* path lengths for TOP_CANDIDATES_COUNT candidates
    3. Stop early if no improvement is found (adjusted with MIN_SCORE_IMPROVEMENT)
    
Conflagration rules: 
    1. Ignites a random neighbor
    2. Only ignites if at a goal cell
    3. Can only ignite again after movement
"""


class Wumpus:
    """Standard-issue Wumpus"""

    def __init__(self, world: 'World', preset_rows: tuple[int, int], preset_cols: tuple[int, int]):
        self.world = world
        self.location = self.world.field.initialize_location(preset_rows, preset_cols)
        self.goal = None  # (row, col) goal location
        self.path = []

        # Cache cell priorities (invalidated when state changes)
        self._cached_priorities = None

        # Flag if already done the deed at the location
        self.has_ignited_at_location = False

        # Load and scale the wumpus image
        self.image = pygame.image.load('assets/wumpus.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (world.cell_dimensions, world.cell_dimensions))

    def _graph_from_field(self) -> nx.Graph:
        """Build a graph from the field, removing impassable cells"""
        field = self.world.field
        G = nx.grid_2d_graph(field.grid_dimensions, field.grid_dimensions)

        # Remove obstacle, burning, and burned cells
        impassable = np.isin(field.cells, [Cell.OBSTACLE, Cell.BURNING, Cell.BURNED])
        avoid = zip(*impassable.nonzero())
        G.remove_nodes_from(avoid)
        return G

    def plan_path_to(self, goal: tuple[int, int], graph: nx.Graph = None) -> list[tuple[int, int]]:
        """Plan a path from the current location to the goal using A*"""
        if not graph:
            graph = self._graph_from_field()

        try:
            # Euclidean distance heuristic
            euclidean_distance = lambda u, v: ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) ** 0.5
            path = nx.astar_path(graph, source=self.location, target=goal, heuristic=euclidean_distance)
            return path
        except (NodeNotFound, NetworkXNoPath) as e:
            logger.warning(f"No path found from {self.location} to {goal}: {e}")
            return []

    def calculate_cell_priorities(self) -> np.ndarray:
        """
        Calculate priority scores for all EMPTY cells that are adjacent to UNBURNED obstacles.
        Cells near the truck or near burning cells are excluded from consideration.
        Returns: 2D array of priority scores (same shape as grid)
        """
        from scipy.ndimage import binary_dilation

        field = self.world.field
        priorities = np.full((field.grid_dimensions, field.grid_dimensions), -np.inf, dtype=float)

        # Find empty cells adjacent to UNBURNED obstacles only
        obstacle_mask = (field.cells == Cell.OBSTACLE)  # Only unburned obstacles
        dilated_obstacles = binary_dilation(obstacle_mask, structure=np.ones((3, 3)))
        empty_cells = (field.cells == Cell.EMPTY)

        # Valid targets: empty cells adjacent to unburned obstacles
        valid_targets = empty_cells & dilated_obstacles

        # Exclude cells near the truck
        truck_row, truck_col = self.world.firetruck.get_location()
        row_indices, col_indices = np.meshgrid(
            np.arange(field.grid_dimensions),
            np.arange(field.grid_dimensions),
            indexing='ij'
        )
        truck_distances = np.sqrt((row_indices - truck_row) ** 2 + (col_indices - truck_col) ** 2)
        truck_exclusion = truck_distances < TRUCK_EXCLUSION_RADIUS

        # Remove truck-adjacent cells from valid targets
        valid_targets = valid_targets & ~truck_exclusion

        radius = field.spread_radius_cells
        spread_mask = field.spread_mask

        # Pre-compute Wumpus distance penalty map
        wumpus_row, wumpus_col = self.location
        wumpus_distances = np.sqrt((row_indices - wumpus_row) ** 2 + (col_indices - wumpus_col) ** 2)
        path_penalty = wumpus_distances * WEIGHT_PATH_DISTANCE

        # Iterate through only valid target cells
        for row in range(field.grid_dimensions):
            for col in range(field.grid_dimensions):
                if not valid_targets[row, col]:
                    continue

                # Extract region within the spread radius
                r_slice = slice(max(0, row - radius), min(field.grid_dimensions, row + radius + 1))
                c_slice = slice(max(0, col - radius), min(field.grid_dimensions, col + radius + 1))

                mask_slice = spread_mask[
                    radius - (row - r_slice.start):radius + (r_slice.stop - row),
                    radius - (col - c_slice.start):radius + (c_slice.stop - col)
                ]

                region = field.cells[r_slice, c_slice]

                # Count cells in radius
                unaffected_obstacles = np.sum(mask_slice & (region == Cell.OBSTACLE))
                burning_cells = np.sum(mask_slice & (region == Cell.BURNING))

                # Calculate priority score (can be negative!)
                score = 0.0
                score += unaffected_obstacles * WEIGHT_UNAFFECTED_OBSTACLES
                score -= burning_cells * WEIGHT_BURNING_PENALTY
                score -= path_penalty[row, col]

                priorities[row, col] = score

        # Cache the result
        self._cached_priorities = priorities
        return priorities

    def _select_best_goal(self) -> tuple[int, int] | None:
        """
        Find the best target cell for Wumpus considering priorities.
        Returns: (row, col) of the best target, or None if no valid target
        """
        # Find top candidates
        flat_indices = np.argsort(self._cached_priorities.ravel())[::-1]  # Descending order
        top_indices = flat_indices[:TOP_CANDIDATES_COUNT]
        top_candidates = [np.unravel_index(idx, self._cached_priorities.shape) for idx in top_indices]

        # Build graph once for all path calculations
        graph = self._graph_from_field()

        # Second pass: Evaluate actual path lengths for top candidates
        best_score = -np.inf
        best_cell = None
        for row, col in top_candidates:
            # Get actual path length
            path_length = len(self.plan_path_to((row, col), graph))

            if path_length is None:
                continue  # No valid path to this cell

            # Recalculate score with actual path distance
            base_score = self._cached_priorities[row, col]

            # Remove Euclidean penalty and add the actual path penalty
            wumpus_row, wumpus_col = self.location
            euclidean_dist = np.sqrt((row - wumpus_row) ** 2 + (col - wumpus_col) ** 2)
            euclidean_penalty = euclidean_dist * WEIGHT_PATH_DISTANCE
            actual_penalty = path_length * WEIGHT_PATH_DISTANCE

            adjusted_score = base_score + euclidean_penalty - actual_penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_cell = (row, col)
            elif best_score - adjusted_score > MIN_SCORE_IMPROVEMENT:
                # Early stopping: No significant improvement expected from remaining candidates
                break

        return best_cell

    def set_location(self, row: int, col: int):
        """Set the Wumpus's location"""
        if not self.world.field.in_bounds(row, col):
            return

        if self.world.field.get_cell(row, col) == Cell.EMPTY:
            self.location = (row, col)

            # Reset ignition flag when location is set (even if it's the same location)
            self.has_ignited_at_location = False  # Make sure we shouldn't check if the location hasn't changed instead

    def set_goal(self, row: int, col: int):
        """Set a goal and plan a path to it"""
        if not self.world.field.in_bounds(row, col):
            logger.warning(f"Goal {row, col} out of bounds")
            return

        if self.world.field.get_cell(row, col) != Cell.EMPTY:
            logger.warning(f"Goal {row, col} is not empty")
            return

        self.goal = (row, col)
        self.path = self.plan_path_to(self.goal)

        # Reset ignition flag when a new goal is set
        self.has_ignited_at_location = False  # TODO: make sure goal setting should reset the flag

    def set_goal_auto(self):
        """Automatically select and set the best goal"""
        target = self._select_best_goal()

        if target:
            self.set_goal(target[0], target[1])
        else:
            logger.warning("No valid goal found for Wumpus")

    def get_location(self) -> tuple[int, int]:
        return int(self.location[0]), int(self.location[1])

    def update(self):
        # Update cache when state changes
        self._cached_priorities = self.calculate_cell_priorities()

        # Set a random neighbor on fire (only once per location and only if at a goal)
        if not self.has_ignited_at_location and self.location == self.goal:
            ignited = self.world.field.ignite_random_neighbor(self.location)
            if ignited:
                self.has_ignited_at_location = True
                logger.info(f"Wumpus ignited a cell")

    def render(self):
        """Render the wumpus at its current location"""
        row, col = self.location
        x = col * self.world.cell_dimensions
        y = row * self.world.cell_dimensions
        self.world.display.blit(self.image, (x, y))

    def render_path(self):
        """Render the planned path"""
        PATH_COLOR = (200, 200, 200)
        GOAL_COLOR = (162, 32, 174)  # Wumpus color

        if len(self.path) < 2:
            return

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

    def render_priority_heatmap(self):
        """Render priority heatmap overlay for debugging. Only shows valid targets."""
        valid_mask = self._cached_priorities > -np.inf

        if not np.any(valid_mask):
            return  # No valid targets to visualize

        # Normalize only valid priorities
        valid_priorities = self._cached_priorities[valid_mask]
        p_min, p_max = valid_priorities.min(), valid_priorities.max()

        if p_max - p_min < EPSILON:
            return

        priorities_norm = (self._cached_priorities - p_min) / (p_max - p_min)

        cell_dim = self.world.cell_dimensions
        WUMPUS_PURPLE = (162, 32, 174)

        # Render heatmap only for valid target cells
        for row in range(self.world.field.grid_dimensions):
            for col in range(self.world.field.grid_dimensions):
                if not valid_mask[row, col]:
                    continue

                # Vary alpha: high priority = opaque, low priority = transparent
                alpha = int(priorities_norm[row, col] * 200)

                x, y = col * cell_dim, row * cell_dim
                surf = pygame.Surface((cell_dim, cell_dim), pygame.SRCALPHA)
                surf.fill((*WUMPUS_PURPLE, alpha))
                self.world.display.blit(surf, (x, y))

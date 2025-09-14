# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT for the intricate numpy and scipy syntax on masking

import numpy as np
import networkx as nx
from networkx.exception import NodeNotFound, NetworkXNoPath
from scipy.ndimage import binary_dilation

from world import EMPTY, WALL, HERO, ENEMY, HUSK, GOAL
from world import World, GRID_SIZE

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# For enemy reckless path planning, all cells "available"
# Using global grid size const to instantiate this graph only once
G_abstract = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)


def graph_from_grid(grid_size, grid: np.ndarray):
    G = nx.grid_2d_graph(grid_size, grid_size)  # 4-connected, square
    impassable = np.isin(grid, [WALL, HUSK])
    enemies = np.isin(grid, [ENEMY])  # actual cells taken by enemies

    # Additional masking to avoid the areas too close to the enemies
    # I found this redundant, as 10 enemies don't seem to give the hero too much trouble most of the time.
    # 3x3 masking creates too many conditions when there is no valid path to the goal.
    # An alternate approach would be to bias the planner against the paths closer to the enemies.
    mask = np.ones((3, 3), bool)
    enemies = binary_dilation(enemies, structure=mask)  # added 3x3 mask

    avoid = zip(*np.where(impassable + enemies))  # TODO: add exclusion zones around enemies?
    G.remove_nodes_from(map(tuple, avoid))
    return G


def get_heros_journey(world: World) -> list[tuple[int, int]]:
    G = graph_from_grid(world.grid_size, world.grid)
    try:
        return nx.astar_path(G, (world.hero_y, world.hero_x), (world.goal_y, world.goal_x))
    except (NodeNotFound, NetworkXNoPath) as e:
        logger.debug(f"No path found: {e}")
        return []

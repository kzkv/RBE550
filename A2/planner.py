import numpy as np
import networkx as nx

from world import EMPTY, WALL, HERO, ENEMY, HUSK, GOAL
from world import World, GRID_SIZE

# For enemy reckless path planning, all cells "available"
# Using global grid size const to instantiate this graph only once
G_abstract = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)


def graph_from_grid(grid_size, grid: np.ndarray):
    G = nx.grid_2d_graph(grid_size, grid_size)  # 4-connected, square
    IMPASSABLE = [WALL, ENEMY, HUSK]
    blocked = zip(*np.where(np.isin(grid, IMPASSABLE)))  # TODO: add exclusion zones around enemies?
    G.remove_nodes_from(map(tuple, blocked))
    return G


def get_heros_journey(world: World):
    G = graph_from_grid(world.grid_size, world.grid)
    return nx.astar_path(G, (world.hero_y, world.hero_x), (world.goal_y, world.goal_x))

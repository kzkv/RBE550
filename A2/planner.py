import numpy as np
import networkx as nx

from world import EMPTY, WALL, HERO, ENEMY, HUSK, GOAL
from world import World
from render import render_grid

world = World(grid_size=64, rho=0.20)

G_abstract = nx.grid_2d_graph(world.grid_size,
                              world.grid_size)  # For enemy reckless path planning, all cells "available"


def graph_from_grid(grid: np.ndarray):
    G = nx.grid_2d_graph(world.grid_size, world.grid_size)  # 4-connected, square
    IMPASSABLE = [WALL, ENEMY, HUSK]
    blocked = zip(*np.where(np.isin(grid, IMPASSABLE)))  # TODO: add exclusion zones around enemies?
    G.remove_nodes_from(map(tuple, blocked))
    return G


def get_enemy_path(start, goal):
    return nx.astar_path(G_abstract, start, goal)


def get_heros_journey():
    G = graph_from_grid(world.grid)
    return nx.astar_path(G, (world.hero_y, world.hero_x), (world.goal_y, world.goal_x))

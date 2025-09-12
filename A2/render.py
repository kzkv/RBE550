# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to make the most laconic and simple implementation of my Dwarf Fortress ambitions

import numpy as np
from blessed import Terminal
from world import EMPTY, WALL, HERO, ENEMY, HUSK, GOAL

term = Terminal()

# Glyphs (need more dwarfs)
GLYPHS = {
    EMPTY: term.gray93("█"),
    WALL: term.gray50("▒"),
    HERO: term.black_on_gray93("●"),
    ENEMY: term.red1_on_gray93("▲"),
    HUSK: term.brown(""),
    GOAL: term.green("◎"),
    # TODO: teleports @ and wumpus ☺
}


def render_grid(grid: np.ndarray):
    print(end="\n")
    H, W = grid.shape
    for y in range(H):
        row = "".join(GLYPHS[int(v)] for v in grid[y])
        # print(term.on_gray99(row))
        print(row)
    print(end="\n")


def render_stats(counts):
    heroes, enemies, husks, wumpi = counts
    print(f"HEROES: {heroes:2d}   ENEMIES: {enemies:2d}   HUSKS: {husks:2d}   WUMPI: {wumpi:2d}")
    print(end="\n" * 2)

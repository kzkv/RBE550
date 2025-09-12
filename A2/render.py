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
    HUSK: term.white_on_brown("%"),
    GOAL: term.green("◎"),
    # TODO: teleports @ and wumpus ☺
}


def render_grid(grid: np.ndarray):
    print(term.clear)
    H, W = grid.shape
    for y in range(H):
        row = "".join(GLYPHS[int(v)] for v in grid[y])
        print(row)


def render_stats(counts):
    heroes, enemies, husks, wumpi = counts
    print(f"\nHEROES: {heroes:2d}   ENEMIES: {enemies:2d}   HUSKS: {husks:2d}   WUMPI: {wumpi:2d}")


def render_game_over():
    print(term.red1_on_black(" YOU DIED "), end="\n" * 2)
    # https://www.youtube.com/watch?v=-ZGlaAxB7nI


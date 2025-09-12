# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to make the most laconic and simple implementation of my Dwarf Fortress ambitions

import numpy as np
from blessed import Terminal
from world import EMPTY, WALL, HERO, ENEMY, HUSK, GOAL
from time import sleep

term = Terminal()

# Glyphs (need more dwarfs)
# GLYPHS = {
#     EMPTY: term.on_gray93(" "),
#     WALL: term.on_gray60(" "),
#     HERO: term.black_on_gray93("●"),
#     ENEMY: term.red1_on_gray93("▲"),
#     HUSK: term.white_on_brown("x"),
#     GOAL: term.white_on_green("◎"),
#     # TODO: teleports @ and wumpus ☺
# }

# Glyphs dict (needs more dwarfs); defines overridable attributes separate from the character
GLYPHS = {
    EMPTY: (term.on_gray93, " "),
    WALL: (term.on_gray60, " "),
    HERO: (term.black_on_gray93, "●"),
    ENEMY: (term.red1_on_gray93, "▲"),
    HUSK: (term.white_on_brown, "x"),
    GOAL: (term.white_on_green, "◎"),
}


def render_grid(grid: np.ndarray, path: list):
    print(term.clear)
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            attrs, char = GLYPHS[int(grid[y, x])]
            if (y, x) in path:  # Highlight the path
                cell = f"{term.black_on_lightgreen}{char}{term.normal}"
            else:
                cell = f"{attrs}{char}{term.normal}"
            print(cell, end="")
        print()


def render_stats(counts):
    heroes, enemies, husks, wumpi = counts
    print(f"\nHEROES: {heroes:2d}   ENEMIES: {enemies:2d}   HUSKS: {husks:2d}   WUMPI: {wumpi:2d}")


def render_game_over():
    print(term.red1_on_black("          \n YOU DIED \n          "), end="\n" * 2)
    # TODO: add the Dark Souls sound effect https://www.youtube.com/watch?v=-ZGlaAxB7nI

#
# def render_path(path):
#     for p in path:
#         print(p)
#         term.move_xy(*p)
#         sleep(1)

# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to make the most laconic and simple implementation of my Dwarf Fortress ambitions

import numpy as np
from blessed import Terminal

from world import World, EMPTY, WALL, HERO, ENEMY, HUSK, GOAL, GRAVE, WUMPUS


# Glyphs dict (needs more dwarfs); defines overridable attributes separate from the character
def get_glyphs(term: Terminal):
    return {
        EMPTY: (term.on_gray93, " "),
        WALL: (term.on_gray60, " "),
        HERO: (term.black_on_gray93, "."),
        ENEMY: (term.red1_on_gray93, "▲"),
        HUSK: (term.white_on_gray25, "_"),
        GOAL: (term.white_on_green, "▄"),
        GRAVE: (term.red1_on_black, "✝"),
        WUMPUS: (term.gray93_on_gray93, "w"),
    }

def render_grid(term: Terminal, grid: np.ndarray, path: list):
    GLYPHS = get_glyphs(term)

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


def render_stats(world: World):
    heroes, enemies, husks, wumpi = world.calculate_stats()
    print(
        f"\nHEROES: {heroes:1d}   ENEMIES: {enemies:2d}   HUSKS: {husks:2d}   WUMPI: {wumpi:1d}   TELEPORTS: {world.teleports:1d}")


def render_game_over(term: Terminal):
    print(term.red1_on_black("          \n YOU DIED "), end="\n" * 2)
    # TODO: add the Dark Souls sound effect https://www.youtube.com/watch?v=-ZGlaAxB7nI


def render_great_success(term: Terminal):
    print(term.green("\nTHE HERO, IN FACT, DIDN'T DIE. HEART EMOJI: <3"), end="\n" * 2)


def render_stalemate(term: Terminal):
    print(term.brown("\nTHE HERO WILL PROBABLY DIE OF STARVATION. SAD TIMES :("), end="\n" * 2)


def render_stop(term: Terminal):
    print(term.brown("\nTHIS COULD HAVE BEEN A NEW ANY% RECORD"), end="\n" * 2)

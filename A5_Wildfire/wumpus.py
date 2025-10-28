# Tom Kazakov
# RBE 550, Assignment 5, Wildfire

from typing import Optional

import pygame
from world import World


class Wumpus:
    """Standard-issue Wumpus"""

    def __init__(self, world: 'World', preset_rows: tuple[int, int], preset_cols: tuple[int, int]):
        self.world = world
        self.row = None
        self.col = None

        # Load and scale the wumpus image
        self.image = pygame.image.load('assets/wumpus.png')
        self.image = pygame.transform.scale(self.image, (world.cell_dimensions, world.cell_dimensions))

        self._initialize_position(preset_rows, preset_cols)

    def _initialize_position(self, preset_rows, preset_cols) -> Optional[tuple[int, int]]:
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
                if self.world.field.has_all_empty_neighbors(row, col):
                    valid_cells.append((row, col))

        if not valid_cells:
            return None

        # Choose a random valid cell
        self.row, self.col = self.world.field.rng.choice(valid_cells)
        return self.row, self.col

    def render(self):
        """Render the wumpus at its current position"""
        x = self.col * self.world.cell_dimensions
        y = self.row * self.world.cell_dimensions
        self.world.display.blit(self.image, (x, y))

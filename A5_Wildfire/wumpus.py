import pygame
from world import World

class Wumpus:
    def __init__(self, world: World):
        self.world = world

        # Load and scale the wumpus image
        self.image = pygame.image.load('assets/wumpus.png')

        # Calculate scaled size with margin
        self.image = pygame.transform.scale(self.image, (world.cell_dimensions, world.cell_dimensions))

    def render(self, row: int, col: int):
        """Render the wumpus at the specified grid cell."""
        x = col * self.world.cell_dimensions
        y = row * self.world.cell_dimensions
        self.world.display.blit(self.image, (x, y))

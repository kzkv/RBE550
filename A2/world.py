# Tom Kazakov
# RBE 550
# Assignment 2
# Obstacle generator: parametric generation for the grid of specified size filling with tetrominoes
# Refactor of the original A0 code. See A0 for the reasoning behind the tetromino placement.

import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng()

# Fundamental constants
GRID_SIZE = 64
RHO = 0.2

# Tetromino shapes
canonical_shapes = (
    [[1, 1, 1, 1]],  # straight (or I)
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 1, 1], [1, 1, 0]],  # S-skew
    [[1, 1, 0], [0, 1, 1]],  # Z-skew
    [[1, 1, 1], [0, 1, 0]]  # T
)

# Mapping of concepts to grid values
EMPTY = 0
WALL = 1
HERO = 2
ENEMY = 3
HUSK = 4
GOAL = 5
GRAVE = 6
WUMPUS = 42

# Starting counts of objects
ENEMY_COUNT = 10
TELEPORTS = 5


def rotate(shape: np.ndarray):
    """
    Outputs a random rotation of the given shape, -90 to 180 degrees.
    """
    return np.rot90(shape, rng.choice([-1, 0, 1, 2]))


class World:
    """
    We are only supporting square grids for now.
    """

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.rho = RHO
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.teleports = TELEPORTS

        # location of the hero
        self.hero_y, self.hero_x = 0, 0
        self.hero_alive = True

        # location of the goal
        self.goal_y, self.goal_x = 0, 0
        self.goal_reached = False

        # init
        self.place_obstacles()
        self.place_robots()
        self.place_goal()

    def place_obstacles(self):
        cell_target = int(self.rho * self.grid_size * self.grid_size)
        coverage = 0.0
        cell_coverage = 0

        while coverage < self.rho:
            # Calculate the batch, place at least one tetromino
            placements_count = max(1, int((cell_target - cell_coverage) / 4))
            logger.debug(f"Batch placements: {placements_count}")

            for _ in range(placements_count):
                # Get a shape
                idx = rng.integers(len(canonical_shapes))
                shape = rotate(canonical_shapes[int(idx)])
                mask = np.array(shape)

                # Place the shape in a random location using masking
                mask_width = mask.shape[1]
                mask_height = mask.shape[0]
                row = np.random.randint(0, self.grid_size - mask_height + 1)
                col = np.random.randint(0, self.grid_size - mask_width + 1)
                substitute = self.grid[row:row + mask_height, col:col + mask_width]
                substitute[:] = np.maximum(substitute, mask)

            cell_coverage = np.count_nonzero(self.grid)
            coverage = cell_coverage / self.grid_size / self.grid_size
            logger.debug(f"After-batch coverage: ~{coverage * 100:.2f}%")

        # Resulting coverage
        coverage = np.count_nonzero(self.grid) / self.grid_size / self.grid_size
        logger.debug(f"Resulting coverage: ~{coverage * 100:.2f}%")
        logger.debug(f"Target coverage rate: {self.rho * 100:.2f}%")

    def random_unoccupied_cell(self) -> tuple[int, int]:
        coords = np.argwhere(self.grid == EMPTY)
        y, x = coords[np.random.randint(len(coords))]
        return y, x

    def place_robots(self):
        self.hero_y, self.hero_x = self.random_unoccupied_cell()
        self.grid[self.hero_y, self.hero_x] = HERO

        for i in range(ENEMY_COUNT):
            self.grid[self.random_unoccupied_cell()] = ENEMY

    def place_goal(self):
        self.goal_y, self.goal_x = self.random_unoccupied_cell()
        self.grid[self.goal_y, self.goal_x] = GOAL

    def count_enemies(self):
        return np.count_nonzero(self.grid == ENEMY)

    def calculate_stats(self):
        heroes = np.count_nonzero(self.grid == HERO)
        enemies = self.count_enemies()
        husks = np.count_nonzero(self.grid == HUSK)
        wumpi = np.count_nonzero(self.grid == WUMPUS)
        return heroes, enemies, husks, wumpi

    def move_enemies(self):
        # Move enemies toward the hero; convert to husk on collision

        if not self.hero_alive:
            return

        enemy_locs = np.argwhere(self.grid == ENEMY)
        # There is an inherent order of these calculations, it might be worth to consider randomizing
        for y, x in enemy_locs:
            # Calculate the direction and new coordinates
            dy, dx = np.sign(self.hero_y - y), np.sign(self.hero_x - x)

            # Randomly select the axis to move, if both are available. This proved to be the easiest and cheapest
            # implementation. A more involved alternative would be to navigate first toward the axis with a higher
            # delta, but that was too much code to write.
            # In testing, the random selection approach looks quite natural.
            moves = []
            if dy: moves.append((y + dy, x))
            if dx: moves.append((y, x + dx))
            new_y, new_x = rng.choice(moves)

            # Check boundaries
            if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
                continue
                # I was too lazy to write unit tests for this, so I just manually tested it by placing the hero
                # at 0,0 and 63,63. Seems to be working as expected.

            cell = self.grid[new_y, new_x]
            if cell == HERO:
                self.hero_alive = False
                # Convert the hero to a grave, keep attacking enemy alive
                self.grid[new_y, new_x] = GRAVE
                logger.debug(f"Enemy {x, y} unalived the hero {new_x, new_y}")
            elif cell != EMPTY:
                # Convert to husk on collision with any other object
                # If two enemies collide, only the "current" one, which we are calculating for, will get husked
                # TODO: spawn wumpus on collision of two live enemies
                self.grid[y, x] = HUSK
                logger.debug(f"Enemy {x, y} husked itself")
            elif cell == EMPTY:
                # Move enemy into an empty cell
                self.grid[y, x] = EMPTY
                self.grid[new_y, new_x] = ENEMY

    def move_hero(self, new_loc):
        if not self.hero_alive:
            return

        if new_loc == (self.goal_y, self.goal_x):
            self.goal_reached = True

        self.grid[self.hero_y, self.hero_x] = EMPTY
        self.hero_y, self.hero_x = new_loc
        self.grid[self.hero_y, self.hero_x] = HERO

    def teleport_hero(self):
        if not self.hero_alive:
            return

        self.grid[self.hero_y, self.hero_x] = EMPTY
        self.hero_y, self.hero_x = self.random_unoccupied_cell()
        self.grid[self.hero_y, self.hero_x] = HERO

        self.teleports -= 1

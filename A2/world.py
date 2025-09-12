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


def rotate(shape: np.ndarray):
    """
    Outputs a random rotation of the given shape, -90 to 180 degrees.
    """
    return np.rot90(shape, rng.choice([-1, 0, 1, 2]))


class World:
    """
    We are only supporting square grids for now.

    """

    def __init__(self, grid_size: int, rho: float):
        self.grid_size = grid_size
        self.rho = rho
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.place_obstacles()

    def place_obstacles(self):
        canonical_shapes = (
            [[1, 1, 1, 1]],  # straight (or I)
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 1, 1], [1, 1, 0]],  # S-skew
            [[1, 1, 0], [0, 1, 1]],  # Z-skew
            [[1, 1, 1], [0, 1, 0]]  # T
        )

        cell_target = int(self.rho * self.grid_size * self.grid_size)

        coverage = 0.0
        cell_coverage = 0

        while coverage < self.rho:
            """
            Work in batches; each assumes no overlap. This can be further improved by gain modeling and a basic control loop. 
            """
            # Calculate the batch, place at least one tetromino
            placements_count = max(1, int((cell_target - cell_coverage) / 4))
            logger.info(f"Batch placements: {placements_count}")

            for _ in range(placements_count):
                # Get a shape
                idx = rng.integers(len(canonical_shapes))
                shape = rotate(canonical_shapes[int(idx)])
                mask = np.array(shape)

                # Place the shape in a random location using masking
                mask_width = mask.shape[1]
                mask_height = mask.shape[0]
                row, col = np.random.randint(0, self.grid_size - mask_height + 1), np.random.randint(0,
                                                                                                     self.grid_size - mask_width + 1)
                substitute = self.grid[row:row + mask_height, col:col + mask_width]
                substitute[:] = np.maximum(substitute, mask)
                # We could count added coverage here and increment in-flight, but I would want to unit-test this if implemented.

            cell_coverage = np.count_nonzero(self.grid)
            coverage = cell_coverage / self.grid_size / self.grid_size
            logger.info(f"After-batch coverage: ~{coverage * 100:.2f}%")

        # Resulting coverage
        coverage = np.count_nonzero(self.grid) / self.grid_size / self.grid_size
        logger.info(f"Resulting coverage: ~{coverage * 100:.2f}%")

        logger.info(f"Target coverage rate: {self.rho * 100:.2f}%")

# Tom Kazakov
# RBE 550
# Assignment 0
# Obstacle field: parametric generation for the 128x128 grid filling with tetrominoes without overlap
# Gen AI usage: ChatGPT to refresh myself on matplotlib and numpy syntax

import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import signedinteger
from numpy._typing import _32Bit, _64Bit

# Assumptions for the exercise
"""
Based on the Figure 3 as the example output:
1. Only four tetromino shapes are allowed: straight, L, skew, and T (the square is excluded).
2. Shapes can be rotated.
3. Shapes can overlap each other.
4. Every shape can be constructed using overlapping and rotated tetronimoes, but they may not stick out of the field.

This is in conflict with the "Figure 2: Example free tetromino obstacle shapes (non-standard shapes cost extra)",
which I readily acknowledge.

Based on the LaTex listing provided, I'm not entirely sure how rotation and mirroring have been accomplished.
But as the listing is referred to as a minimal implementation and not explicitly mentioned as a qualified solution,
I opt to operate using Figure 3 as the primary guidance.

Coverage clarification: "a coverage rate of 10% would place approximately 400 obstacles in the field, occupying maybe 
1600 cells out of 16384 total"
This implementation will place 10% or more obstacle cells (or another specified percentage), relying on the operative
"approximately 400" in the assignment.

This is not the only way to structure the assignment implementation, but for clarity and to limit the scope,
these are the guiding principles I will apply.
"""

# Rotate programmatically or list explicitly?
"""
I considered three ways of implementing this: 
1. List all 18 shapes. Silly, but might be actually the most laconic form. And saves tiny amount of compute in-flight.
2. Pre-compute all mirrors and rotations, but do it algorithmically. Saves on compute, not silly, but needs 
to be unit-tested or at least validated.
3. Compute on the fly. Might be the most elegant, but a tiny bit inefficient, and I would have to consider the RNG 
weights strategy to make sure straight tetrominoes (the fewest non-canonical forms) are not dominating the selection.

Key decision in shape selection is if each one of the 18 resulting shapes should be in the random selection with equal 
probability (option a). Alternative and valid option is (b): to have an expanded list of canonical shapes which includes 
mirrored ones, and then treat rotation as a separate in-flight process. 

I chose (b), and the most laconic form is combination of 1 and 3 approached: expanded canonical list + on-the-fly rot. 
The only tiny downside of this is slight bump in the number of I-tetrominoes being placed. 
"""

canonical_shapes = (
    [[1, 1, 1, 1]],  # straight (or I)
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 1, 1], [1, 1, 0]],  # S-skew
    [[1, 1, 0], [0, 1, 1]],  # Z-skew
    [[1, 1, 1], [0, 1, 0]]  # T
)


def rotate(shape: np.ndarray):
    """
    Outputs a random rotation of the given shape, -90 to 180 degrees.
    """
    return np.rot90(shape, random.choice([-1, 0, 1, 2]))


grid_size = 128
grid = np.zeros((grid_size, grid_size), dtype=int)

# Strategy to populate the grid
"""
The goal is to cover the specified rate of the field. One feasible and laconic way of doing this is to calculate the 
required number of cells, divide by four per tetromino, and adjust by an empirically found coefficient coming out of 
a statistically significant number of test runs. This was not a success for me, as the higher the rho goes, the more 
tetrominoes are overlapped.

Another approach is to place a guaranteed minimum number of shapes onto the grid assuming no overlap, and then iterate
adding more computing the coverage occasionally (batch trickling; fancy) or on every addition (brute force, but simple). 

Another concern is going outside of the field boundaries. This is not valid according to the chosen assumptions. 
"""

rho = 0.70  # EDIT THIS
print(f"Target coverage rate: {rho * 100:.2f}%")
cell_target = int(rho * grid_size * grid_size)

coverage = 0.0
cell_coverage = 0

while coverage < rho:
    """
    Work in batches; each assumes no overlap.
    """
    # calculate the batch, place at least one tetromino
    placements_count = max(1, int((cell_target - cell_coverage) / 4))
    print(f"Batch placements: {placements_count}")

    for _ in range(placements_count):
        # get a shape
        shape = rotate(random.choice(canonical_shapes))
        mask = np.array(shape)

        # place the shape in a random location using masking
        mask_width = mask.shape[1]
        mask_height = mask.shape[0]
        row, col = np.random.randint(0, grid_size - mask_height + 1), np.random.randint(0, grid_size - mask_width + 1)
        substitute = grid[row:row + mask_height, col:col + mask_width]
        substitute[:] = np.maximum(substitute, mask)

    cell_coverage = np.count_nonzero(grid)
    coverage = cell_coverage / grid_size / grid_size
    print(f"After-batch coverage: ~{coverage * 100:.2f}%")

# Resulting coverage
coverage = np.count_nonzero(grid) / grid_size / grid_size
print(f"Resulting coverage: ~{coverage * 100:.2f}%")

# Show with matplotlib
plt.figure(figsize=(10, 10))  # image size in inches (ew, why inches)
plt.imshow(grid, cmap="gray_r")
plt.axis("on")
plt.show()

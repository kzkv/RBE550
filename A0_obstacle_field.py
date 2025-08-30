# Tom Kazakov
# RBE 550
# Assignment 0
# Obstacle field: parametric generation for the 128x128 grid filling with tetrominoes without overlap
# Gen AI usage: ChatGPT to refresh myself on matplotlib and numpy syntax

"""
=== Assumptions for the exercise

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

Coverage clarification: "a coverage rate of 10% would place approximately 400
obstacles in the field, occupying maybe 1600 cells out of 16384 total"
This implementation will place 10% or more obstacle cells, relying on the operative
"approximately 400" in the assignment.
"""

import matplotlib.pyplot as plt
import numpy as np

# TODO: rotate programmatically or list explicitly?
canonical_shapes = (
    [[1, 1, 1, 1]],  # straight tetromino
    [[0, 0, 1], [1, 1, 1]],  # left-hand L
    [[0, 1, 1], [1, 1, 0]],  # skew
    [[1, 1, 1], [0, 1, 0]]
)

def rotate(shape: np.ndarray):
    return np.rot90(shape, k=1)

for _ in range(3):
    I = rotate(I)
    print(I)

grid_size = 128
grid = np.zeros((grid_size, grid_size), dtype=int)

# random filling of the array
grid[np.random.randint(0, grid_size, 1000), np.random.randint(0, grid_size, 1000)] = 1

# show with matplotlib
plt.figure(figsize=(10,10))  # image size in inches (ew)
plt.imshow(grid, cmap="gray_r")
plt.axis("off")
plt.grid(False)
plt.show()

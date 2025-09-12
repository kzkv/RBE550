# Tom Kazakov
# RBE 550
# Assignment 2
# Gen AI usage: ChatGPT to ideate the pseudo-graphics implementation tech stack

import matplotlib.pyplot as plt
from world import World
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

world = World(grid_size=64, rho=0.20)

# Show with matplotlib
plt.figure(figsize=(10, 10))  # image size in inches (ew, why inches)
plt.imshow(world.grid, cmap="gray_r")
plt.axis("on")
plt.show()

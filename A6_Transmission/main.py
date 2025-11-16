# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import matplotlib

matplotlib.use("macosx")

from geometry import Transmission
from render import Renderer

if __name__ == "__main__":
    transmission = Transmission()
    renderer = Renderer(transmission)

    renderer.show(title="SM-465 Transmission")

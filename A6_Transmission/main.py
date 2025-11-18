# Tom Kazakov
# RBE 550, Assignment 6, Transmission
# See Gen AI usage approach write-up in the report

import numpy as np

from transmission import Transmission


if __name__ == "__main__":
    transmission = Transmission()

    # Camera presets
    top_view = {"angles": [0, 0, 0], "distance": 600}
    side_view = {"angles": [0, np.radians(90), 0], "distance": 600}
    front_view = {"angles": [np.radians(90), 0, 0], "distance": 600}

    transmission.show(top_view)

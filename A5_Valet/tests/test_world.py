# Pycharm-generated and then refactored

import unittest
from A5_Valet.world import grid_to_world, world_to_grid


class TestGridToWorld(unittest.TestCase):
    """
    Assumes CELL_SIZE = 3
    x,y order is flipped to row,col and back in the helper functions to match the conventions
    """

    def test_center_of_origin_cell(self):
        result = grid_to_world(row=0, col=0)
        expected = (1.5, 1.5)
        self.assertEqual(expected, result)

    def test_center_of_cell_in_row(self):
        result = grid_to_world(row=2, col=0)
        expected = (1.5, 7.5)
        self.assertEqual(expected, result)

    def test_center_of_cell_in_column(self):
        result = grid_to_world(row=0, col=3)
        expected = (10.5, 1.5)
        self.assertEqual(expected, result)

    def test_center_of_cell_at_arbitrary_position(self):
        result = grid_to_world(row=4, col=7)
        expected = (22.5, 13.5)
        self.assertEqual(expected, result)

    def test_origin_corner(self):
        result = world_to_grid(0.0, 0.0)
        expected = (0, 0)
        self.assertEqual(expected, result)

    def test_down_right_corner_origin_cell(self):
        result = world_to_grid(2.9, 2.9)
        expected = (0, 0)
        self.assertEqual(expected, result)

    def test_upper_left_corner(self):
        result = world_to_grid(3.0, 3.0)  # round coordinates belong to the next cell
        expected = (1, 1)
        self.assertEqual(expected, result)

if __name__ == "__main__":
    unittest.main()

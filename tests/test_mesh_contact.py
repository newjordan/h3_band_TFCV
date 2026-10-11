import math
import unittest

import numpy as np

from blockout.mesh_contact import box_signed_depth


class MeshBoxTests(unittest.TestCase):
    def test_nearest_face_does_not_call_a_shallow_front_touch_deep_penetration(self):
        # A finger can be 9 mm below the top but only .2 mm inside the front.
        points = [[0, .0002, -.009], [0, .020, -.006]]
        result = box_signed_depth(points, np.eye(4), [-.01115, 0, -.018], [.01115, .15, 0])
        np.testing.assert_allclose(result, [.0002, .006], atol=1e-12)

    def test_white_key_side_gap_is_not_a_solid_surface(self):
        points = [[.01145, .05, -.005], [.010, .05, -.005]]
        result = box_signed_depth(points, np.eye(4), [-.01115, 0, -.018], [.01115, .15, 0])
        self.assertLess(result[0], 0)
        self.assertAlmostEqual(result[1], .00115)

    def test_front_top_and_bottom_rotate_together(self):
        angle = .010/.150
        c, s = math.cos(angle), math.sin(angle)
        transform = np.array([[1, 0, 0, .12], [0, c, -s, .15], [0, s, c, 0], [0, 0, 0, 1]])
        local = np.array([[0, -.12, -.017], [0, -.12, -.019], [0, -.151, -.009]])
        world = local@transform[:3, :3].T+transform[:3, 3]
        result = box_signed_depth(world, transform, [-.01115, -.15, -.018], [.01115, 0, 0])
        np.testing.assert_allclose(result, [.001, -.001, -.001], atol=1e-12)
        # The first point lies below the old unrotated bottom in world space,
        # but is correctly inside the rotating key.
        self.assertLess(world[0, 2], -.018)

    def test_raised_black_key_at_rotated_rear_edge(self):
        angle = .010/.095
        c, s = math.cos(angle), math.sin(angle)
        transform = np.array([[1, 0, 0, 0], [0, c, -s, .15], [0, s, c, 0], [0, 0, 0, 1]])
        local = np.array([[0, -.003, .006], [0, .001, .006]])
        world = local@transform[:3, :3].T+transform[:3, 3]
        np.testing.assert_allclose(box_signed_depth(world, transform, [-.00685, -.095, 0], [.00685, 0, .012]),
                                   [.003, -.001], atol=1e-12)

    def test_invalid_geometry_cannot_silently_pass(self):
        with self.assertRaises(ValueError):
            box_signed_depth([[np.nan, 0, 0]], np.eye(4), [-1]*3, [1]*3)
        scaled = np.diag([2, 1, 1, 1])
        with self.assertRaises(ValueError):
            box_signed_depth([[0, 0, 0]], scaled, [-1]*3, [1]*3)


if __name__ == "__main__":
    unittest.main()

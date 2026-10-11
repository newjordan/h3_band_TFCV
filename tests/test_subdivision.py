import unittest

import numpy as np

from blockout.subdivision import catmull_clark


class SubdivisionTests(unittest.TestCase):
    def test_open_quad_has_expected_smooth_boundary_and_center(self):
        matrix, faces = catmull_clark(4, [[0, 1, 2, 3]], levels=1)
        square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        result = matrix @ square
        np.testing.assert_allclose(result[0], [.125, .125, 0.])
        np.testing.assert_allclose(result[-1], [.5, .5, 0.])
        np.testing.assert_allclose(np.array(matrix.sum(1)).ravel(), 1.)
        self.assertEqual(faces.shape, (4, 4))

    def test_two_levels_preserve_rigid_translation(self):
        matrix, faces = catmull_clark(4, [[0, 1, 2, 3]], levels=2)
        square = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        offset = np.array([.1, .5, -.2])
        np.testing.assert_allclose(matrix @ (square + offset) - matrix @ square,
                                   np.tile(offset, (matrix.shape[0], 1)), atol=1e-12)
        self.assertEqual(faces.shape, (16, 4))

    def test_nonmanifold_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            catmull_clark(5, [[0, 1, 2], [1, 0, 3], [0, 1, 4]])

    def test_limit_projection_stays_affine_and_in_plane(self):
        matrix, _ = catmull_clark(4, [[0, 1, 2, 3]], levels=2, limit=True)
        np.testing.assert_allclose(np.asarray(matrix.sum(1)).ravel(), 1., atol=1e-12)
        self.assertGreaterEqual(matrix.data.min(), 0.)
        points = np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2], [0., 1., .2]])
        np.testing.assert_allclose((matrix @ points)[:, 2], .2)


if __name__ == '__main__':
    unittest.main()

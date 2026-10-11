import unittest

import numpy as np

from blockout.piano_geometry import PianoScene, piano_boxes, triangle_box_overlap, triangle_box_witness, triangle_box_margin


KEYS = [dict(pitch=60, x=0., w=.0235, y0=0., top=0., black=False),
        dict(pitch=61, x=.01175, w=.012, y0=.06, top=.012, black=True)]


class PianoGeometryTests(unittest.TestCase):
    def test_real_gap_and_rotating_key_solid(self):
        scene = PianoScene(KEYS, {60: 1.})
        points = np.array([[0., -.075, -.009], [.0113, -.075, -.009]])
        # Construct world points at the local middle of a white key and its gap.
        world = points @ scene.rotations[0].T + scene.origins[0]
        distance = scene.distances(world, [0])[:, 0]
        self.assertLess(distance[0], -.001)
        self.assertGreater(distance[1], 0.)
        front = np.array([0., -.15, 0.]) @ scene.rotations[0].T + scene.origins[0]
        self.assertAlmostEqual(front[2], -.15 * np.sin(.010 / .150), places=12)

    def test_signed_distance_is_euclidean_outside_corner(self):
        scene = PianoScene(KEYS)
        corner = np.array(scene.specs[0].upper)
        self.assertAlmostEqual(scene.distances([corner + [.003, .004, 0.]], [0])[0, 0], .005)
        self.assertEqual(scene.distances([corner], [0])[0, 0], 0.)

    def test_crossing_triangle_has_no_inside_vertices(self):
        triangle = np.array([[-2., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        lo, hi = np.array([-1., -1., -1.]), np.array([1., 1., 1.])
        self.assertFalse(np.any(np.all((triangle > lo) & (triangle < hi), axis=1)))
        self.assertTrue(triangle_box_overlap(triangle[None], lo, hi)[0])
        self.assertLess(triangle_box_margin(triangle[None], lo, hi)[0], 0.)
        barycentric = triangle_box_witness(triangle, lo, hi)
        point = barycentric @ triangle
        self.assertAlmostEqual(barycentric.sum(), 1.)
        self.assertTrue(np.all((point >= lo) & (point <= hi)))

    def test_separating_triangle_plane_and_touching_surface(self):
        lo, hi = np.full(3, -1.), np.full(3, 1.)
        # All coordinate bounds overlap, but the triangle plane misses the box.
        triangle = np.array([[.5, 2., 2.], [2., .5, 2.], [2., 2., .5]])
        self.assertFalse(triangle_box_overlap(triangle[None], lo, hi)[0])
        touch = np.array([[-.5, -.5, 1.], [.5, -.5, 1.], [0., .5, 1.]])
        self.assertFalse(triangle_box_overlap(touch[None], lo, hi)[0])
        self.assertTrue(triangle_box_overlap(touch[None], lo, hi, tolerance=0.)[0])
        self.assertAlmostEqual(triangle_box_margin(touch[None], lo, hi)[0], 0.)
        self.assertGreater(triangle_box_margin((touch + [0., 0., .001])[None], lo, hi)[0], 0.)

    def test_case_and_floor_are_included(self):
        specs = piano_boxes(KEYS)
        self.assertEqual({b.name for b in specs if b.pitch is None},
                         {'keybed', 'front_rail', 'cheek_l', 'cheek_r', 'fallboard', 'lid_body', 'floor'})
        with self.assertRaises(ValueError):
            PianoScene(KEYS, {60: 1.1})
        with self.assertRaises(ValueError):
            PianoScene(KEYS, {60: float('nan')})


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from blockout.triangle_contact import triangle_overlap, intersection_relation


class TriangleContactTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    def test_overlapping_bounds_do_not_certify_triangle_intersection(self):
        b = self.a + [.8, .8, 0.]
        self.assertFalse(triangle_overlap(self.a[None], b[None])[0])
        self.assertFalse(intersection_relation(self.a, b)['unexpected'])

    def test_noncoplanar_crossing_has_a_real_witness(self):
        b = np.array([[.2, .2, -1.], [.2, .2, 1.], [.2, .8, 0.]])
        r = intersection_relation(self.a, b)
        self.assertEqual(r['kind'], 'surface_crossing')
        self.assertTrue(r['unexpected'])
        self.assertGreater(r['intersection_length_mm'], 500.)
        self.assertAlmostEqual(r['witness'][2], 0.)

    def test_shared_vertex_is_not_an_exemption_for_crossing_elsewhere(self):
        b = np.array([[0., 0., 0.], [1., 0., -1.], [0., 1., 1.]])
        r = intersection_relation(self.a, b, [0, 1, 2], [0, 3, 4])
        self.assertTrue(r['unexpected'])
        self.assertGreater(r['intersection_length_mm'], 600.)
        outside = np.array([[0., 0., 0.], [-1., 0., 1.], [0., -1., 1.]])
        self.assertEqual(intersection_relation(self.a, outside, [0, 1, 2], [0, 3, 4])['kind'], 'expected_shared_feature')

    def test_shared_edge_coplanar_fold_is_rejected_but_regular_adjacency_is_allowed(self):
        folded = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .4, 0.]])
        r = intersection_relation(self.a, folded, [0, 1, 2], [0, 1, 3])
        self.assertTrue(r['unexpected'])
        self.assertGreater(r['intersection_area_mm2'], 100000.)
        regular = folded.copy()
        regular[2, 1] = -.4
        self.assertEqual(intersection_relation(self.a, regular, [0, 1, 2], [0, 1, 3])['kind'], 'expected_shared_feature')

    def test_coplanar_containment_and_duplicate_faces_are_not_discarded(self):
        r = intersection_relation(self.a, self.a * .3 + [.1, .1, 0.])
        self.assertEqual(r['kind'], 'coplanar_area_overlap')
        self.assertGreater(r['intersection_area_mm2'], 40000.)
        self.assertEqual(intersection_relation(self.a, self.a, [0, 1, 2], [0, 1, 2])['kind'], 'duplicate_face')

    def test_world_frame_does_not_change_the_crossing(self):
        from blockout.rig.hand import rotvec
        b = np.array([[.2, .2, -1.], [.2, .2, 1.], [.2, .8, 0.]])
        rotation = rotvec(np.array([.2, -.7, .4]))
        r = intersection_relation(self.a @ rotation.T + [2., 3., -4.], b @ rotation.T + [2., 3., -4.])
        self.assertEqual(r['kind'], 'surface_crossing')
        self.assertAlmostEqual(r['intersection_length_mm'], 600., places=7)

    def test_broadphase_tolerance_does_not_invent_a_crossing_of_nearby_planes(self):
        b = self.a + [.01, .01, 1e-8]
        self.assertTrue(triangle_overlap(self.a[None], b[None])[0])
        self.assertFalse(intersection_relation(self.a, b)['unexpected'])

    def test_invalid_and_degenerate_triangles_fail_closed(self):
        for bad in (np.zeros((3, 3)), np.full((3, 3), float('nan')), np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])):
            with self.assertRaises(ValueError):
                triangle_overlap(self.a[None], bad[None])

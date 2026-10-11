import unittest

import numpy as np

from blockout.key_shapes import NotchedPianoScene
from blockout.skin_contacts import KeyTop, key_support


class KeyTopTests(unittest.TestCase):
    def setUp(self):
        keys = [dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False),
                dict(pitch=61, x=.01175, w=.0137, y0=.055, l=.095, top=.013, black=True),
                dict(pitch=62, x=.0235, w=.0235, y0=0., l=.15, top=0., black=False)]
        self.scene = NotchedPianoScene(keys, {60: .8, 61: .6})
        self.ids = [i for i,s in enumerate(self.scene.specs) if s.pitch == 60]
        self.top = KeyTop(self.scene, 60)

    def world(self, x, y):
        i = self.ids[0]
        return np.array([x, y, self.scene.upper[i, 2]]) @ self.scene.rotations[i].T + self.scene.origins[i]

    def test_connected_head_stem_seam_is_interior_not_a_forbidden_contact_gap(self):
        head, stem = self.ids
        x = self.scene.lower[stem, 0] + .003
        seam = self.scene.upper[head, 1]
        for offset in (-1e-6, 0., 1e-6):
            point = self.world(x, seam + offset)
            self.assertGreater(self.top.margin(point), .0029)
            self.assertTrue(key_support(point[None], [0], 60, self.scene)['supported'])

    def test_cutout_and_front_outside_remain_outside_actual_top(self):
        head, stem = self.ids
        point = self.world(self.scene.upper[stem, 0] + .002, self.scene.lower[stem, 1] + .01)
        self.assertAlmostEqual(self.top.margin(point), -.002, places=12)
        self.assertFalse(key_support(point[None], [0], 60, self.scene)['supported'])
        self.assertAlmostEqual(self.top.margin(self.world(0., self.scene.lower[head, 1] - .001)), -.001, places=12)

    def test_hinged_boundary_distance_preserves_physical_margin(self):
        head = self.ids[0]
        point = self.world(self.scene.lower[head, 0] + .00001, self.scene.lower[head, 1] + .01)
        self.assertAlmostEqual(self.top.margin(point), .00001, places=12)
        self.assertAlmostEqual(self.top.margin(self.world(self.scene.lower[head, 0], self.scene.lower[head, 1] + .01)), 0., places=12)

    def test_invalid_pitch_or_nonfinite_point_fails_closed(self):
        with self.assertRaises(ValueError):
            KeyTop(self.scene, 70)
        with self.assertRaises(ValueError):
            self.top.margin([np.nan, 0., 0.])

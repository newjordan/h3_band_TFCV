import unittest

import numpy as np

from blockout.piano_geometry import PianoScene
from blockout.skin_contacts import key_support


class ActualSkinSupportTests(unittest.TestCase):
    def setUp(self):
        self.keys = [dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False),
                     dict(pitch=61, x=.01175, w=.014, y0=.06, l=.09, top=.012, black=True)]

    def test_support_uses_moving_top_and_rejects_underneath_or_off_key(self):
        scene = PianoScene(self.keys, {60: 1.})
        box = 0
        local = np.array([[0., -.12, .00005], [0., -.12, -.025], [.04, -.12, .00005]])
        points = local @ scene.rotations[box].T + scene.origins[box]
        self.assertTrue(key_support(points, [0], 60, scene)['supported'])
        self.assertFalse(key_support(points, [1], 60, scene)['supported'])
        self.assertFalse(key_support(points, [2], 60, scene)['supported'])
        self.assertAlmostEqual(key_support(points, [0], 60, scene)['top_gap_mm'], .05)

    def test_white_contact_occluded_by_tilted_black_key_cannot_count(self):
        scene = PianoScene(self.keys, {60: .6, 61: 1.})
        local = np.array([[.007, -.05, .00005], [0., -.12, .00005]])
        points = local @ scene.rotations[0].T + scene.origins[0]
        self.assertFalse(key_support(points, [0], 60, scene)['supported'])
        result = key_support(points, [0, 1], 60, scene)
        self.assertTrue(result['supported'])
        self.assertEqual(result['vertex'], 1)

    def test_missing_surface_and_nonfinite_tolerance_are_rejected(self):
        scene = PianoScene(self.keys)
        self.assertFalse(key_support(np.empty((0, 3)), [], 60, scene)['supported'])
        with self.assertRaises(ValueError):
            key_support(np.zeros((1, 3)), [0], 60, scene, contact_gap=np.nan)


if __name__ == '__main__':
    unittest.main()

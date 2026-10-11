import unittest

import numpy as np

from blockout.key_shapes import NotchedPianoScene, key_outline, notched_boxes
from blockout.piano_geometry import PianoScene, scene_from_animation
from lab.piano_interference_audit import interference


class MechanicalKeyShapeTests(unittest.TestCase):
    def setUp(self):
        self.keys = [dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False),
                     dict(pitch=61, x=.01175, w=.0137, y0=.055, l=.095, top=.013, black=True),
                     dict(pitch=62, x=.0235, w=.0235, y0=0., l=.15, top=0., black=False)]

    def test_rectangular_interference_is_removed_over_independent_key_sweeps(self):
        self.assertTrue(interference(PianoScene(self.keys, {61: 1.})))
        for white in np.linspace(0., 1., 7):
            for black in np.linspace(0., 1., 9):
                scene = NotchedPianoScene(self.keys, {60: white, 61: black, 62: 1. - white})
                self.assertEqual(interference(scene, include_case=True), [])

    def test_front_widths_hinges_and_depth_stops_are_preserved(self):
        original = PianoScene(self.keys)
        scene = NotchedPianoScene(self.keys, {60: 1.})
        for key in self.keys:
            pieces = [p for p in scene.specs if p.pitch == key['pitch']]
            source = next(p for p in original.specs if p.pitch == key['pitch'])
            self.assertEqual(pieces[0].lower, source.lower)
            self.assertEqual(pieces[0].upper[0], source.upper[0])
            self.assertTrue(all(p.origin == source.origin and p.length == source.length for p in pieces))
            polygon = key_outline(pieces)
            signed_area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)) / 2
            expected_area = sum((p.upper[0] - p.lower[0]) * (p.upper[1] - p.lower[1]) for p in pieces)
            self.assertAlmostEqual(signed_area, expected_area, places=12)
        with self.assertRaises(ValueError):
            NotchedPianoScene(self.keys, {61: 1.01})

    def test_geometry_identity_is_shared_and_unknown_shapes_fail(self):
        source = dict(keyboard=self.keys, geometry=dict(key_shape='notched'))
        scene = scene_from_animation(source, dict(keys={61: 1.}))
        self.assertIsInstance(scene, NotchedPianoScene)
        self.assertFalse(interference(scene, include_case=True))
        source['geometry']['key_shape'] = 'unknown'
        with self.assertRaises(ValueError):
            scene_from_animation(source, dict(keys={}))


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from scipy import sparse
from types import SimpleNamespace

from blockout.key_shapes import NotchedPianoScene
from blockout.rig.contact_placement import calibrated_target, surface_target
from blockout.skin_contacts import key_support
from blockout.rig.skin_contact import musical_anchors


class ContactPlacementTests(unittest.TestCase):
    def setUp(self):
        keyboard = [dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False),
                    dict(pitch=61, x=.01175, w=.0137, y0=.055, l=.095, top=.013, black=True),
                    dict(pitch=62, x=.0235, w=.0235, y0=0., l=.15, top=0., black=False)]
        self.keys = {k['pitch']: k for k in keyboard}
        self.scene = NotchedPianoScene(keyboard, {60: .8, 61: .6, 62: .2})

    def test_deep_contact_uses_the_stem_and_the_actual_moving_top(self):
        box, target, metadata = calibrated_target(4, 60, self.scene, self.keys, [61])
        self.assertEqual(self.scene.specs[box].name, 'key60_stem')
        self.assertEqual(metadata['kind'], 'white_near_black')
        self.assertAlmostEqual(target[1], .073, places=12)
        self.assertTrue(key_support(target[None], [0], 60, self.scene)['supported'])
        self.assertLess(target[2], 0.)
        chosen, current_target = surface_target(target + [0., 0., .003], 60, self.scene)
        self.assertEqual(chosen, box)
        np.testing.assert_allclose(target, current_target, atol=.0003)

    def test_thumb_front_and_black_targets_preserve_key_identity(self):
        for finger, pitch in ((1, 60), (2, 61), (5, 62)):
            box, target, _ = calibrated_target(finger, pitch, self.scene, self.keys)
            self.assertEqual(self.scene.specs[box].pitch, pitch)
            self.assertTrue(key_support(target[None], [0], pitch, self.scene)['supported'])
        with self.assertRaises(ValueError):
            surface_target([0., 0., 0.], 70, self.scene)

    def test_musical_anchor_keeps_a_deep_source_contact_on_its_stem(self):
        _, target, _ = calibrated_target(4, 60, self.scene, self.keys, [61])
        skin = SimpleNamespace(bone_index={'finger4-3': 0}, weights=np.ones((2, 1)))
        vertices = np.array([target + [0., 0., -.0001], target + [0., 0., .002]])
        hand = dict(assigned_keys=[None, None, None, 60, None],
                    activation=[0., 0., 0., .8, 0.], pads=[target.tolist()] * 5)
        anchor = musical_anchors(skin, sparse.eye(2, format='csr'), vertices, hand, self.scene, self.keys)[0]
        self.assertEqual(self.scene.specs[anchor.box].name, 'key60_stem')
        self.assertEqual(anchor.vertex, 0)
        self.assertAlmostEqual(anchor.target[1], target[1], places=4)
        self.assertTrue(key_support(anchor.target[None], [0], 60, self.scene)['supported'])

    def test_pooled_depth_band_is_soft_while_the_selected_surface_remains_finite(self):
        _, target, _ = calibrated_target(4, 60, self.scene, self.keys, [61])
        skin = SimpleNamespace(bone_index={'finger4-3': 0}, weights=np.ones((1, 1)))
        hand = dict(assigned_keys=[None, None, None, 60, None],
                    activation=[0., 0., 0., .8, 0.], pads=[target.tolist()] * 5)
        hard = musical_anchors(skin, sparse.eye(1, format='csr'), target[None], hand, self.scene, self.keys,
                               placement='calibrated', active_pitches=[61])[0]
        soft = musical_anchors(skin, sparse.eye(1, format='csr'), target[None], hand, self.scene, self.keys,
                               placement='calibrated_soft', active_pitches=[61])[0]
        self.assertEqual(hard.tangential_tolerance, .002)
        self.assertIsNone(soft.tangential_tolerance)
        self.assertEqual(soft.placement['depth_iqr_mm'], [54., 93.])
        self.assertEqual(self.scene.specs[hard.box].name, self.scene.specs[soft.box].name)
        union = musical_anchors(skin, sparse.eye(1, format='csr'), target[None], hand, self.scene, self.keys,
                                placement='calibrated_soft_union', active_pitches=[61])[0]
        self.assertEqual(union.key_components, 'union')
        self.assertIsNone(union.tangential_tolerance)
        np.testing.assert_array_equal(union.target, soft.target)
        self.assertEqual(union.placement, soft.placement)

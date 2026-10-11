import json
import unittest

import numpy as np

from blockout.rig.hand import HandRig, rotvec
from blockout.rig.pose_io import animation_hand_scale, serialize_hand
from blockout.rig.skin_contact import pose_vector


class PoseIdentityTests(unittest.TestCase):
    def test_tagged_size_cannot_be_silently_overridden(self):
        self.assertEqual(animation_hand_scale({'hand_scale': .7284}, legacy_default=1.), .7284)
        self.assertEqual(animation_hand_scale({'hand_scale': .7284}, requested=.7284), .7284)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            animation_hand_scale({'hand_scale': .7284}, requested=1.)
        for scale in (0., -1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                animation_hand_scale({'hand_scale': scale})

    def test_untagged_frozen_inputs_require_explicit_size(self):
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            animation_hand_scale({})
        self.assertEqual(animation_hand_scale({}, requested=1.), 1.)
        self.assertEqual(animation_hand_scale({}, requested=.7284), .7284)

    def test_json_roundtrip_preserves_boundary_angles_and_physical_landmarks(self):
        rig = HandRig('R')
        reference, _ = rig.sk.rest_root()
        rotation = rotvec(np.array([.11111111, -.22222222, .33333333])) @ reference
        root = np.array([.15123456789, -.113456789, .053456789])
        for q in (rig.lo, rig.hi):
            vector = np.r_[root, np.zeros(3), q]
            _, _, points = rig.points(vector, rotation)
            pads = rig.effectors(points, rig._rots)
            packed = serialize_hand(rig, vector, rotation, root, points, pads, [.8] * 5,
                                    [60, None, None, None, None], root)
            restored = json.loads(json.dumps(packed))
            np.testing.assert_array_equal(pose_vector(rig, restored)[6:], q)
            self.assertEqual(restored['wrist'], restored['root']['pos'])
            _, _, reconstructed = rig.points(pose_vector(rig, restored), np.asarray(restored['root']['rot']))
            np.testing.assert_allclose(restored['fingers'], reconstructed, atol=1e-14, rtol=0.)
            # The legacy rest skeleton itself stores rounded frames. This
            # export preserves its input rotation without further quantizing
            # it; it does not silently change the underlying skeleton.
            np.testing.assert_array_equal(restored['root']['rot'], rotation)
            original_error = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            restored_error = np.max(np.abs(np.asarray(restored['root']['rot']).T @ restored['root']['rot'] - np.eye(3)))
            self.assertEqual(restored_error, original_error)

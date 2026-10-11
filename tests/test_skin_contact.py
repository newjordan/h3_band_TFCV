import unittest

import numpy as np
from scipy import sparse
from types import SimpleNamespace
from unittest.mock import patch

from blockout.performer import UPPER_ARM, FOREARM
from blockout.rig.skin_contact import continuous_elbow, musical_anchors, restore_pose
from blockout.piano_geometry import PianoScene


class ElbowProjectionTests(unittest.TestCase):
    def test_wrist_correction_keeps_both_arm_lengths(self):
        shoulder = np.array([0., -.4, .3])
        reference = np.array([.2, -.3, 0.])
        for wrist in (np.array([.1, -.1, .05]), np.array([.13, -.12, .06])):
            elbow = continuous_elbow(shoulder, wrist, reference)
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), UPPER_ARM, places=12)
            self.assertAlmostEqual(np.linalg.norm(wrist - elbow), FOREARM, places=12)
            self.assertLess(np.linalg.norm(continuous_elbow(shoulder, wrist + [1e-7, 0., 0.], reference)
                                           - elbow), 1e-6)

    def test_unreachable_wrist_remains_unreachable(self):
        shoulder, wrist = np.zeros(3), np.array([0., 1., 0.])
        elbow = continuous_elbow(shoulder, wrist, np.array([.2, .2, 0.]))
        self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), UPPER_ARM)
        self.assertGreater(np.linalg.norm(wrist - elbow), FOREARM)


class SkinAnchorTests(unittest.TestCase):
    def test_uses_supporting_skin_instead_of_point_nearest_internal_pad(self):
        keys = {60: dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)}
        scene = PianoScene(list(keys.values()))
        vertices = np.array([[0., .03, .0015], [-.004, .03, -.0001]])
        skin = SimpleNamespace(bone_index={'finger1-3': 0}, weights=np.ones((2, 1)))
        hand = dict(assigned_keys=[60, None, None, None, None],
                    activation=[.8, 0., 0., 0., 0.], pads=[[0., .03, 0.]] * 5)
        operator = sparse.eye(2, format='csr')
        old = musical_anchors(skin, operator, vertices, hand, scene, keys, mode='nearest_pad')[0]
        current = musical_anchors(skin, operator, vertices, hand, scene, keys)[0]
        self.assertEqual(old.vertex, 0)
        self.assertEqual(current.vertex, 1)
        np.testing.assert_allclose(current.target, [-.004, .03, 0.])

    def test_an_off_key_pad_does_not_create_a_supported_note(self):
        keys = {60: dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)}
        scene = PianoScene(list(keys.values()))
        skin = SimpleNamespace(bone_index={'finger1-3': 0}, weights=np.ones((1, 1)))
        hand = dict(assigned_keys=[60], activation=[1.], pads=[[0., -.02, -.01]])
        self.assertEqual(musical_anchors(skin, sparse.eye(1), np.array([[0., .03, 0.]]), hand, scene, keys), [])

    def test_required_held_note_keeps_a_contact_requirement_when_pad_slips_off(self):
        keys = {60: dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)}
        scene = PianoScene(list(keys.values()), {60: 1.})
        skin = SimpleNamespace(bone_index={'finger1-3': 0}, weights=np.ones((1, 1)))
        hand = dict(assigned_keys=[60], activation=[.8], pads=[[0., -.001, -.01]])
        anchors = musical_anchors(skin, sparse.eye(1, format='csr'), np.array([[0., .003, -.009]]),
                                  hand, scene, keys, required_pitches={60})
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].pitch, 60)


class RejectedPoseTests(unittest.TestCase):
    def test_optimizer_success_cannot_export_an_out_of_bounds_pose(self):
        from tests.test_rig_motion import fixture
        from blockout.rig.hand import HandRig
        from blockout.skin import Skin
        from blockout.subdivision import catmull_clark
        rig, skin = HandRig('R'), Skin('R')
        rotation, _ = rig.sk.rest_root()
        frame = fixture(rig, [.15, -.1, .5], rotation, rig.q_rest, 0.)
        hand = frame['hands']['R']
        surface = catmull_clark(len(skin.vertices), skin.faces, 1, limit=True)
        scene = PianoScene([dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)])
        bad = np.r_[np.zeros(6), rig.q_rest]
        bad[6] = rig.lo[0] - .01
        fake = SimpleNamespace(x=bad, success=True, message='fake success', nit=1)
        with patch('blockout.rig.skin_contact.minimize', return_value=fake):
            report, output = restore_pose(rig, skin, surface, hand, scene,
                frame['body']['shoulders']['R'], None, None, 1. / 24., passes=1)
        self.assertFalse(report['accepted'])
        self.assertFalse(report['candidate_exported'])
        self.assertGreater(report['configuration_bounds_violation'], .009)
        self.assertEqual(output, hand)

    def test_nonfinite_descent_cannot_poison_a_bounded_initialization(self):
        from tests.test_rig_motion import fixture
        from blockout.rig.hand import HandRig
        from blockout.skin import Skin
        from blockout.subdivision import catmull_clark
        rig, skin = HandRig('R'), Skin('R')
        rotation, _ = rig.sk.rest_root()
        frame = fixture(rig, [.15, -.1, .5], rotation, rig.q_rest, 0.)
        hand = frame['hands']['R']
        surface = catmull_clark(len(skin.vertices), skin.faces, 1, limit=True)
        scene = PianoScene([dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)])
        fake = SimpleNamespace(x=np.full(26, np.nan), success=False, message='nonfinite descent', nit=1)
        with patch('blockout.rig.skin_contact.minimize', return_value=fake):
            report, output = restore_pose(rig, skin, surface, hand, scene,
                frame['body']['shoulders']['R'], None, None, 1. / 24., passes=1,
                allow_wrist=False, feasibility_iterations=2)
        self.assertTrue(np.isfinite(report['final_penetration_mm']))
        self.assertEqual(report['configuration_bounds_violation'], 0.)
        self.assertEqual(report['history'][0]['candidate_source'], 'bounded feasibility seed')
        if report['accepted']:
            q = np.array([output['angles'].get(bone, {}).get(dof, 0.)
                          for chain in rig.chains for bone, dof, *_ in chain.dofs])
            self.assertTrue(np.all(q >= rig.lo) and np.all(q <= rig.hi))
        else:
            self.assertEqual(output, hand)

    def test_feasible_optimized_pose_is_kept_when_positive_slack_decreases(self):
        from tests.test_rig_motion import fixture
        from blockout.rig.hand import HandRig
        from blockout.skin import Skin
        from blockout.subdivision import catmull_clark
        rig, skin = HandRig('R'), Skin('R')
        rotation, _ = rig.sk.rest_root()
        frame = fixture(rig, [.15, -.1, .5], rotation, rig.q_rest, 0.)
        # The old fixture pointed the hand backward against its forearm,
        # folding the wrist skin. Use a geometrically clear arm/hand posture
        # for this test of retaining an otherwise feasible optimized step.
        from blockout.skin import rotation_between
        forward = rotation @ rig.sk.R0['wrist'].T @ rig.M_rest[1]
        rotation = rotation_between(forward, np.asarray(frame['hands']['R']['wrist'])
                                    - frame['hands']['R']['elbow']) @ rotation
        frame = fixture(rig, [.15, -.1, .5], rotation, rig.q_rest, 0.)
        hand = frame['hands']['R']
        surface = catmull_clark(len(skin.vertices), skin.faces, 1, limit=True)
        scene = PianoScene([dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)])
        candidate = np.r_[4., np.zeros(5), rig.q_rest]
        fake = SimpleNamespace(x=candidate, success=True, message='feasible optimized step', nit=1)
        with patch('blockout.rig.skin_contact.minimize', return_value=fake), \
             patch.object(rig, 'finger_clearances', side_effect=lambda x, r: np.array([.01 - 2. * (x[0] - .15)])):
            report, output = restore_pose(rig, skin, surface, hand, scene,
                frame['body']['shoulders']['R'], None, None, 1. / 24., passes=1,
                feasibility_iterations=2)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['history'][0]['candidate_source'], 'SLSQP')
        self.assertAlmostEqual(output['wrist'][0], .154)

    def test_proxy_spacing_and_optimizer_success_cannot_hide_folded_wrist_skin(self):
        from tests.test_rig_motion import fixture
        from blockout.rig.hand import HandRig
        from blockout.skin import Skin
        from blockout.subdivision import catmull_clark
        rig, skin = HandRig('R'), Skin('R')
        rotation, _ = rig.sk.rest_root()
        frame = fixture(rig, [.15, -.1, .5], rotation, rig.q_rest, 0.)
        hand = frame['hands']['R']
        surface = catmull_clark(len(skin.vertices), skin.faces, 1, limit=True)
        scene = PianoScene([dict(pitch=60, x=0., w=.0235, y0=0., l=.15, top=0., black=False)])
        candidate = np.r_[np.zeros(6), rig.q_rest]
        fake = SimpleNamespace(x=candidate, success=True, message='proxy-feasible', nit=1)
        with patch('blockout.rig.skin_contact.minimize', return_value=fake), \
             patch.object(rig, 'finger_clearances', return_value=np.array([.01])):
            report, output = restore_pose(rig, skin, surface, hand, scene,
                frame['body']['shoulders']['R'], None, None, 1./24., passes=1,
                mesh_self_constraints=False)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['final_penetration_mm'], 0.)
        self.assertEqual(report['configuration_bounds_violation'], 0.)
        self.assertGreater(report['proxy_self_clearance_mm'], 0.)
        self.assertFalse(report['mesh_self']['surface_intersection_clear'])
        self.assertFalse(report['candidate_exported'])
        self.assertEqual(output, hand)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from blockout import piano
from blockout.contact import activation, key_depth, on_key


class PianoContactTests(unittest.TestCase):
    def setUp(self):
        self.keys = {k['pitch']: k for k in piano.keyboard()}

    def test_key_front_middle_and_rear_use_same_hinged_depth(self):
        for pitch in (60, 61):
            k = self.keys[pitch]
            for u in (0.05, 0.5, 0.98):
                y = k['y0'] + u * k['l']
                z = k['top'] - 0.65 * piano.KEY_TRAVEL * (1 - u)
                self.assertAlmostEqual(key_depth(pitch, [k['x'], y, z], self.keys, piano.KEY_TRAVEL), 0.65)

    def test_no_force_outside_key_or_at_pivot(self):
        for p in (60, 61):
            k = self.keys[p]
            for xyz in ([k['x'] + k['w'] / 2 + 0.0001, k['y0'] + 0.01, -0.1],
                        [k['x'], k['y0'] - 0.0001, -0.1],
                        [k['x'], k['y0'] + k['l'], -0.1]):
                self.assertEqual(key_depth(p, xyz, self.keys, piano.KEY_TRAVEL), 0.0)

    def test_black_key_occludes_white_top(self):
        x = self.keys[61]['x'] - 0.001
        self.assertTrue(on_key(60, [x, 0.03, 0], self.keys))
        self.assertFalse(on_key(60, [x, 0.08, 0], self.keys))
        self.assertTrue(on_key(61, [x, 0.08, 0], self.keys))

    def test_white_key_gap_and_pivot_have_no_invented_surface(self):
        key = self.keys[60]
        self.assertFalse(on_key(60, [key['x'] + key['w'] / 2 - .0003, .02, -.01], self.keys))
        self.assertTrue(on_key(60, [key['x'] + key['w'] / 2 - .0007, .02, -.01], self.keys))
        self.assertEqual(piano.key_lever(60, piano.WHITE_L), 0)
        self.assertAlmostEqual(piano.key_lever(60, piano.WHITE_L * .99), .01)

    def test_key_bed_is_solid_in_both_collision_modes(self):
        sample = (2, .8, np.array([0., .025, -.018]), -.02, .003)
        samples = [[sample], [], [], [], []]
        for soft in (True, False):
            with patch.object(piano, 'SOFT_ENV', soft):
                env = piano._key_env({60: 1.}, [True] * 5)
                r = np.asarray(env(samples, np.eye(3), np.zeros(3)))
                cost = float(r @ r) if r.ndim else float(r)
                with patch.object(piano, 'KEYBED_STOP_W', 0):
                    r0 = np.asarray(env(samples, np.eye(3), np.zeros(3)))
                    old_cost = float(r0 @ r0) if r0.ndim else float(r0)
                self.assertGreater(cost, 20 * old_cost)

    def test_pressing_finger_gets_no_clearance_inside_neighbor_key(self):
        # Thumb is assigned C, but its skin is over D. Being a pressing thumb
        # must not grant the old four-millimetre exemption through D's top.
        sample = (2, .8, np.array([piano.key_x(62), .025, -.001]), -.002, .002)
        samples = [[sample], [], [], [], []]
        for soft in (True, False):
            with patch.object(piano, 'SOFT_ENV', soft):
                pressed = piano._key_env({60: 1.}, [True] + [False] * 4, pressing_keys=[60, None, None, None, None])
                free = piano._key_env({60: 1.}, [False] * 5, pressing_keys=[None] * 5)
                np.testing.assert_allclose(pressed(samples, np.eye(3), np.zeros(3)),
                                           free(samples, np.eye(3), np.zeros(3)))

    def test_actual_key_depth_has_no_hole_for_a_missed_note(self):
        keys = {k['pitch']: k for k in piano.keyboard()}
        pads = [np.array([piano.key_x(62), .025, -.02])] + [np.zeros(3)] * 4
        hands = {'R': {'act': [[(60, 0., piano.TIP_R[0]), None, None, None, None]]}}
        solved = {'R': (None, None, None, None, None, pads)}
        depth, press = piano._solved_key_depth(hands, 0, solved, keys)
        self.assertEqual(depth, {})
        self.assertEqual(press, {})

    def test_other_thumb_cannot_hide_inside_wrist_envelope(self):
        rig = piano._rig('R')
        rr, rp = rig.sk.rest_root()
        elbow = rp + np.array([0., -.3, 0.])
        empty = [[] for _ in range(5)]
        other = (np.array([rp + [0., -.02, 0.]]), np.array([.007]))
        with patch.object(piano, 'DISTAL_ARM_CONTACT', True):
            c, r = piano._hand_blobs(empty, rr, rp, rig, elbow)
            # Continuous 30mm-radius distal arm is covered even between spheres.
            for d in np.linspace(0., .09, 101):
                point = rp + [0., -d, .03]
                self.assertLessEqual(np.min(np.linalg.norm(c - point, axis=1) - r), 1e-10)
            env = piano._key_env({}, [False] * 5, other, rig, elbow=elbow)
            cost = float(np.sum(np.asarray(env(empty, rr, rp)) ** 2))
        with patch.object(piano, 'DISTAL_ARM_CONTACT', False):
            old_cost = float(np.sum(np.asarray(env(empty, rr, rp)) ** 2))
        self.assertGreater(cost, old_cost + .001)

    def test_unreachable_wrist_is_projected_before_finger_contact(self):
        from blockout import performer
        shoulder = np.array([.2, -.3, .4])
        for position in (shoulder, shoulder + [2., 0., 0.], shoulder + [.2, .1, -.1]):
            wrist = piano._reachable_wrist(position, shoulder)
            d = np.linalg.norm(wrist - shoulder)
            self.assertLess(d, performer.UPPER_ARM + performer.FOREARM)
            self.assertGreater(d, abs(performer.UPPER_ARM - performer.FOREARM))
            elbow = performer.arm_ik(shoulder, wrist, 1)
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), performer.UPPER_ARM, places=8)
            self.assertAlmostEqual(np.linalg.norm(wrist - elbow), performer.FOREARM, places=8)
        reachable = shoulder + [.2, .1, -.1]
        np.testing.assert_allclose(piano._reachable_wrist(reachable, shoulder), reachable)

    def test_unilateral_contact_and_bed_bound(self):
        self.assertEqual(key_depth(60, [0, 0.03, 0.01], self.keys, piano.KEY_TRAVEL), 0)
        self.assertEqual(key_depth(60, [0, 0.03, -0.04], self.keys, piano.KEY_TRAVEL), 1)
        self.assertEqual(key_depth(60, [float('nan'), 0, 0], self.keys, piano.KEY_TRAVEL), 0)

    def test_hold_target_respects_leverage_for_every_finger(self):
        for pitch in (60, 61):
            for finger in range(5):
                event = (1.0, 1.5, pitch, 100, True, 0.5, 1.0)
                radius = piano.TIP_R[finger]
                tip, depth = piano.finger_tip(1.2, event, np.zeros(3), radius, finger)
                bed = self.keys[pitch]['top'] - piano.KEY_TRAVEL * piano.key_lever(pitch, tip[1])
                self.assertAlmostEqual(tip[2] - radius, bed)
                self.assertEqual(depth, 1)

    def test_soft_loud_hold_and_release_stay_controlled(self):
        for touch in (0.0, 0.5, 1.0):
            e = (1.0, 1.5, 60, 100, False, touch, 1.0)
            values = np.array([activation(t, e) for t in np.linspace(0.6, 1.8, 1001)])
            self.assertTrue(np.all((values >= 0) & (values <= 1)))
            self.assertEqual(activation(1.0, e), 1.0)
            self.assertGreaterEqual(activation(1.45, e), 0.8)
            self.assertEqual(activation(1.8, e), 0)
            self.assertLess(np.max(np.abs(np.diff(values))), 0.04)
            # Short notes must release from their actual support gain.
            short = (1.0, 1.01, 60, 100, False, touch, 1.0)
            self.assertLess(abs(activation(1.01, short) - activation(1.010001, short)), 1e-5)

    def test_two_hand_collision_pass_does_not_move_free_target_twice(self):
        class Rig:
            q_rest = np.zeros(5)
            sl = np.arange(6)
            lo = np.full(5, -1.0)
            hi = np.ones(5)
            _rots = [None] * 5
            chains = []
            def __init__(self):
                self.seen = []
            def solve(self, rr, rp, targets, weights, **kwargs):
                self.seen.append(np.array(targets))
                return np.zeros(11), (rr, rp, [np.zeros((4, 3))] * 5, targets), [0] * 5
            def points(self, x, rr):
                return rr, np.zeros(3), []
        rigs = {name: Rig() for name in ('R', 'L')}
        hands = {name: {'act': [[None] * 5]} for name in rigs}
        targets = {name: np.zeros((5, 3)) for name in rigs}
        refs = {name: (np.eye(3), np.array([0.1, 0.2, 0.0])) for name in rigs}
        with patch.object(piano, '_rig', side_effect=rigs.get), \
             patch.object(piano, '_plan_root', return_value=(np.eye(3), np.zeros(3))), \
             patch.object(piano, '_key_env', return_value=None), \
             patch.object(piano, '_hand_blobs', return_value=None):
            piano._solve_frame(hands, 0, targets, {}, {}, refs=refs)
        for rig in rigs.values():
            self.assertEqual(len(rig.seen), 2)
            np.testing.assert_allclose(rig.seen[0], rig.seen[1])
            np.testing.assert_allclose(rig.seen[1][:, :2], np.tile([0.1, 0.2], (5, 1)))
        for target in targets.values():
            np.testing.assert_array_equal(target, 0)


if __name__ == '__main__':
    unittest.main()

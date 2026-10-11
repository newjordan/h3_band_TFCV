import unittest
from unittest.mock import patch

import numpy as np

from blockout import dynamics, piano
from blockout.midi import Note
from blockout.rig.composure import quiet_pad_targets, relaxed_pads, solo_wrist_reference
from blockout.rig.hand import HandRig


class PianoComposureTests(unittest.TestCase):
    def test_idle_middle_is_wrist_carried_and_assigned_targets_are_unchanged(self):
        rig = HandRig('R')
        rotation, position = rig.root_from_hand_frame(np.eye(3), np.array([0., 0., .15]))
        pads = relaxed_pads(rig, rotation, position)
        targets = pads + np.array([.03, .01, .02])
        pressing = [True, False, False, False, True]
        result = quiet_pad_targets(rig, rotation, position, targets, pressing, np.ones(5),
                                   lambda x, y, f: -1.)
        np.testing.assert_allclose(result[2], pads[2], atol=1e-12)
        np.testing.assert_array_equal(result[pressing], targets[pressing])
        shift = np.array([.04, -.02, .01])
        moved = quiet_pad_targets(rig, rotation, position + shift, targets + shift,
                                  pressing, np.ones(5), lambda x, y, f: -1.)
        np.testing.assert_allclose(moved - result, np.tile(shift, (5, 1)), atol=1e-12)
        np.testing.assert_array_equal(targets, pads + [.03, .01, .02])

    def test_free_preparation_is_bounded_and_instrument_floor_is_respected(self):
        rig = HandRig('L')
        rotation, position = rig.sk.rest_root()
        pads = relaxed_pads(rig, rotation, position)
        result = quiet_pad_targets(rig, rotation, position, pads + [.2, .1, .3],
                                  [False] * 5, np.zeros(5), lambda x, y, f: -1.)
        np.testing.assert_allclose(np.linalg.norm(result - pads, axis=1), .004)
        floor = quiet_pad_targets(rig, rotation, position, pads, [False] * 5,
                                 np.ones(5), lambda x, y, f: .7)
        self.assertTrue(np.all(floor[:, 2] >= .703))

    def test_solo_reach_translates_the_wrist_without_changing_finger_angles(self):
        for side in ('R', 'L'):
            rig = HandRig(side)
            rotation, position = rig.sk.rest_root()
            pads = relaxed_pads(rig, rotation, position)
            targets = pads.copy(); targets[2, 0] += .035
            keys = [None, None, 72, None, None]
            moved = solo_wrist_reference(rig, rotation, position, targets, keys, [0, 0, 1, 0, 0])
            self.assertAlmostEqual(moved[0] - position[0], .035)
            np.testing.assert_allclose(relaxed_pads(rig, rotation, moved)[2, 0], targets[2, 0])
            np.testing.assert_array_equal(solo_wrist_reference(rig, rotation, position, targets, keys,
                [0, 0, 1, 0, 0], other_score_note_held=True), position)
            keys[0] = 60
            np.testing.assert_array_equal(solo_wrist_reference(rig, rotation, position, targets, keys,
                [1, 0, 1, 0, 0]), position)

    def test_free_posture_gain_fades_in_both_whole_hand_and_finger_solves(self):
        rig = HandRig('R')
        rotation, position = rig.sk.rest_root()
        vector = np.concatenate([position, np.zeros(3), rig.q_rest])
        finger = 2; a, b = rig.sl[finger:finger + 2]
        vector[6 + a:6 + b] += .01
        base = dict(comfort=1., couple=0., root_pos=0., root_rot=0.,
                    smooth=0., self_collision=0., activation=np.zeros(5))
        for gain in (1., 16.):
            weights = {**base, 'comfort_free': [1., gain, gain, gain, gain]}
            full, _ = rig.residuals(vector, rotation, position, [None] * 5, [0.] * 5, None, None, weights)
            digit = rig.finger_residuals(finger, vector[6 + a:6 + b], vector, rotation,
                                        None, 0., None, weights, None)
            np.testing.assert_allclose(full[a:b], digit[:b - a])
        self.assertEqual(rig._comfort_gain(2, 1., {'comfort_free': 16.}), 1.)
        self.assertAlmostEqual(rig._comfort_gain(2, .5, {'comfort_free': 16.}) ** 2, 8.5)
        with self.assertRaises(ValueError):
            rig._comfort_gain(2, 0., {'comfort_free': float('nan')})

    def test_first_chord_release_does_not_free_a_still_held_note(self):
        a, b, c = Note(0., 1., 60, 90, 0), Note(0., .15, 64, 90, 0), Note(1.2, 1.4, 72, 90, 0)
        fingering = [[(a, 0), (b, 2)], [(c, 2)]]
        with patch.object(piano, 'WRIST_CARRY', True):
            positions, _ = piano.hand_track(fingering, 'R', np.array([.5, .9]), 0.)
        np.testing.assert_allclose(positions, piano._place([(60, 0)], False, 'R'))
        self.assertEqual(a.end, 1.)
        self.assertEqual(b.end, .15)

    def test_controlled_free_step_has_no_target_overshoot(self):
        target = np.r_[np.zeros(48), np.ones(96)]
        result = dynamics.filter_track(target, 24, *piano.SPRING_CONTROLLED_FREE)
        self.assertGreater(result[-1], .999)
        self.assertLessEqual(result.max(), 1. + 1e-10)
        self.assertTrue(np.all(np.diff(result) >= -1e-10))


if __name__ == '__main__':
    unittest.main()

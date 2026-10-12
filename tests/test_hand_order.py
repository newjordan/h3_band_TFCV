import unittest
from unittest.mock import patch

import numpy as np

from blockout import piano, performer
from blockout.midi import Note
from blockout.rig.hand import HandRig
from blockout.rig.hand_order import ordered_notes, ordered_wrists, wrist_order_audit


class HandOrderTests(unittest.TestCase):
    def test_future_lower_voice_does_not_cross_a_held_left_note(self):
        notes = [Note(0., 2., 64, 80, 2), Note(1., 1.5, 60, 90, 1)]
        original = [vars(n).copy() for n in notes]
        right, left, report = ordered_notes(notes, ['L', 'R'], piano.key_x, .19)
        self.assertEqual(right, [])
        self.assertEqual(left, notes)
        self.assertEqual([vars(n) for n in notes], original)
        self.assertEqual(report['allocated_events'], 2)

    def test_wide_held_arpeggio_uses_other_hand_without_changing_releases(self):
        notes = [Note(0., 3., 39, 70, 2), Note(.4, 2., 51, 70, 2),
                 Note(.6, 2., 55, 70, 2), Note(.8, 2., 60, 70, 2)]
        right, left, report = ordered_notes(notes, ['L'] * 4, piano.key_x, .19)
        self.assertTrue(right)
        for time in [0., .5, .7, 1., 2.5]:
            r = [n.pitch for n in right if n.start <= time < n.end]
            l = [n.pitch for n in left if n.start <= time < n.end]
            if r and l:
                self.assertLess(max(l), min(r))
        self.assertEqual(sum(len(v) for v in [right, left]), 4)
        self.assertEqual([n.end for n in notes], [3., 2., 2., 2.])

    def test_impossible_chord_is_retained_and_reported(self):
        notes = [Note(0., 1., p, 70, 0) for p in range(21, 109, 5)]
        right, left, report = ordered_notes(notes, ['R'] * len(notes), piano.key_x, .19)
        self.assertEqual({id(n) for n in right + left}, {id(n) for n in notes})
        self.assertTrue(report['capacity_or_span_conflicts'])

    def test_repeated_pitch_has_one_owner_through_overlap(self):
        notes = [Note(0., 1., 60, 70, 1), Note(.8, 1.3, 60, 70, 2)]
        right, left, _ = ordered_notes(notes, ['R', 'L'], piano.key_x, .19)
        self.assertEqual(right, notes)
        self.assertEqual(left, [])

    def test_more_weight_keeps_supported_wrist_while_idle_hand_gives_way(self):
        r, l = ordered_wrists([-.2, -.1, .02], [.1, -.12, .04], weights=(1., 8.))
        self.assertAlmostEqual(r[0] - l[0], .06)
        self.assertLess(abs(l[0] - .1), abs(r[0] + .2))
        np.testing.assert_array_equal(r[1:], [-.1, .02])
        np.testing.assert_array_equal(l[1:], [-.12, .04])

    def test_coupled_projection_preserves_both_physical_arm_lengths(self):
        shoulders = np.array([[.19, -.45, .22], [-.19, -.45, .22]])
        points = [piano._reachable_wrist(p, s) for p, s in zip([[-.2, -.1, .03], [.2, -.1, .03]], shoulders)]
        r, l = ordered_wrists(*points, shoulders=shoulders,
                             arm_lengths=(performer.UPPER_ARM, performer.FOREARM))
        self.assertGreaterEqual(r[0] - l[0], .06 - 1e-12)
        for side, wrist, shoulder in zip([1, -1], [r, l], shoulders):
            elbow = performer.arm_ik(shoulder, wrist, side)
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), performer.UPPER_ARM, places=8)
            self.assertAlmostEqual(np.linalg.norm(wrist - elbow), performer.FOREARM, places=8)
        with self.assertRaises(ValueError):
            ordered_wrists([-.5, 0., 0.], [.5, 0., 0.], shoulders=[[-.5, 0., 0.], [.5, 0., 0.]],
                            arm_lengths=(.2, .2))

    def test_whole_hand_target_cannot_overpower_wrist_order_bound(self):
        rig = HandRig('R')
        rotation, position = rig.sk.rest_root()
        vector = np.r_[position, np.zeros(3), rig.q_rest]
        _, _, points = rig.points(vector, rotation)
        targets = np.asarray(rig.effectors(points, rig._rots)) - [1., 0., 0.]
        lower = position[0] - .02
        result, _, _ = rig.solve(rotation, position, targets, [20.] * 5, prev=vector,
            iters=12, W={'root_bounds': [[lower, -np.inf, -np.inf], [np.inf]*3]})
        self.assertGreaterEqual(result[0], lower)
        self.assertTrue(np.all(result[6:] >= rig.lo))
        self.assertTrue(np.all(result[6:] <= rig.hi))
        with self.assertRaises(ValueError):
            rig.solve(rotation, position, targets, [1.] * 5, prev=vector, fix_root=True,
                W={'root_bounds': [[position[0]+.1, -np.inf, -np.inf], [np.inf]*3]})

    def test_learned_wrist_waits_until_real_first_note_approach(self):
        n = Note(10., 11., 51, 70, 1)
        H = {'fing': [[(n, 1)]]}
        fallback = np.array([[.15, -.1, .05]] * 4)
        times = np.array([0., 9., 9.7, 9.99])
        with patch.object(piano, 'NO_HAND_CROSS', True):
            path = piano._learned_wrist(H, 'R', times, fallback)
        np.testing.assert_array_equal(path[:2], fallback[:2])
        self.assertLess(path[2, 0], fallback[2, 0])
        self.assertLess(path[3, 0], path[2, 0])

    def test_audit_distinguishes_crossing_from_required_gap(self):
        frames = [{'hands': {side: {'wrist': [x, 0., 0.], 'root': {'pos': [x, 0., 0.]}}
                            for side, x in [('R', gap), ('L', 0.)]}} for gap in [.1, .04, -.2]]
        report = wrist_order_audit(frames, 24)
        self.assertEqual(report['violating_frames'], [1, 2])
        self.assertEqual(report['crossing_frames'], [2])
        self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main()

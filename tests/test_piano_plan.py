import copy
import unittest
from unittest.mock import patch

import numpy as np

from blockout import piano
from blockout.midi import Note
from blockout.performance_windows import export_window, rejoin_windows
from blockout.rig.hand import HandRig
from blockout.rig.hand_order import wrist_order_audit


class PianoPlanTests(unittest.TestCase):
    def notes(self):
        notes = [Note(0., 1.6, 48, 60, 2), Note(.6, 1.2, 60, 80, 1),
                 Note(1.3, 1.8, 64, 70, 1), Note(3., 3.5, 80, 90, 1)]
        notes[0].touch, notes[0].lift = .8, .4
        return notes

    def test_compiled_fingering_and_body_centre_survive_window_choice(self):
        notes = self.notes()
        original = copy.deepcopy([vars(n) for n in notes])
        score = piano.compile_score(notes)
        with patch.object(piano, 'fingering', side_effect=AssertionError('recompiled')):
            a = piano.prepare_plan(compiled_score=score, start=.6, dur=.3)
            b = piano.prepare_plan(compiled_score=score, start=3., dur=.3)
        self.assertEqual(a.centre_x, b.centre_x)
        for side in ('R', 'L'):
            self.assertEqual(a.hands[side]['fing'], b.hands[side]['fing'])
            self.assertEqual(a.hands[side]['ev'], b.hands[side]['ev'])
            self.assertEqual(a.hands[side]['score_notes'], b.hands[side]['score_notes'])
        a.hands['R']['ev'][0].append((999.,))
        self.assertNotEqual(a.hands['R']['ev'][0], score.hands['R']['ev'][0])
        self.assertEqual([vars(n) for n in notes], original)

    def test_timing_copy_retains_touch_and_original_releases(self):
        notes = self.notes()
        score = piano.compile_score(notes, speed=2.)
        self.assertEqual(score.notes[0].end, .8)
        self.assertEqual(score.notes[0].touch, .8)
        self.assertEqual(score.notes[0].lift, .4)
        self.assertEqual(notes[0].end, 1.6)

    def test_complete_score_handles_silence_and_reports_out_of_keyboard(self):
        score = piano.compile_score([Note(0., 1., 20, 80, 2)])
        plan = piano.prepare_plan(compiled_score=score, start=10., dur=.2)
        self.assertEqual(plan.outside_keyboard, [[0., 1., 20]])
        self.assertEqual(plan.hands['L']['score_notes'], [(0., 1., 20)])

    def test_changed_controller_is_rejected_before_physics(self):
        plan = piano.prepare_plan(self.notes(), start=.6, dur=.2)
        with patch.object(piano, 'FREE_CONTROL', not piano.FREE_CONTROL):
            with self.assertRaises(ValueError):
                piano.solve_plan(plan)
        self.assertFalse(plan.consumed)

    def test_physical_warmup_reaches_ik_and_is_trimmed_only_after_arm_solve(self):
        for warm in (False, True):
            with self.subTest(warm=warm):
                plan = piano.prepare_plan(self.notes(), fps=24, start=.6, dur=.25,
                                          complete_score=True, physical_warmup=warm)
                observed = {}
                def physics(hands, times, j0, count, start, *args):
                    observed['j0'] = j0
                    posed = {}
                    for side, x in [('R', .2), ('L', -.2)]:
                        hand = HandRig(side)
                        rotation, position = hand.sk.rest_root()
                        position = np.array([x, -.1, .03])
                        _, _, points = hand.points(np.r_[position, np.zeros(3), hand.q_rest], rotation)
                        posed[side] = dict(wrist=position.tolist(), fingers=np.asarray(points).tolist(),
                            elbow=(position+[0., -.25, .1]).tolist(), root={'pos': position.tolist()})
                    return [{'t': float(times[j]-start), 'keys': {}, 'hands': copy.deepcopy(posed)}
                            for j in range(j0, count)]
                def arms(frames, fps):
                    observed['arm_frames'] = len(frames)
                    return {}
                with patch.object(piano, 'HAND_MODEL', 'mpfb'):
                    plan.settings = piano._plan_settings()
                    with patch.object(piano, '_pass3_rig', side_effect=physics), \
                         patch.object(piano.performer, 'relax_arm_track', side_effect=arms), \
                         patch.object(piano, '_rig') as rig:
                        rig.return_value.sk.scale = 1.
                        result = piano.solve_plan(plan)
                self.assertEqual(observed['j0'], 0 if warm else 48)
                self.assertEqual(observed['arm_frames'], 54 if warm else 6)
                self.assertEqual(len(result['frames']), 6)
                self.assertAlmostEqual(result['frames'][0]['t'], 0.)
                self.assertEqual(len(result['warmup']['frames']), 48 if warm else 0)
                self.assertEqual(result['key_action_ledger'],plan.action_ledger)
                with self.assertRaises(ValueError):
                    piano.solve_plan(plan)

    def test_empty_warmup_is_unobserved_and_nan_cannot_pass_order_gate(self):
        self.assertIsNone(wrist_order_audit([], 24)['passed'])
        with self.assertRaises(ValueError):
            wrist_order_audit([{'hands': {'R': {'wrist': [np.nan, 0., 0.]},
                                        'L': {'wrist': [0., 0., 0.]}}}], 24)

    def test_render_partitions_rejoin_exactly_and_reject_modified_motion(self):
        animation = dict(start=282.9, fps=24, frames=[dict(t=i/24., keys={'60': .5+i*.01},
            hands={'R': {'root': {'pos': [i*.001, 0., 0.]}}}, body={'marker': i}) for i in range(96)])
        original = copy.deepcopy(animation)
        parts = [export_window(animation, 0, 48), export_window(animation, 48, 96)]
        self.assertEqual(rejoin_windows(parts)['frames'], original['frames'])
        self.assertEqual(animation, original)
        with self.assertRaises(ValueError):
            rejoin_windows(parts[::-1])
        with self.assertRaises(ValueError):
            rejoin_windows(parts[:1])
        parts[1]['frames'][0]['keys']['60'] = .999
        with self.assertRaises(ValueError):
            rejoin_windows(parts)


if __name__ == '__main__':
    unittest.main()

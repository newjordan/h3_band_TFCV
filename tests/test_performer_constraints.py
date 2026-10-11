import copy
import json
from pathlib import Path
import unittest

import numpy as np

from blockout import performer as p
from blockout.rig.skeleton import Skeleton


class PerformerConstraintsTest(unittest.TestCase):
    def test_continuous_bend_preserves_both_lengths(self):
        shoulder = np.array([-.19, -.45, .3])
        wrist = np.array([-.05, -.05, .05])
        old = p.arm_ik(shoulder, wrist, -1, hand_fwd=[0, 1, 0])
        new = p.arm_ik(shoulder, wrist, -1, hand_fwd=[-1, 0, 0], previous=old, dt=1 / 24)
        self.assertAlmostEqual(np.linalg.norm(new - shoulder), p.UPPER_ARM, places=12)
        self.assertAlmostEqual(np.linalg.norm(new - wrist), p.FOREARM, places=12)
        axis = wrist - shoulder
        length = np.linalg.norm(axis); axis /= length
        along = (p.UPPER_ARM ** 2 - p.FOREARM ** 2 + length ** 2) / (2 * length)
        centre = shoulder + along * axis
        a, b = old - centre, new - centre
        angle = np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
        self.assertLessEqual(angle, 4.0 / 24 + 1e-9)

    def test_recorded_chopin_elbow_branch_does_not_snap(self):
        # Actual part_17 frames at performed 290.367 and 290.4087 seconds,
        # corresponding to video 250.46/250.50. The original elbow jumped104mm.
        shoulder = np.array([-.2663537275, -.4506754321, .2779037756])
        wrist = np.array([-.13762, -.05737, .07367])
        previous = np.array([-.2781174862, -.3320688848, .0083403223])
        forward = [.6028697814, .7928039714, -.0895082912]
        torso = ([-.0404077346, -.4874855382, -.1011653796],
                 [-.0725865437, -.4509850245, .2454355967],
                 {'L': shoulder, 'R': [.1119866946, -.4408658988, .3119962682]},
                 np.array([[.9956326898, -.0162075215, -.0919394546],
                           [.0258145615, .9942121464, .1042871818],
                           [.0897170858, -.1062051041, .9902885036]]))
        unseeded = p.arm_ik(shoulder, wrist, -1, forward, torso=torso)
        seeded = p.arm_ik(shoulder, wrist, -1, forward, torso=torso, previous=previous, dt=1 / 24)
        self.assertGreater(np.linalg.norm(unseeded - previous), .09)
        self.assertLess(np.linalg.norm(seeded - previous), .04)
        self.assertLessEqual(p.arm_clearance(shoulder, seeded, wrist, torso, 'L'), .0005)
        self.assertAlmostEqual(np.linalg.norm(seeded - wrist), p.FOREARM, places=12)

    def test_coupled_crossing_arms_clear_each_other_without_stretch(self):
        body = p.Body([0, -.5, -.22])
        wrists = {'L': [.12, -.08, .06], 'R': [-.12, -.08, .025]}
        forward = {'L': [0, 1, -.1], 'R': [0, 1, -.1]}
        solved = body.pose(0, 0, 0, 0, wrists, forward)
        shoulder, elbow = solved['shoulders'], solved['elbows']
        pen = p.arm_pair_clearance(shoulder['L'], elbow['L'], wrists['L'],
                                  (shoulder['R'], elbow['R'], wrists['R']))
        self.assertLess(pen, .001)
        hitbox = p.torso_box(solved)
        for side in wrists:
            self.assertLessEqual(p.arm_clearance(shoulder[side], elbow[side], wrists[side], hitbox, side), .0005)
            self.assertAlmostEqual(np.linalg.norm(np.array(elbow[side]) - shoulder[side]), p.UPPER_ARM, places=12)
            self.assertAlmostEqual(np.linalg.norm(np.array(elbow[side]) - wrists[side]), p.FOREARM, places=12)

    def test_unreachable_endpoints_stay_finite_with_rigid_upper_arm(self):
        shoulder = np.zeros(3)
        for wrist in (np.array([0., 2., 0.]), np.array([0., .001, 0.]), shoulder):
            elbow = p.arm_ik(shoulder, wrist, 1)
            self.assertTrue(np.all(np.isfinite(elbow)))
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), p.UPPER_ARM, places=12)
            # No exact two-bone solution exists; the remaining reach error must
            # stay observable to the caller, instead of hiding a stretched bone.
            self.assertGreater(abs(np.linalg.norm(elbow - wrist) - p.FOREARM), .001)

    def test_invalid_control_interval_is_rejected(self):
        shoulder, wrist = np.array([-.19, -.45, .3]), np.array([-.05, -.05, .05])
        previous = p.arm_ik(shoulder, wrist, -1)
        for dt in (0, -0.1, float('nan')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                p.arm_ik(shoulder, wrist, -1, previous=previous, dt=dt)

    def test_offline_track_clears_crossing_without_moving_wrists(self):
        body = p.Body([0, -.5, -.22])
        skeletons = {s: Skeleton(s) for s in 'LR'}
        frames = []
        for j in range(12):
            pose = body.pose(0, 0, 0, 0, {})
            wrists = {'L': [.12 + .005 * np.sin(j / 5), -.08, .06],
                      'R': [-.12, -.08 + .005 * np.cos(j / 5), .025]}
            hands = {}
            for side in 'LR':
                elbow = p.arm_ik(np.array(pose['shoulders'][side]), np.array(wrists[side]),
                                 1 if side == 'R' else -1, [0, 1, -.1], torso=p.torso_box(pose))
                hands[side] = {'wrist': wrists[side], 'elbow': elbow.tolist(), 'fingers': [],
                               'root': {'rot': skeletons[side].R0['wrist'].tolist()}}
                pose['elbows'][side] = elbow.tolist()
            frames.append({'t': j / 24, 'body': pose, 'hands': hands})
        before = copy.deepcopy(frames)
        audit = p.relax_arm_track(frames, 24)
        self.assertTrue(all(v['applied'] for v in audit['passes']))
        self.assertLess(audit['max_overlap_mm'], 1.)
        self.assertLessEqual(audit['max_torso_penetration_mm'], .5)
        for j, frame in enumerate(frames):
            for side in 'LR':
                hand = frame['hands'][side]
                np.testing.assert_array_equal(hand['wrist'], before[j]['hands'][side]['wrist'])
                np.testing.assert_array_equal(hand['root']['rot'], before[j]['hands'][side]['root']['rot'])
                shoulder, wrist, elbow = (np.asarray(v) for v in (frame['body']['shoulders'][side],
                                                                  hand['wrist'], hand['elbow']))
                self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), p.UPPER_ARM, places=12)
                self.assertAlmostEqual(np.linalg.norm(elbow - wrist), p.FOREARM, places=12)
                if j:
                    axis = wrist - shoulder; length = np.linalg.norm(axis); axis /= length
                    along = (p.UPPER_ARM ** 2 - p.FOREARM ** 2 + length ** 2) / (2 * length)
                    centre = shoulder + along * axis
                    previous = np.array(frames[j - 1]['hands'][side]['elbow']) - centre
                    previous -= (previous @ axis) * axis
                    current = elbow - centre
                    angle = np.arccos(np.clip(previous @ current / (np.linalg.norm(previous) * np.linalg.norm(current)), -1, 1))
                    self.assertLessEqual(angle * 24, 4. + 1e-8)

    def test_offline_infeasible_torso_reports_and_retains_input(self):
        sk = Skeleton('R')
        shoulder, wrist = np.array([0., 0., .3]), np.array([0., 0., .89])
        elbow = p.arm_ik(shoulder, wrist, 1)
        frame = {'t': 0., 'body': {'hips': [0., 0., -.5], 'chest': [0., 0., 1.5],
                                  'torso_rot': np.eye(3).tolist(), 'shoulders': {'R': shoulder.tolist()},
                                  'elbows': {'R': elbow.tolist()}},
                 'hands': {'R': {'wrist': wrist.tolist(), 'elbow': elbow.tolist(), 'fingers': [],
                                  'root': {'rot': sk.R0['wrist'].tolist()}}}}
        frames = [frame]; before = copy.deepcopy(frames)
        result = p.relax_arm_track(frames, 24)
        self.assertTrue(all(not v['applied'] for v in result['passes']))
        self.assertTrue(all('torso-clear' in v['reason'] for v in result['passes']))
        self.assertEqual(frames, before)

    def test_opposite_thumb_contact_at_fixed_wrist_cannot_be_hidden(self):
        # Captured v1 Debussy frame at performed51.6917: the wrist itself lies
        # inside the opposite thumb/forearm capsule envelope. Elbows alone
        # cannot fix this; the caller must receive a remaining-collision report.
        frame = json.loads((Path(__file__).parent / 'fixtures/debussy_arm_contact.json').read_text())
        before = copy.deepcopy(frame)
        result = p.relax_arm_track([frame], 24)
        self.assertGreater(result['max_overlap_mm'], 25.)
        for side in 'LR':
            self.assertEqual(frame['hands'][side]['wrist'], before['hands'][side]['wrist'])
            self.assertEqual(frame['hands'][side]['fingers'], before['hands'][side]['fingers'])

    def test_unreachable_warmup_is_reported_without_disabling_later_track(self):
        source = json.loads((Path(__file__).parent / 'fixtures/debussy_arm_contact.json').read_text())
        bad = copy.deepcopy(source)
        bad['hands']['L']['wrist'][0] += 2.0
        later = copy.deepcopy(source); later['t'] = 1.0
        frames = [bad, later]; retained = copy.deepcopy(bad)
        result = p.relax_arm_track(frames, 24)
        self.assertEqual(frames[0], retained)
        self.assertEqual(result['unreachable_frames'][0]['frame'], 0)
        self.assertTrue(result['reachable_run_metrics_only'])
        self.assertTrue(any(v['applied'] and v['run_start'] == 1 for v in result['passes']))
        for side in 'LR':
            shoulder = np.array(later['body']['shoulders'][side])
            elbow = np.array(later['hands'][side]['elbow'])
            wrist = np.array(later['hands'][side]['wrist'])
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), p.UPPER_ARM, places=12)
            self.assertAlmostEqual(np.linalg.norm(elbow - wrist), p.FOREARM, places=12)


if __name__ == '__main__':
    unittest.main()

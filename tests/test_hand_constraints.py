"""Behavior checks on solved, physical hand poses (no Blender dependency)."""
import math
import unittest

import numpy as np

from blockout.rig.hand import HandRig


class HandConstraintsTest(unittest.TestCase):
    def hand(self, side="R"):
        rig = HandRig(side)
        rotation, position = rig.sk.rest_root()
        pose = np.r_[position, np.zeros(3), rig.q_rest]
        _, _, points = rig.points(pose, rotation)
        targets = [p.copy() for p in rig.effectors(points, rig._rots)]
        return rig, rotation, position, pose, targets

    def solve(self, rig, rotation, position, pose, targets, weights, whole, **kw):
        if whole:
            return rig.solve(rotation, position, targets, weights, prev=pose,
                             fix_root=True, iters=30, **kw)[0]
        return rig.solve_fingers(rotation, pose, targets, weights, iters=30, **kw)

    def test_crossing_targets_do_not_pass_fingers_through_each_other(self):
        # The targets exchange two adjacent fingers; accuracy must yield to the
        # finger volumes even under a twentyfold strike-position weight.
        for side in "RL":
            for whole in (False, True):
                with self.subTest(side=side, whole=whole):
                    rig, rot, pos, pose, targets = self.hand(side)
                    targets[1], targets[2] = targets[2], targets[1]
                    free = self.solve(rig, rot, pos, pose, targets, [20] * 5, whole,
                                      W={"self_collision": 0})
                    safe = self.solve(rig, rot, pos, pose, targets, [20] * 5, whole)
                    self.assertLess(float(rig.finger_clearances(free, rot).min()), -0.008)
                    self.assertGreater(float(rig.finger_clearances(safe, rot).min()), -0.001)
                    self.assertTrue(np.all(safe[6:] >= rig.lo))
                    self.assertTrue(np.all(safe[6:] <= rig.hi))

    def test_impossible_reach_preserves_joint_bounds_and_bone_lengths(self):
        rig, rot, pos, pose, targets = self.hand()
        limits = rig.lo.copy(), rig.hi.copy()
        lengths = [np.array(c.L) for c in rig.chains]
        targets = [t + np.array([1.0, -1.0, 1.0]) for t in targets]
        for whole in (False, True):
            with self.subTest(whole=whole):
                result = self.solve(rig, rot, pos, pose, targets, [20] * 5, whole)
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertTrue(np.all(result[6:] >= limits[0]))
                self.assertTrue(np.all(result[6:] <= limits[1]))
                np.testing.assert_array_equal(result[:6], pose[:6])
                _, _, points = rig.points(result, rot)
                for chain, points_f, length in zip(rig.chains, points, lengths):
                    np.testing.assert_array_equal(chain.L, length)
                    # The last bone ends at the rigid distal tip, independent of pad contact.
                    # Exported bone frames have single-precision rounding.
                    self.assertAlmostEqual(np.linalg.norm(points_f[-1] - points_f[-2]), length[-1], delta=1e-6)
        np.testing.assert_array_equal(rig.lo, limits[0])
        np.testing.assert_array_equal(rig.hi, limits[1])

    def test_velocity_uses_actual_previous_pose_not_smoothed_regularizer(self):
        rig, rot, pos, previous, targets = self.hand()
        warm = previous.copy(); warm[6:] = rig.hi
        prior = previous.copy(); prior[6:] = rig.lo
        dt, speed = 1 / 60, 2.0
        targets = [t + np.array([0.03, 0.02, -0.03]) for t in targets]
        for whole in (False, True):
            with self.subTest(whole=whole):
                result = self.solve(rig, rot, pos, warm, targets, [20] * 5, whole,
                                    last=prior, W={"dt": dt, "motion_last": previous,
                                                   "max_joint_speed": speed, "smooth": 0.01})
                self.assertLessEqual(float(np.max(np.abs(result[6:] - previous[6:]))), speed * dt + 1e-12)

    def test_attack_activation_releases_comfort_continuously(self):
        rig, rot, pos, pose, targets = self.hand()
        targets[1] += np.array([0.0, 0.0, -0.02])
        misses = []
        for active in (0.0, 0.5, 1.0):
            result = rig.solve_fingers(rot, pose, targets, [1] * 5, iters=30,
                                       W={"self_collision": 0, "comfort": 0.003,
                                          "comfort_press": 0.02, "activation": [active] * 5})
            _, _, points = rig.points(result, rot)
            misses.append(np.linalg.norm(rig.effectors(points, rig._rots)[1] - targets[1]))
        self.assertGreater(misses[0], misses[1])
        self.assertGreater(misses[1], misses[2])

    def test_chain_reports_skin_pad_error(self):
        rig, rot, pos, pose, targets = self.hand()
        chain = rig.chains[2]
        base_rot, base_head = rig.bases(rot, pos)[2]
        q, points, miss = chain.solve(base_rot, base_head, targets[2],
                                     q_prev=chain.q_rest, w_comfort=0, w_couple=0)
        points, frames = chain.fk(base_rot, base_head, q)
        actual = np.linalg.norm(chain.effector(points, frames) - targets[2])
        self.assertAlmostEqual(miss, actual, places=12)
        self.assertLess(miss, 1e-8)
        self.assertGreater(np.linalg.norm(points[-1] - targets[2]), 0.002)

    def test_invalid_motion_timestep_fails_before_solving(self):
        rig, rot, _, pose, targets = self.hand()
        for dt in (0.0, -0.01, math.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                rig.solve_fingers(rot, pose, targets, [1] * 5,
                                  W={"motion_last": pose, "dt": dt})


if __name__ == "__main__":
    unittest.main()

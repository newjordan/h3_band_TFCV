import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from blockout import piano
from blockout.rig.orientation import absolute_pose, relative_pose, forearm_angles, smooth_rotations, split_reference_rotations


class OrientationTests(unittest.TestCase):
    def test_export_precision_metric_is_retained_through_history_and_smoothing(self):
        from blockout.rig.hand import HandRig
        rig=HandRig('R')
        rest=rig.sk.R0['wrist']
        _,metric=split_reference_rotations(rest)
        turns=Rotation.from_euler('xyz',[[.2,.1,.3],[-.2,.3,.1],[.1,.4,-.2]]).as_matrix()
        actual=turns[0] @ rest
        vector=np.r_[[.1,.2,.3],np.zeros(23)]
        physical=absolute_pose(vector,actual,metric)
        for turn in turns:
            reference=turn @ rest
            encoded=relative_pose(physical,reference,metric)
            rebuilt=Rotation.from_rotvec(encoded[3:6]).as_matrix() @ reference
            np.testing.assert_allclose(rebuilt,actual,atol=1e-12)
        output,receipt=smooth_rotations(turns @ rest,np.ones(3),12.)
        np.testing.assert_allclose(split_reference_rotations(output)[1],np.tile(metric,(3,1,1)),atol=1e-12)
        self.assertTrue(receipt['fixed_reference_stretch_retained'])
        with self.assertRaisesRegex(ValueError,'fixed wrist metric'):
            relative_pose(physical,np.eye(3),metric)

    def test_time_varying_affine_stretch_is_not_disguised_as_rotation_smoothing(self):
        matrices=np.tile(np.eye(3),(3,1,1))
        matrices[1,0,0]+=1e-6
        with self.assertRaisesRegex(ValueError,'fixed exported reference metric'):
            smooth_rotations(matrices,np.ones(3),12.)

    def test_history_preserves_physical_pose_across_moving_noncommuting_references(self):
        old = Rotation.from_euler('xyz', [1.2, -.7, .4]).as_matrix()
        vector = np.r_[[.2, .1, .03], [.3, -.1, .15], np.arange(20)/100.]
        actual = (Rotation.from_rotvec(vector[3:6]) * Rotation.from_matrix(old)).as_matrix()
        physical = absolute_pose(vector, actual)
        original = physical.copy()
        for reference in (np.eye(3), Rotation.from_euler('zyx', [-2., .4, .9]).as_matrix(), old):
            transported = relative_pose(physical, reference)
            rebuilt = (Rotation.from_rotvec(transported[3:6]) * Rotation.from_matrix(reference)).as_matrix()
            np.testing.assert_allclose(rebuilt, actual, atol=1e-12)
            np.testing.assert_array_equal(transported[:3], vector[:3])
            np.testing.assert_array_equal(transported[6:], vector[6:])
        np.testing.assert_array_equal(physical, original)

    def test_backward_forearm_cannot_alias_to_straight_wrist(self):
        axes = np.eye(3)
        self.assertEqual(forearm_angles(axes[1], *axes), (0., 0.))
        flex, dev = forearm_angles(-axes[1], *axes)
        self.assertAlmostEqual(abs(flex), np.pi)
        self.assertAlmostEqual(dev, 0.)
        flex, dev = forearm_angles([0., 1., 1.], *axes)
        self.assertAlmostEqual(flex, np.pi/4)
        self.assertAlmostEqual(dev, 0.)

    def test_smoothing_does_not_spin_at_principal_log_branch(self):
        angles = np.deg2rad([177., 178., 179., -180., -179., -178., -177.])
        rotations = Rotation.from_rotvec(np.c_[np.zeros((7, 2)), angles]).as_matrix()
        output, receipt = smooth_rotations(rotations, np.ones(7), 12.)
        np.testing.assert_allclose(output, rotations, atol=1e-10)
        self.assertLess(receipt['final_cost'], 1e-20)

    def test_fit_reduces_roughness_and_is_equivariant_to_world_frame(self):
        angles = np.c_[np.linspace(0., .2, 9), [.0, .01, -.01, .02, .25, -.02, .01, 0., 0.], np.zeros(9)]
        original = Rotation.from_rotvec(angles).as_matrix()
        saved = original.copy()
        output, receipt = smooth_rotations(original, np.ones(9), 12.)
        self.assertLess(receipt['final_cost'], receipt['initial_cost']*.5)
        world = Rotation.from_euler('xyz', [.7, -.6, .2]).as_matrix()
        transformed, other = smooth_rotations(world @ original, np.ones(9), 12.)
        np.testing.assert_allclose(transformed, world @ output, atol=5e-6)
        self.assertAlmostEqual(other['final_cost'], receipt['final_cost'], places=8)
        np.testing.assert_array_equal(original, saved)

    def test_nonrotations_nonfinite_inputs_and_missing_anchor_are_rejected(self):
        for bad in (np.diag([1., 1., -1.]), np.eye(3)*1.01, np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                absolute_pose(np.zeros(26), bad)
        for weights, penalty in (([0.]*3, 12.), ([1., np.nan, 1.], 12.), ([1.]*3, -1.)):
            with self.assertRaises(ValueError):
                smooth_rotations(np.tile(np.eye(3), (3, 1, 1)), weights, penalty)
        with self.assertRaises(ValueError):
            forearm_angles([0., 0., 0.], *np.eye(3))

    def test_chronological_history_is_rebased_but_same_frame_warm_start_is_not(self):
        class Rig:
            q_rest = np.zeros(5)
            _rots = []
            chains = []
            def __init__(self):
                self.calls = []
            def solve(self, rr, rp, targets, weights, **kwargs):
                self.calls.append(copy.deepcopy(kwargs))
                x = kwargs['prev']
                rotation = (Rotation.from_rotvec(x[3:6]) * Rotation.from_matrix(rr)).as_matrix()
                return x, (rotation, x[:3], [], targets), [0.]*5
            def points(self, x, rr):
                return rr, x[:3], []
        rig = Rig()
        actual = Rotation.from_euler('xyz', [.2, -.3, .1]).as_matrix()
        reference = Rotation.from_euler('zyx', [.7, .3, -.2]).as_matrix()
        physical = absolute_pose(np.r_[[.1, .2, .3], np.zeros(8)], actual)
        warm = np.r_[[.1, .2, .3], [.01, .02, .03], np.zeros(5)]
        hands = {'R': dict(act=[[None]*5])}
        refs = {'R': (reference, physical[:3])}
        with patch.object(piano, '_rig', return_value=rig), \
             patch.object(piano, '_plan_root', return_value=refs['R']), \
             patch.object(piano, '_key_env', return_value=None), \
             patch.object(piano, '_hand_blobs', return_value=None), \
             patch.object(piano, 'WRIST_SO3_HISTORY', True):
            solved = piano._solve_frame(hands, 0, {'R': np.zeros((5, 3))}, {}, {'R': physical}, refs=refs)
            np.testing.assert_allclose(solved['R'][1], actual, atol=1e-12)
            piano._solve_frame(hands, 0, {'R': np.zeros((5, 3))}, {}, {'R': physical}, refs=refs, warm={'R': warm})
        np.testing.assert_array_equal(rig.calls[1]['prev'], warm)
        np.testing.assert_allclose(rig.calls[1]['last'], relative_pose(physical, reference), atol=1e-12)


if __name__ == '__main__':
    unittest.main()

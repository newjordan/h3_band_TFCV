import unittest

import numpy as np

from blockout.skin import Skin, clean_hand_weights, rotation_between


class SkinTests(unittest.TestCase):
    def test_other_hand_weight_is_migrated_to_matching_bone(self):
        weights = np.array([[.0006, .9594, .04], [.8, .1, .1]])
        names = ['finger3-1.L', 'finger2-1.R', 'finger3-1.R']
        with self.assertRaises(ValueError):
            clean_hand_weights(weights, names)  # left-owned row lacks finger2-1.L
        weights = weights[:1]
        clean, repairs = clean_hand_weights(weights, names)
        np.testing.assert_allclose(clean, [[0., .9594, .0406]])
        self.assertEqual(repairs[0]['to'], 'finger3-1.R')
        np.testing.assert_array_equal(weights, [[.0006, .9594, .04]])

    def test_exported_skin_keeps_mixed_joint_vertices(self):
        skin = Skin('R', scale=1.)
        self.assertGreater(np.count_nonzero((skin.weights > 1e-6).sum(1) > 1), 200)
        np.testing.assert_allclose(skin.weights.sum(1), 1., atol=1e-12)
        for finger in range(1, 6):
            self.assertGreater(len(skin.affected(finger)), 100)
        self.assertGreater(len(skin.triangles), len(skin.vertices))

    def test_pose_translation_moves_every_skin_vertex_equally(self):
        for side in ('L', 'R'):
            skin = Skin(side, scale=.7284)
            root, position = skin.sk.rest_root()
            elbow = position + [0., -.3, .05]
            angles = {'finger2-1': {'flex': .4, 'abd': -.1}, 'finger2-2': {'flex': .6}}
            before = skin.pose(root, position, angles, elbow)
            offset = np.array([.11, -.07, .03])
            after = skin.pose(root, position + offset, angles, elbow + offset)
            np.testing.assert_allclose(after - before, np.tile(offset, (len(before), 1)), atol=1e-12)

    def test_rotation_handles_opposed_and_aligned_arm_axes(self):
        for first, second in (([0., 1., 0.], [0., -1., 0.]),
                              ([1., 0., 0.], [1., 0., 0.]),
                              ([0., 1., 0.], [0., 0., 1.])):
            rotation = rotation_between(first, second)
            np.testing.assert_allclose(rotation @ first, second, atol=1e-12)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_invalid_pose_or_scale_is_rejected(self):
        for scale in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                Skin('L', scale=scale)
        skin = Skin('L', scale=1.)
        root, position = skin.sk.rest_root()
        with self.assertRaises(ValueError):
            skin.pose(root, position, {}, position)


if __name__ == '__main__':
    unittest.main()

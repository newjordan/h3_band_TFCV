import copy
import unittest

import numpy as np

from blockout.performer import UPPER_ARM, FOREARM
from blockout.rig.hand import HandRig, rotvec
from blockout.rig.motion import interpolate_frame, rotation_angle
from blockout.rig.skin_contact import continuous_elbow


def fixture(rig, root, rotation, q, time):
    root = np.asarray(root, float)
    shoulder = np.array([.15, -.4, .3])
    elbow = continuous_elbow(shoulder, root, [.2, -.4, .05])
    vector = np.r_[root, np.zeros(3), q]
    _, _, points = rig.points(vector, rotation)
    hand = dict(root=dict(pos=root.tolist(), rot=rotation.tolist()), wrist=root.tolist(),
                fingers=np.array(points).tolist(), pads=np.array(rig.effectors(points, rig._rots)).tolist(),
                angles=rig.angles(vector), elbow=elbow.tolist(), activation=[.8] * 5,
                assigned_keys=[60, None, None, None, None])
    return dict(t=time, hands={'R': hand}, keys={'60': .5},
                body=dict(shoulders={'R': shoulder.tolist()}, elbows={'R': elbow.tolist()}))


class MotionTests(unittest.TestCase):
    def test_interpolation_keeps_anatomy_and_both_arm_lengths(self):
        rig = HandRig('R')
        rotation, _ = rig.sk.rest_root()
        first = fixture(rig, [.15, -.1, .03], rotation, rig.q_rest, 0.)
        second = fixture(rig, [.16, -.095, .035], rotvec(np.array([0., 0., .05])) @ rotation,
                         np.clip(rig.q_rest + .02, rig.lo, rig.hi), 1. / 24.)
        original = copy.deepcopy(first)
        for fraction in np.linspace(0., 1., 11):
            frame = interpolate_frame(first, second, fraction, {'R': rig})
            hand = frame['hands']['R']
            shoulder, elbow, wrist = (np.asarray(v) for v in
                (frame['body']['shoulders']['R'], hand['elbow'], hand['wrist']))
            self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), UPPER_ARM, places=12)
            self.assertAlmostEqual(np.linalg.norm(wrist - elbow), FOREARM, places=12)
            self.assertAlmostEqual(np.linalg.det(hand['root']['rot']), 1., places=12)
        self.assertEqual(first, original)

    def test_rotation_uses_short_arc_and_rejects_invalid_time(self):
        first, second = rotvec(np.array([0., 0., np.deg2rad(179.)])), rotvec(np.array([0., 0., np.deg2rad(-179.)]))
        self.assertAlmostEqual(rotation_angle(first, second), np.deg2rad(2.), places=12)
        with self.assertRaises(ValueError):
            interpolate_frame({}, {}, float('nan'), {})


if __name__ == '__main__':
    unittest.main()

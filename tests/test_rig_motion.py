import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from blockout.performer import UPPER_ARM, FOREARM
from blockout.rig.hand import HandRig, rotvec
from blockout.rig.motion import interpolate_frame, rotation_angle
from blockout.rig.skin_contact import continuous_elbow
from blockout.rig.path_checkpoint import save_checkpoint, load_checkpoint


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
    def test_failed_path_numeric_state_roundtrips_and_rejects_mismatched_geometry(self):
        rig = HandRig('R')
        rotation, _ = rig.sk.rest_root()
        frames = [fixture(rig, [.15, -.1, .03], rotation, rig.q_rest, 0.)]
        identity = dict(source='frozen-source', geometry='notched')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'working.checkpoint.npz'
            save_checkpoint(path, frames, {'R': rig}, identity, 2)
            resumed, metadata = load_checkpoint(path, frames, {'R': rig}, identity)
            self.assertFalse(metadata['candidate_exported'])
            self.assertEqual(metadata['sweeps_completed'], 2)
            np.testing.assert_allclose(resumed[0]['hands']['R']['fingers'], frames[0]['hands']['R']['fingers'], atol=1e-12)
            self.assertEqual(resumed[0]['hands']['R']['angles'], frames[0]['hands']['R']['angles'])
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                load_checkpoint(path, frames, {'R': rig}, dict(identity, geometry='rectangular'))
            with np.load(path, allow_pickle=False) as stored:
                corrupt = {name: stored[name].copy() for name in stored.files}
            corrupt['R_rotations'][0, 0, 0] = float('nan')
            np.savez_compressed(path, **corrupt)
            with self.assertRaisesRegex(ValueError, 'numeric pose'):
                load_checkpoint(path, frames, {'R': rig}, identity)

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

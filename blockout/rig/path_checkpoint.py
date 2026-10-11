"""Numeric solver state for resuming failed experiments; not an animation."""
import copy
import json

import numpy as np

from .skin_contact import pose_vector


def save_checkpoint(path, frames, rigs, identity, sweeps_completed):
    metadata = dict(schema='unvalidated-sampled-path-state/v1', identity=identity,
                    sweeps_completed=sweeps_completed, candidate_exported=False)
    values = dict(metadata=np.array(json.dumps(metadata, sort_keys=True)),
                  times=np.array([f['t'] for f in frames]))
    for side, rig in rigs.items():
        values[side + '_poses'] = np.array([pose_vector(rig, f['hands'][side]) for f in frames])
        values[side + '_rotations'] = np.array([f['hands'][side]['root']['rot'] for f in frames])
        values[side + '_elbows'] = np.array([f['hands'][side]['elbow'] for f in frames])
    np.savez_compressed(path, **values)


def load_checkpoint(path, frames, rigs, identity):
    """Reject mismatched inputs before applying any stored numeric pose."""
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
        if metadata.get('schema') != 'unvalidated-sampled-path-state/v1' or metadata.get('identity') != identity:
            raise ValueError('Checkpoint source, solver, geometry, score or sampling identity differs')
        if not np.array_equal(stored['times'], [f['t'] for f in frames]):
            raise ValueError('Checkpoint timestamps differ')
        arrays = {}
        for side, rig in rigs.items():
            shape = pose_vector(rig, frames[0]['hands'][side]).shape
            poses, rotations, elbows = (stored[side + suffix].copy() for suffix in ('_poses', '_rotations', '_elbows'))
            if (poses.shape != (len(frames), *shape) or rotations.shape != (len(frames), 3, 3)
                    or elbows.shape != (len(frames), 3) or not all(np.isfinite(a).all() for a in (poses, rotations, elbows))
                    or np.any(poses[:, 3:6] != 0.)
                    # Exported bind/root matrices are rounded to six decimals;
                    # preserve them exactly within that serialization error.
                    or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0.)
                    or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5, rtol=0.)):
                raise ValueError('Checkpoint has invalid numeric pose data')
            arrays[side] = poses, rotations, elbows
    resumed = copy.deepcopy(frames)
    for side, rig in rigs.items():
        for frame, vector, rotation, elbow in zip(resumed, *arrays[side]):
            _, root, points = rig.points(vector, rotation)
            hand = frame['hands'][side]
            hand.update(wrist=root.tolist(), angles=rig.angles(vector),
                        fingers=[np.asarray(p).tolist() for p in points],
                        pads=np.asarray(rig.effectors(points, rig._rots)).tolist(), elbow=elbow.tolist())
            hand['root'].update(pos=root.tolist(), rot=rotation.tolist())
            frame['body']['elbows'][side] = elbow.tolist()
    return resumed, metadata

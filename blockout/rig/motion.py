"""Anatomical interpolation for inspecting motion between exported poses.

Joints interpolate in their bounded coordinates; wrist rotations use SO(3).
The elbow stays on the exact two-bone circle instead of blending bone lengths.
Oversampled checks of this interpolation are empirical checks, not CCD proofs.
"""
import copy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .skin_contact import continuous_elbow


def rotation_angle(first, second):
    return float((Rotation.from_matrix(np.asarray(first)).inv()
                  * Rotation.from_matrix(np.asarray(second))).magnitude())


def blend_rotation(first, second, fraction):
    return Slerp([0., 1.], Rotation.from_matrix(np.array([first, second])))(fraction).as_matrix()


def bend_plane_angle(shoulder, wrist, elbow, reference):
    shoulder, wrist, elbow, reference = map(lambda v: np.asarray(v, float),
                                           (shoulder, wrist, elbow, reference))
    axis = wrist - shoulder
    axis /= np.linalg.norm(axis)
    def radial(point):
        direction = point - shoulder
        direction -= axis * (direction @ axis)
        length = np.linalg.norm(direction)
        if length < 1e-10:
            raise ValueError('Elbow plane is undefined at a straight-arm singularity')
        return direction / length
    first, second = radial(elbow), radial(reference)
    return float(np.arctan2(np.linalg.norm(np.cross(first, second)), np.clip(first @ second, -1., 1.)))


def interpolate_frame(first, second, fraction, rigs):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Interpolation fraction must be finite and within [0, 1]')
    blend = lambda a, b: (1. - fraction) * np.asarray(a, float) + fraction * np.asarray(b, float)
    frame = copy.deepcopy(first)
    frame['t'] = float(blend(first['t'], second['t']))
    frame['keys'] = {pitch: float(blend(first['keys'].get(pitch, 0.), second['keys'].get(pitch, 0.)))
                     for pitch in first['keys'].keys() | second['keys'].keys()}
    for field, value in first.get('body', {}).items():
        if field in ('shoulders', 'elbows'):
            frame['body'][field] = {side: blend(value[side], second['body'][field][side]).tolist()
                                     for side in value}
        elif field.endswith('_rot'):
            frame['body'][field] = blend_rotation(value, second['body'][field], fraction).tolist()
        elif isinstance(value, (list, tuple)):
            frame['body'][field] = blend(value, second['body'][field]).tolist()
    for side, rig in rigs.items():
        a, b = first['hands'][side], second['hands'][side]
        hand = frame['hands'][side]
        root = blend(a['root']['pos'], b['root']['pos'])
        rotation = blend_rotation(a['root']['rot'], b['root']['rot'], fraction)
        names = [(bone, dof) for chain in rig.chains for bone, dof, *_ in chain.dofs]
        q = np.array([blend(a['angles'].get(bone, {}).get(dof, 0.),
                            b['angles'].get(bone, {}).get(dof, 0.)) for bone, dof in names])
        vector = np.r_[root, np.zeros(3), q]
        _, _, points = rig.points(vector, rotation)
        hand.update(wrist=root.tolist(), angles=rig.angles(vector),
                    fingers=[np.asarray(p).tolist() for p in points],
                    pads=np.asarray(rig.effectors(points, rig._rots)).tolist(),
                    activation=blend(a['activation'], b['activation']).tolist())
        hand['root'].update(pos=root.tolist(), rot=rotation.tolist())
        shoulder = frame['body']['shoulders'][side]
        elbow = continuous_elbow(shoulder, root, blend(a['elbow'], b['elbow']))
        hand['elbow'] = elbow.tolist()
        frame['body']['elbows'][side] = elbow.tolist()
    return frame

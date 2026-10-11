"""NumPy evaluation of the exact normalized linear blend skin used by Blender.

This shares the anatomical FK, inverse binds and vertex weights with the
renderer. Surface points include mixed skin at joints and the palm webbing.
Subdivision is evaluated separately; a base-mesh check is not a render check.
"""
from pathlib import Path

import numpy as np

from .rig.skeleton import Skeleton, _rot


ASSET = Path(__file__).resolve().parent / 'hand_model' / 'skin_bind.npz'


def clean_hand_weights(weights, names):
    """Repair accidental opposite-hand weights, preserving mass and bone kind.

    The source mesh has one right-index vertex with a tiny left-middle weight.
    Such a vertex must not follow the independent other hand across the piano.
    Both the renderer and asset exporter apply this same correction.
    """
    weights = np.asarray(weights, float).copy()
    if weights.ndim != 2 or weights.shape[1] != len(names):
        raise ValueError('Expected one weight column per named bone')
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(1) <= 0):
        raise ValueError('Skin weights must be finite, nonnegative and supported')
    sides = np.array([name.rsplit('.', 1)[1] for name in names])
    owner = sides[weights.argmax(1)]
    ids = {name: i for i, name in enumerate(names)}
    rows, columns = np.nonzero((sides[None, :] != owner[:, None]) & (weights > 0))
    repairs = []
    for row, column in zip(rows, columns):
        target = names[column].rsplit('.', 1)[0] + '.' + owner[row]
        if target not in ids:
            raise ValueError('Opposite-hand skin weight has no matching bone')
        value = weights[row, column]
        repairs.append({'vertex': int(row), 'from': names[column], 'to': target, 'weight': float(value)})
        weights[row, ids[target]] += value
        weights[row, column] = 0.
    weights /= weights.sum(1)[:, None]
    return weights, repairs


def rotation_between(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cross = np.cross(first, second)
    sine, cosine = np.linalg.norm(cross), float(np.clip(first @ second, -1., 1.))
    if sine > 1e-12:
        return _rot(cross / sine, np.arctan2(sine, cosine))
    if cosine > 0:
        return np.eye(3)
    axis = np.cross(first, np.eye(3)[np.argmin(np.abs(first))])
    return _rot(axis / np.linalg.norm(axis), np.pi)


class Skin:
    def __init__(self, side, scale=None, asset=ASSET):
        self.side = side
        self.sk = Skeleton(side, scale=scale)
        with np.load(asset, allow_pickle=False) as data:
            names = data['bone_names'].tolist()
            self.vertex_ids = np.flatnonzero(data['sides'] == side)
            self.bone_ids = np.array([i for i, n in enumerate(names) if n.endswith('.' + side)])
            self.names = [names[i].rsplit('.', 1)[0] for i in self.bone_ids]
            self.bind = data['bind'][self.bone_ids]
            self.vertices = data['vertices'][self.vertex_ids]
            self.weights = data['weights'][np.ix_(self.vertex_ids, self.bone_ids)]
            if not np.allclose(self.weights.sum(1), 1., atol=1e-8):
                raise ValueError('Cross-hand or incomplete bone weights in skin asset')
            inverse = data['inverse_bind'][self.bone_ids]
            hom = np.column_stack([self.vertices, np.ones(len(self.vertices))])
            self.local = np.einsum('bij,nj->nbi', inverse, hom)[:, :, :3]
            # Weighted local coordinates let each trial use one batch contraction.
            self.weighted_local = self.local * self.weights[:, :, None]
            mapping = np.full(len(data['vertices']), -1, dtype=int)
            mapping[self.vertex_ids] = np.arange(len(self.vertex_ids))
            triangles = mapping[data['triangles']]
            self.triangles = triangles[np.all(triangles >= 0, axis=1)]
            sizes = data['face_sizes']
            offset, self.faces = 0, []
            for size in sizes:
                face = mapping[data['face_vertices'][offset:offset + size]]
                if np.all(face >= 0):
                    self.faces.append(face.tolist())
                elif np.any(face >= 0):
                    raise ValueError('Mesh polygon spans independent hand surfaces')
                offset += size
        self.bind_rot = {n: self.bind[i, :3, :3] for i, n in enumerate(self.names)}
        self.bone_index = {n: i for i, n in enumerate(self.names)}

    def transforms(self, root_rot, root_pos, angles, elbow):
        root_rot, root_pos, elbow = map(lambda v: np.asarray(v, float), (root_rot, root_pos, elbow))
        if not all(np.isfinite(v).all() for v in (root_rot, root_pos, elbow)):
            raise ValueError('Skin pose must be finite')
        length = np.linalg.norm(root_pos - elbow)
        if length < 1e-10:
            raise ValueError('Forearm direction must be nonzero')
        u = (root_pos - elbow) / length
        rotations, heads, _tails = self.sk.fk(root_rot, root_pos, angles)
        posed = np.tile(np.eye(4), (len(self.names), 1, 1))
        for i, name in enumerate(self.names):
            if name in rotations:
                posed[i, :3, :3] = rotations[name] * self.sk.scale
                posed[i, :3, 3] = heads[name]
        for name, back, share in (('lowerarm02', 0., .6),
                                  ('lowerarm01', self.sk.L['lowerarm02'], .2)):
            rest = self.bind_rot[name]
            r3 = rotation_between(rest[:, 1], u) @ rest
            rigid = root_rot @ self.bind_rot['wrist'].T @ rest
            za = r3[:, 2] - u * (r3[:, 2] @ u)
            zb = rigid[:, 2] - u * (rigid[:, 2] @ u)
            if np.linalg.norm(za) > 1e-6 and np.linalg.norm(zb) > 1e-6:
                za, zb = za / np.linalg.norm(za), zb / np.linalg.norm(zb)
                angle = np.arctan2(np.cross(za, zb) @ u, np.clip(za @ zb, -1., 1.))
                r3 = _rot(u, float(angle) * share) @ r3
            i = self.bone_index[name]
            posed[i, :3, :3] = r3
            posed[i, :3, 3] = root_pos - u * (self.sk.L['lowerarm02'] + back)
        return posed

    def pose(self, root_rot, root_pos, angles, elbow):
        matrices = self.transforms(root_rot, root_pos, angles, elbow)
        return (np.einsum('bij,nbj->ni', matrices[:, :3, :3], self.weighted_local)
                + self.weights @ matrices[:, :3, 3])

    def from_frame(self, hand):
        return self.pose(hand['root']['rot'], hand['root']['pos'], hand['angles'], hand['elbow'])

    def affected(self, finger):
        """Vertices influenced by any joint of a finger, including blended webbing."""
        ids = [i for i, n in enumerate(self.names) if n.startswith(f'finger{finger}-')]
        return np.flatnonzero(self.weights[:, ids].sum(1) > 1e-10)

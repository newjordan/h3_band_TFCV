"""Independent support measurements on the rendered distal finger surface.

These are geometric contact tolerances, not force or hammer-action simulation.
Collision clearance is a separate mandatory gate: finding a point at the top
does not excuse another part of the finger passing through the instrument.
"""
import numpy as np


class KeyTop:
    """Finite connected key-top union; internal head/stem seams are not edges.

    Signed distance is in the moving key's top plane, with positive values
    inside the physical outline. This does not establish exposed support or
    skin/instrument clearance; those remain independent requirements.
    """
    def __init__(self, scene, pitch):
        from .key_shapes import key_outline
        ids = [i for i, spec in enumerate(scene.specs) if spec.pitch == int(pitch)]
        if not 1 <= len(ids) <= 2:
            raise ValueError('A finite single-head or connected head/stem key is required')
        first = ids[0]
        self.origin, self.rotation = scene.origins[first], scene.rotations[first]
        self.lower, self.upper = scene.lower[ids, :2], scene.upper[ids, :2]
        if (not np.allclose(scene.origins[ids], self.origin, atol=1e-12, rtol=0.)
                or not np.allclose(scene.rotations[ids], self.rotation, atol=1e-12, rtol=0.)
                or not np.allclose(scene.upper[ids, 2], scene.upper[first, 2], atol=1e-12, rtol=0.)):
            raise ValueError('Key components must share one moving top plane')
        if len(ids) == 2:
            head, stem = (scene.specs[i] for i in ids)
            if (abs(head.upper[1] - stem.lower[1]) > 1e-12
                    or stem.lower[0] < head.lower[0] or stem.upper[0] > head.upper[0]):
                raise ValueError('Unsupported disconnected key components')
        polygon = key_outline([scene.specs[i] for i in ids]) - self.origin[:2]
        self.starts, self.edges = polygon, np.roll(polygon, -1, axis=0) - polygon
        self.squared_length = np.einsum('ij,ij->i', self.edges, self.edges)
        if np.any(self.squared_length <= 0):
            raise ValueError('Physical key boundary must have nonzero edges')

    def margin(self, point):
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('A finite world point is required')
        local = ((point - self.origin) @ self.rotation)[:2]
        inside = bool(np.any(np.all((local >= self.lower) & (local <= self.upper), axis=1)))
        delta = local - self.starts
        fraction = np.clip(np.einsum('ij,ij->i', delta, self.edges) / self.squared_length, 0., 1.)
        distance = float(np.linalg.norm(delta - fraction[:, None] * self.edges, axis=1).min())
        return distance if inside else -distance


def key_support(vertices, distal_indices, pitch, scene, contact_gap=.0001, tolerance=1e-5):
    """Find actual distal skin near the exposed, moving top of an assigned key.

    A black solid blocks the white top where a ray along the white key's normal
    crosses it. All transforms use the actual hinged geometry. Skin below the
    key's underside or outside its finite footprint cannot count as support.
    """
    if (not np.isfinite(contact_gap) or contact_gap < 0
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Contact tolerances must be finite and nonnegative')
    points = np.asarray(vertices, float)[np.asarray(distal_indices, int)]
    if not np.isfinite(points).all():
        raise ValueError('Skin positions must be finite')
    boxes = [i for i, spec in enumerate(scene.specs) if spec.pitch == int(pitch)]
    best = None
    for box in boxes:
        local = (points - scene.origins[box]) @ scene.rotations[box]
        inside = np.all((local[:, :2] >= scene.lower[box, :2])
                        & (local[:, :2] <= scene.upper[box, :2]), axis=1)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        if scene.specs[box].material == 'white':
            top = local[ids].copy()
            top[:, 2] = scene.upper[box, 2]
            world = top @ scene.rotations[box].T + scene.origins[box]
            normal = scene.rotations[box][:, 2]
            exposed = np.ones(len(ids), bool)
            for other, spec in enumerate(scene.specs):
                if spec.material != 'black':
                    continue
                ray_origin = (world - scene.origins[other]) @ scene.rotations[other]
                direction = normal @ scene.rotations[other]
                near, far = np.full(len(ids), -np.inf), np.full(len(ids), np.inf)
                possible = np.ones(len(ids), bool)
                for axis in range(3):
                    if abs(direction[axis]) < 1e-12:
                        possible &= ((ray_origin[:, axis] >= scene.lower[other, axis])
                                     & (ray_origin[:, axis] <= scene.upper[other, axis]))
                    else:
                        a = (scene.lower[other, axis] - ray_origin[:, axis]) / direction[axis]
                        b = (scene.upper[other, axis] - ray_origin[:, axis]) / direction[axis]
                        near = np.maximum(near, np.minimum(a, b))
                        far = np.minimum(far, np.maximum(a, b))
                exposed &= ~(possible & (far >= np.maximum(near, 0.)) & (far > tolerance))
            ids = ids[exposed]
        if not len(ids):
            continue
        gaps = local[ids, 2] - scene.upper[box, 2]
        valid = (gaps >= -tolerance) & (gaps <= contact_gap + tolerance)
        # Prefer an actual supported point; otherwise retain the closest gap
        # as diagnostic evidence rather than inventing a contact.
        chosen = np.flatnonzero(valid)
        chosen = chosen[np.argmin(np.abs(gaps[chosen]))] if len(chosen) else int(np.argmin(np.abs(gaps)))
        candidate = dict(supported=bool(valid[chosen]), vertex=int(np.asarray(distal_indices)[ids[chosen]]),
                         top_gap_mm=float(gaps[chosen] * 1000.), box=scene.specs[box].name)
        if best is None or (candidate['supported'], -abs(candidate['top_gap_mm'])) > (best['supported'], -abs(best['top_gap_mm'])):
            best = candidate
    return best if best is not None else dict(supported=False, vertex=None, top_gap_mm=None, box=None)

"""Independent support measurements on the rendered distal finger surface.

These are geometric contact tolerances, not force or hammer-action simulation.
Collision clearance is a separate mandatory gate: finding a point at the top
does not excuse another part of the finger passing through the instrument.
"""
import numpy as np


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

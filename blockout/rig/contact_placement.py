"""Human-calibrated contact depth on finite, exposed moving key tops."""
import json
from pathlib import Path

import numpy as np

from ..skin_contacts import key_support


REFERENCE = Path(__file__).with_name('pianist_reference.json')


def surface_target(point, pitch, scene, margin=.001):
    """Choose the nearest exposed component, preserving notches and stems."""
    point = np.asarray(point, float)
    if point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(margin) or margin < 0:
        raise ValueError('Finite point and nonnegative footprint margin required')
    candidates = []
    for box, spec in enumerate(scene.specs):
        if spec.pitch != int(pitch):
            continue
        lo, hi = scene.lower[box, :2] + margin, scene.upper[box, :2] - margin
        if np.any(lo > hi):
            continue
        local = (point - scene.origins[box]) @ scene.rotations[box]
        local[:2] = np.clip(local[:2], lo, hi)
        local[2] = scene.upper[box, 2]
        world = local @ scene.rotations[box].T + scene.origins[box]
        if key_support(world[None], [0], pitch, scene)['supported']:
            candidates.append((float(np.linalg.norm(world - point)), box, world))
    if not candidates:
        raise ValueError(f'No finite exposed surface for pitch {pitch}')
    _, box, target = min(candidates, key=lambda value: value[0])
    return box, target


def calibrated_target(finger, pitch, scene, keys, active_pitches=(), reference=REFERENCE, margin=.001):
    if finger not in range(1, 6):
        raise ValueError('Finger must be in 1–5')
    key = keys[int(pitch)]
    active = set(map(int, active_pitches))
    kind = 'black' if key['black'] else ('white_near_black' if any(
        p in active and p in keys and keys[p]['black'] for p in (int(pitch) - 1, int(pitch) + 1)) else 'white')
    data = json.loads(Path(reference).read_text()) if not isinstance(reference, dict) else reference
    statistics = data['strike'].get(f'f{finger}.{kind}')
    fallback = statistics is None
    if fallback:
        statistics = data['strike'][f'f{finger}.white']
    depth = float(statistics['tip_y']['p50'])
    if not np.isfinite(depth):
        raise ValueError('Contact depth must be finite')
    candidates = []
    for box, spec in enumerate(scene.specs):
        if spec.pitch != int(pitch):
            continue
        rotation, origin = scene.rotations[box], scene.origins[box]
        lo, hi = scene.lower[box, :2] + margin, scene.upper[box, :2] - margin
        if np.any(lo > hi) or abs(rotation[1, 1]) < 1e-8:
            continue
        # Reference depths are world coordinates measured while pressing.
        # Solve the hinged top's actual y, rather than treating the rest top
        # as horizontal or clamping a deep contact to the front head.
        local = np.array([key['x'] - origin[0],
            (depth - origin[1] - rotation[1, 2] * scene.upper[box, 2]) / rotation[1, 1], scene.upper[box, 2]])
        local[:2] = np.clip(local[:2], lo, hi)
        world = local @ rotation.T + origin
        if key_support(world[None], [0], pitch, scene)['supported']:
            displacement = float(np.linalg.norm(world[:2] - [key['x'], depth]))
            candidates.append((displacement, box, world))
    if not candidates:
        raise ValueError(f'No calibrated exposed contact for pitch {pitch}')
    displacement, box, target = min(candidates, key=lambda value: value[0])
    metadata = dict(kind=kind, fallback=fallback, samples=statistics['tip_y']['n'],
                    desired_depth_mm=depth * 1000., achieved_depth_mm=float(target[1] * 1000.),
                    finite_surface_adjustment_mm=displacement * 1000., component=scene.specs[box].name,
                    reference_scope='pooled MANO fingertip depth mapped to actual skin, not subject/per-key force data')
    return box, target, metadata

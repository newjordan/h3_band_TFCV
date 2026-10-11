"""Preserve physical pose precision and explicit hand-size identity in JSON."""
import numpy as np


def animation_hand_scale(animation, requested=None, legacy_default=None):
    """Reject conflicting size overrides; legacy files need an explicit size.

    The renderer can supply its established legacy default. Auditors supply the
    scale of their frozen input. A tagged animation is never silently resized.
    """
    tagged = animation.get('hand_scale')
    values = [v for v in (tagged, requested, legacy_default) if v is not None]
    if any(isinstance(v, bool) or not np.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError('Hand scale must be finite and positive')
    if tagged is not None:
        if requested is not None and not np.isclose(float(tagged), float(requested), atol=1e-9, rtol=0.):
            raise ValueError('Animation hand scale conflicts with the requested scale; regenerate the pose')
        return float(tagged)
    if requested is not None:
        return float(requested)
    if legacy_default is not None:
        return float(legacy_default)
    raise ValueError('Legacy animation has no hand scale; an explicit frozen-input scale is required')


def serialize_hand(rig, vector, rotation, root, points, pads, activation, assigned_keys, plan):
    """Export full solver precision, keeping wrist/root and FK identity intact."""
    arrays = [np.asarray(a, float) for a in (vector, rotation, root, points, pads, activation, plan)]
    vector, rotation, root, points, pads, activation, plan = arrays
    if (any(not np.isfinite(a).all() for a in arrays)
            or vector.shape != (len(rig.lo) + 6,) or rotation.shape != (3, 3)
            or root.shape != (3,) or points.shape != (5, 4, 3) or pads.shape != (5, 3)
            or activation.shape != (5,) or plan.shape != (3,) or len(assigned_keys) != 5):
        raise ValueError('Finite complete physical hand pose required')
    return dict(wrist=root.tolist(), activation=activation.tolist(), assigned_keys=list(assigned_keys),
        fingers=points.tolist(), pads=pads.tolist(), root=dict(rot=rotation.tolist(), pos=root.tolist(), plan=plan.tolist()),
        angles={bone: {dof: float(value) for dof, value in dofs.items()} for bone, dofs in rig.angles(vector).items()})

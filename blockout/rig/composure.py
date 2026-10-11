"""Opt-in wrist-carried free digits and isolated lateral reach references.

These are bounded kinematic controller preferences. They do not model muscle
forces or certify the rendered skin's collision-free motion.
"""
import numpy as np


def relaxed_pads(rig, rotation, position):
    vector = np.concatenate([position, np.zeros(3), rig.q_rest])
    _, _, points = rig.points(vector, rotation)
    return np.asarray(rig.effectors(points, rig._rots))


def quiet_pad_targets(rig, rotation, position, targets, pressing, quiet,
                      floor, preparation_limit=.004, clearance=.003):
    """Carry a curled pose with the wrist; bound preparation before an attack.

    Targets are bottom-of-pad FK points, not sphere centres. The instrument
    floor can raise a free target; collision resolution still belongs to IK.
    Assigned digits retain their complete attack/hold/release targets.
    """
    result = np.array(targets, float, copy=True)
    rest = relaxed_pads(rig, rotation, position)
    for finger in range(5):
        if pressing[finger]:
            continue
        delta = result[finger] - rest[finger]
        length = np.linalg.norm(delta)
        delta *= min(1., preparation_limit / max(length, 1e-12))
        ready = 1. - float(np.clip(quiet[finger], 0., 1.))
        result[finger] = rest[finger] + ready * delta
        result[finger, 2] = max(result[finger, 2],
                                floor(*result[finger, :2], finger) + clearance)
    return result


def solo_wrist_reference(rig, rotation, position, targets, assigned,
                         activation, other_score_note_held=False, limit=.12):
    """Translate toward one assigned pad while keeping the relaxed abduction.

    Another held score note vetoes this preference. The whole-hand solve and
    trajectory smoothing follow; this reference is not an accepted wrist pose.
    """
    occupied = [f for f, pitch in enumerate(assigned) if pitch is not None]
    result = np.array(position, float, copy=True)
    if len(occupied) != 1 or other_score_note_held:
        return result
    finger = occupied[0]
    pad = relaxed_pads(rig, rotation, position)[finger]
    gain = float(np.clip(activation[finger], 0., 1.))
    gain = gain * gain * (3. - 2. * gain)
    result[0] += gain * np.clip(targets[finger, 0] - pad[0], -limit, limit)
    return result

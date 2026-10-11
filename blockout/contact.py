"""Contact geometry and continuous activation for the piano controller.

Activation is a dimensionless controller gain, not a claim about muscle force.
The key has a finite exposed top and a hinged, finite travel. A pad outside
that top cannot actuate it, even if the score assigns that key to the finger.
"""
import numpy as np


def smooth01(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def activation(t, event):
    """Firm by the downstroke, supported hold, then a gradual release.

    Keep a nonzero support gain throughout the hold. Loudness is expressed by
    the approach/downstroke trajectory, never by pushing below the key bed.
    """
    on, off, _pitch, _velocity, _black, touch, _lift = event
    touch = float(np.clip(touch, 0.0, 1.0))
    strike = 0.080 - 0.050 * touch
    approach = 0.10 + 0.08 * touch
    release = 0.10 - 0.05 * touch
    if t < on - strike:
        return smooth01((t - (on - strike - approach)) / approach)
    if t <= on:
        return 1.0
    if t <= off:
        return 1.0 - 0.2 * smooth01((t - on) / 0.08)
    held = 1.0 - 0.2 * smooth01((off - on) / 0.08)
    return held * (1.0 - smooth01((t - off) / release))


def on_key(pitch, pad, keys, margin=0.0):
    """Is a pad on the exposed top of its key? Black keys occlude whites.

    `keys` is the keyboard geometry mapping keyed by MIDI pitch. The small
    physical gaps in Blender are deliberately not converted to extra reach.
    """
    k = keys.get(int(pitch))
    if k is None or not np.all(np.isfinite(pad)):
        return False
    x, y = float(pad[0]), float(pad[1])
    gap = k.get('gap', 0.0 if k['black'] else 0.0006)
    if abs(x - k['x']) > k['w'] / 2 - gap - margin:
        return False
    if not k['y0'] + margin <= y < k['y0'] + k['l'] - margin:
        return False
    if not k['black']:
        for p in (pitch - 1, pitch + 1):
            b = keys.get(int(p))
            if b and b['black'] and b['y0'] <= y < b['y0'] + b['l'] and abs(x - b['x']) <= b['w'] / 2:
                return False
    return True


def key_depth(pitch, pad, keys, travel):
    """Unilateral pad/key contact: depth in [0, 1], zero off its key.

    This is a kinematic contact constraint; it does not simulate hammer action.
    Penetration past the bed must be measured separately, not hidden by depth
    saturation. No finite lever arm is invented at the rear pivot.
    """
    if not on_key(pitch, pad, keys):
        return 0.0
    k = keys[int(pitch)]
    lever = (k['y0'] + k['l'] - float(pad[1])) / k['l']
    return float(np.clip((k['top'] - float(pad[2])) / (travel * lever), 0.0, 1.0))

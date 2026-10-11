"""Notched-key geometry prototype preserving front widths, travel and hinge.

The old rectangular whites intersect lowered black solids. This prototype
keeps a full-width front and a narrower rear stem. Cutouts cover the black
solid's swept extent and the white head's swept rear extent over depth [0,1].
The existing short rear hinge is retained for a controlled shape experiment;
a calibrated balance-pivot/action/force model remains a separate requirement.
"""
import numpy as np

from dataclasses import replace

from .piano_geometry import BoxSpec, PianoScene, piano_boxes


def notched_boxes(keyboard, music_desk=False, clearance=.0001):
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError('A finite positive mechanical clearance is required')
    source = piano_boxes(keyboard, music_desk)
    keys = {int(k['pitch']): k for k in keyboard}
    result = []
    # The old bed sits only 1 mm below the resting 18 mm key body, so a
    # 10 mm stroke drives that body through it. Put the stop plane at the
    # actual lowest moving corner at full depth; do not exempt that contact.
    bed_top = min((spec.lower[1] - spec.origin[1]) * np.sin(.010 / spec.length)
                  + spec.lower[2] * np.cos(.010 / spec.length)
                  for spec in source if spec.pitch is not None)
    for spec in source:
        if spec.name == 'keybed':
            result.append(replace(spec, upper=(spec.upper[0], spec.upper[1], bed_top)))
            continue
        if spec.material != 'white':
            result.append(spec)
            continue
        neighbors = [keys[p] for p in (spec.pitch - 1, spec.pitch + 1)
                     if p in keys and keys[p]['black']]
        if not neighbors:
            result.append(spec)
            continue
        pivot = spec.origin[1]
        white_angle = .010 / spec.length
        black_front = []
        for black in neighbors:
            length = pivot - black['y0']
            maximum = .010 / length
            # The top/front corner gives minimum y. Evaluate the derivative's
            # zero as well as the stops, so the bound covers the whole sweep.
            critical = np.arctan2(black['top'], length)
            angles = [0., maximum, np.clip(critical, 0., maximum)]
            black_front.append(min(pivot - length * np.cos(a) - black['top'] * np.sin(a) for a in angles))
        # White rear/bottom corner moves farthest rearward at the lower stop.
        end = pivot + (min(black_front) - clearance - pivot
                       + spec.lower[2] * np.sin(white_angle)) / np.cos(white_angle)
        left, right = spec.lower[0], spec.upper[0]
        for black in neighbors:
            if black['x'] < keys[spec.pitch]['x']:
                left = max(left, black['x'] + black['w'] / 2 + clearance)
            else:
                right = min(right, black['x'] - black['w'] / 2 - clearance)
        if not spec.lower[1] < end < spec.upper[1] or not left < right:
            raise ValueError('Keyboard dimensions do not permit the notched shape')
        result.append(BoxSpec(spec.name, spec.lower, (spec.upper[0], end, spec.upper[2]),
                              spec.material, spec.origin, spec.pitch, spec.length))
        result.append(BoxSpec(spec.name + '_stem', (left, end, spec.lower[2]),
                              (right, spec.upper[1], spec.upper[2]), spec.material,
                              spec.origin, spec.pitch, spec.length))
    return result


def key_outline(parts):
    """CCW outline of one front/stem union; repeated corners are removed."""
    if len(parts) == 1:
        p = parts[0]
        return np.array([[p.lower[0], p.lower[1]], [p.upper[0], p.lower[1]],
                         [p.upper[0], p.upper[1]], [p.lower[0], p.upper[1]]])
    head, stem = parts
    points = np.array([[head.lower[0], head.lower[1]], [head.upper[0], head.lower[1]],
                       [head.upper[0], head.upper[1]], [stem.upper[0], stem.lower[1]],
                       [stem.upper[0], stem.upper[1]], [stem.lower[0], stem.upper[1]],
                       [stem.lower[0], stem.lower[1]], [head.lower[0], head.upper[1]]])
    return points[np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > 1e-12]


class NotchedPianoScene(PianoScene):
    """Experimental union-of-boxes scene; no production geometry is replaced."""
    def __init__(self, keyboard, depths=None, music_desk=False, clearance=.0001):
        self.specs = notched_boxes(keyboard, music_desk, clearance)
        depths = {int(k): float(v) for k, v in (depths or {}).items()}
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in depths.values()):
            raise ValueError('Key travel must stay inside the physical stops')
        self.rotations, self.origins, self.lower, self.upper = [], [], [], []
        for spec in self.specs:
            origin = np.array(spec.origin if spec.origin is not None else
                              (np.array(spec.lower) + spec.upper) / 2, float)
            angle = 0. if spec.pitch is None else depths.get(spec.pitch, 0.) * .010 / spec.length
            c, s = np.cos(angle), np.sin(angle)
            self.rotations.append([[1., 0., 0.], [0., c, -s], [0., s, c]])
            self.origins.append(origin)
            self.lower.append(np.asarray(spec.lower) - origin)
            self.upper.append(np.asarray(spec.upper) - origin)
        self.rotations, self.origins, self.lower, self.upper = map(np.asarray,
            (self.rotations, self.origins, self.lower, self.upper))
        self.centers = (self.lower + self.upper) / 2
        self.half = (self.upper - self.lower) / 2
        if np.any(self.half <= 0):
            raise ValueError('All solids require positive volume')
        centers = self.origins + np.einsum('bij,bj->bi', self.rotations, self.centers)
        extent = np.einsum('bij,bj->bi', np.abs(self.rotations), self.half)
        self.aabb_lower, self.aabb_upper = centers - extent, centers + extent

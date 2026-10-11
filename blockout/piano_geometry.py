"""Shared finite solids for the piano renderer and contact checks (metres).

The skin solver must clear key sides, gaps and casework as well as key tops.
No assigned finger gets a penetration exemption. Key travel rotates the whole
solid about its rear hinge, exactly as in the renderer.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxSpec:
    name: str
    lower: tuple
    upper: tuple
    material: str
    origin: tuple | None = None
    pitch: int | None = None
    length: float | None = None


def piano_boxes(keyboard, music_desk=False):
    if not keyboard:
        raise ValueError('A piano scene requires a keyboard')
    xs = [key['x'] for key in keyboard]
    left, right = min(xs) - .02, max(xs) + .02
    boxes = []
    for key in keyboard:
        x, width, pitch = key['x'], key['w'], int(key['pitch'])
        if key['black']:
            lower, upper = (x - width / 2, key['y0'], 0.), (x + width / 2, .150, key['top'])
        else:
            lower, upper = (x - width / 2 + .0006, 0., -.018), (x + width / 2 - .0006, .150, 0.)
        boxes.append(BoxSpec(f'key{pitch}', lower, upper, 'black' if key['black'] else 'white',
                             (x, .150, 0.), pitch, .150 - key['y0']))
    for name, lower, upper in (
            ('keybed', (left - .04, -.02, -.10), (right + .04, .30, -.019)),
            ('front_rail', (left - .04, -.03, -.10), (right + .04, -.004, -.012)),
            ('cheek_l', (left - .06, -.03, -.10), (left - .005, .30, .045)),
            ('cheek_r', (right + .005, -.03, -.10), (right + .06, .30, .045)),
            ('fallboard', (left - .005, .153, -.02), (right + .005, .20, .055)),
            ('lid_body', (left - .06, .20, -.10), (right + .06, .32, .07))):
        boxes.append(BoxSpec(name, lower, upper, 'case'))
    if music_desk:
        boxes.append(BoxSpec('music_desk', (left + .45, .22, .07), (right - .45, .25, .30), 'case'))
    boxes.append(BoxSpec('floor', (left - 3., -3., -.80), (right + 3., 4., -.79), 'floor'))
    return boxes


class PianoScene:
    def __init__(self, keyboard, depths=None, music_desk=False):
        self.specs = piano_boxes(keyboard, music_desk)
        depths = {int(k): float(v) for k, v in (depths or {}).items()}
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in depths.values()):
            raise ValueError('Key travel must be finite and within the physical stop')
        self.rotations, self.origins, self.lower, self.upper = [], [], [], []
        for spec in self.specs:
            origin = np.array(spec.origin if spec.origin is not None else
                              (np.array(spec.lower) + spec.upper) / 2, float)
            angle = 0. if spec.pitch is None else depths.get(spec.pitch, 0.) * .010 / spec.length
            c, s = np.cos(angle), np.sin(angle)
            self.rotations.append([[1., 0., 0.], [0., c, -s], [0., s, c]])
            self.origins.append(origin)
            self.lower.append(np.array(spec.lower) - origin)
            self.upper.append(np.array(spec.upper) - origin)
        self.rotations, self.origins, self.lower, self.upper = map(np.array,
            (self.rotations, self.origins, self.lower, self.upper))
        self.centers = (self.lower + self.upper) / 2
        self.half = (self.upper - self.lower) / 2
        if np.any(self.half <= 0) or not all(np.isfinite(v).all() for v in
                (self.rotations, self.origins, self.lower, self.upper)):
            raise ValueError('Scene solids must have finite, positive volume')
        world_centers = self.origins + np.einsum('bij,bj->bi', self.rotations, self.centers)
        extent = np.einsum('bij,bj->bi', np.abs(self.rotations), self.half)
        self.aabb_lower, self.aabb_upper = world_centers - extent, world_centers + extent

    def candidates(self, points, padding=0.):
        points = np.asarray(points, float)
        return np.flatnonzero(np.all(self.aabb_upper >= points.min(0) - padding, axis=1)
                              & np.all(self.aabb_lower <= points.max(0) + padding, axis=1))

    def distances(self, points, boxes=None):
        """Euclidean signed distances: negative inside, zero on a solid face."""
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Contact points must be finite Nx3 coordinates')
        ids = np.arange(len(self.specs)) if boxes is None else np.asarray(boxes, int)
        local = np.einsum('nbi,bij->nbj', points[:, None, :] - self.origins[ids], self.rotations[ids])
        q = np.abs(local - self.centers[ids]) - self.half[ids]
        return np.linalg.norm(np.maximum(q, 0.), axis=2) + np.minimum(q.max(axis=2), 0.)

    def closest(self, points, boxes=None):
        ids = np.arange(len(self.specs)) if boxes is None else np.asarray(boxes, int)
        if not len(ids):
            return np.full(len(points), np.inf), np.full(len(points), -1, dtype=int)
        distances = self.distances(points, ids)
        column = distances.argmin(1)
        return distances[np.arange(len(points)), column], ids[column]

    def triangle_contacts(self, vertices, triangles, tolerance=1e-5):
        """Triangle/solid intersection audit, including edges with no inside vertex.

        The box is eroded by tolerance before SAT. This rejects harmless contact
        and detects penetration into the interior deeper than that tolerance.
        This is a static mesh audit; it does not prove clearance between frames.
        """
        vertices, triangles = np.asarray(vertices, float), np.asarray(triangles, int)
        geometry = vertices[triangles]
        tri_lower, tri_upper = geometry.min(1), geometry.max(1)
        contacts = []
        for box in self.candidates(vertices):
            candidates = np.flatnonzero(np.all(tri_upper >= self.aabb_lower[box], axis=1)
                                         & np.all(tri_lower <= self.aabb_upper[box], axis=1))
            local = (geometry[candidates] - self.origins[box]) @ self.rotations[box]
            hit = triangle_box_overlap(local, self.lower[box], self.upper[box], tolerance)
            if np.any(hit):
                contacts.append((int(box), candidates[hit]))
        return contacts


def triangle_box_overlap(triangles, lower, upper, tolerance=1e-5):
    """13-axis separating-axis test, triangles already in the box's local frame."""
    triangles, lower, upper = map(lambda v: np.asarray(v, float), (triangles, lower, upper))
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError('Triangle coordinates must be Nx3x3')
    if not all(np.isfinite(v).all() for v in (triangles, lower, upper)):
        raise ValueError('Triangle and box geometry must be finite')
    if tolerance < 0 or not np.isfinite(tolerance) or np.any(upper <= lower):
        raise ValueError('Invalid intersection tolerance or box')
    half, center = (upper - lower) / 2 - tolerance, (upper + lower) / 2
    if np.any(half <= 0):
        return np.zeros(len(triangles), dtype=bool)
    local = triangles - center
    edges = np.roll(local, -1, axis=1) - local
    normal = np.cross(edges[:, 0], edges[:, 1])
    axis = np.broadcast_to(np.eye(3), (len(local), 3, 3))
    cross = np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate([axis, normal[:, None, :], cross], axis=1)
    projection = np.einsum('nvi,nai->nva', local, axes)
    radius = np.abs(axes) @ half
    separated = (projection.min(1) > radius) | (projection.max(1) < -radius)
    return ~separated.any(1)


def triangle_box_margin(triangles, lower, upper):
    """Largest normalized separating-axis margin; positive proves separation.

    This is a SAT constraint, not Euclidean distance. Unlike isolated vertex or
    clipped-point constraints, it checks the entire triangle against the box.
    Degenerate edge-cross axes must be ignored rather than supplying false zero
    margins that would hide a coplanar triangle inside the box.
    """
    triangles, lower, upper = map(lambda v: np.asarray(v, float), (triangles, lower, upper))
    center, half = (lower + upper) / 2, (upper - lower) / 2
    local = triangles - center
    edges = np.roll(local, -1, axis=1) - local
    axes = np.concatenate([np.broadcast_to(np.eye(3), (len(local), 3, 3)),
                           np.cross(edges[:, 0], edges[:, 1])[:, None, :],
                           np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(-1, 9, 3)], axis=1)
    length = np.linalg.norm(axes, axis=2)
    axes = axes / np.maximum(length[:, :, None], 1e-30)
    projection = np.einsum('nvi,nai->nva', local, axes)
    radius = np.abs(axes) @ half
    margin = np.maximum(projection.min(1) - radius, -projection.max(1) - radius)
    margin[length <= 1e-12] = -np.inf
    return margin.max(1)


def triangle_box_witness(triangle, lower, upper):
    """A barycentric point inside a triangle/box intersection, or None.

    Clipping retains barycentric coordinates so contact can become a linear
    surface stencil in a subsequent constrained solve, rather than a fake point.
    """
    polygon = [(np.asarray(point, float), np.eye(3)[i]) for i, point in enumerate(triangle)]
    for axis in range(3):
        for bound, sign in ((lower[axis], 1.), (upper[axis], -1.)):
            clipped = []
            if not polygon:
                return None
            for i, (current, bc) in enumerate(polygon):
                previous, bp = polygon[i - 1]
                dc, dp = sign * (current[axis] - bound), sign * (previous[axis] - bound)
                if (dc >= 0) != (dp >= 0):
                    fraction = dp / (dp - dc)
                    clipped.append((previous + fraction * (current - previous), bp + fraction * (bc - bp)))
                if dc >= 0:
                    clipped.append((current, bc))
            polygon = clipped
    if not polygon:
        return None
    return np.mean([barycentric for _, barycentric in polygon], axis=0)

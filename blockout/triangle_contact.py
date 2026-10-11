"""Static triangle intersections, distinguishing expected shared topology.

BVH overlap is only a candidate list. SAT and plane/polygon intersection
confirm the actual triangles. Sharing a vertex is never a blanket exemption:
an adjacent face can fold across another face away from that vertex or edge.
This is not continuous collision detection, a containment or distance proof.
"""
import numpy as np


def triangle_overlap(first, second, tolerance=1e-7):
    a, b = np.asarray(first, float), np.asarray(second, float)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Finite paired triangles and positive tolerance required')
    ea, eb = np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    for edges, normals in ((ea, na), (eb, nb)):
        if np.any(np.linalg.norm(normals, axis=1) <= tolerance * np.linalg.norm(edges, axis=2).max(axis=1)):
            raise ValueError('Degenerate triangles require a separate invalid-geometry gate')
    axes = np.concatenate([na[:, None], nb[:, None],
        np.cross(ea[:, :, None], eb[:, None]).reshape(-1, 9, 3),
        np.cross(na[:, None], ea), np.cross(nb[:, None], eb)], axis=1)
    length = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(length[:, :, None], 1e-30)
    # Centering reduces projection error after a large world translation.
    pa = np.einsum('nac,nvc->nav', axes, a - a[:, :1])
    pb = np.einsum('nac,nvc->nav', axes, b - a[:, :1])
    separated = (pa.max(2) < pb.min(2) - tolerance) | (pb.max(2) < pa.min(2) - tolerance)
    return ~np.any(separated & (length > 1e-20), axis=1)


def triangle_separation(first, second):
    """Signed SAT separation for nonincident, nondegenerate triangle pairs.

    Positive is a separating projection gap; negative indicates overlapping
    projections. This is an active-set constraint, not Euclidean distance or
    a replacement for the complete geometric intersection audit.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    ea, eb = np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    axes = np.concatenate([na[:, None], nb[:, None],
        np.cross(ea[:, :, None], eb[:, None]).reshape(-1, 9, 3),
        np.cross(na[:, None], ea), np.cross(nb[:, None], eb)], axis=1)
    length = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(length[:, :, None], 1e-30)
    pa = np.einsum('nac,nvc->nav', axes, a - a[:, :1])
    pb = np.einsum('nac,nvc->nav', axes, b - a[:, :1])
    separation = np.maximum(pa.min(2) - pb.max(2), pb.min(2) - pa.max(2))
    return np.max(np.where(length > 1e-20, separation, -np.inf), axis=1)


def _section(triangle, origin, normal, tolerance):
    distance = (triangle - origin) @ normal
    points = [p for p, d in zip(triangle, distance) if abs(d) <= tolerance]
    for i in range(3):
        j = (i + 1) % 3
        if distance[i] * distance[j] < 0:
            points.append(triangle[i] + distance[i] / (distance[i] - distance[j]) * (triangle[j] - triangle[i]))
    return np.asarray(points, float).reshape(-1, 3)


def _coplanar_polygon(a, b, normal, tolerance):
    drop = int(np.abs(normal).argmax())
    keep = [i for i in range(3) if i != drop]
    origin = a[0]
    polygon, clip = (a - origin)[:, keep].tolist(), (b - origin)[:, keep]
    cross2 = lambda u, v: u[0] * v[1] - u[1] * v[0]
    orientation = np.sign(cross2(clip[1] - clip[0], clip[2] - clip[0]))
    for i in range(3):
        if not polygon:
            break
        edge = clip[(i + 1) % 3] - clip[i]
        length = np.linalg.norm(edge)
        signed = lambda p: orientation * cross2(edge, np.asarray(p) - clip[i]) / length
        output = []
        start, ds = polygon[-1], signed(polygon[-1])
        for end in polygon:
            de = signed(end)
            if (ds >= -tolerance) != (de >= -tolerance) and abs(ds - de) > 1e-30:
                output.append((np.asarray(start) + ds / (ds - de) * (np.asarray(end) - start)).tolist())
            if de >= -tolerance:
                output.append(end)
            start, ds = end, de
        polygon = output
    points = np.zeros((len(polygon), 3))
    if len(polygon):
        points[:, keep] = np.asarray(polygon)
        points[:, drop] = -(points @ normal) / normal[drop]
        points += origin
    area = 0.
    if len(points) >= 3:
        delta = points - points[0]
        area = abs(sum(float(np.cross(delta[i], delta[(i + 1) % len(points)]) @ normal)
                       for i in range(len(points)))) * .5
    return points, area


def intersection_relation(first, second, first_ids=(), second_ids=(), tolerance=1e-7, sat_confirmed=False):
    """Return a geometric relation; only shared-feature contact is exempted."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    if not sat_confirmed and not triangle_overlap(a[None], b[None], tolerance)[0]:
        return dict(kind='separate', unexpected=False)
    shared = sorted(set(first_ids) & set(second_ids))
    if len(shared) == 3:
        return dict(kind='duplicate_face', unexpected=True)
    normals = [np.cross(t[1] - t[0], t[2] - t[0]) for t in (a, b)]
    na, nb = [n / np.linalg.norm(n) for n in normals]
    # Broad-phase tolerance admits near misses. It must not turn a vertex
    # near another plane into a vertex on that plane: that can fabricate a
    # long crossing when the planes are almost parallel.
    geometric_epsilon = min(tolerance, 1e-12)
    line = np.cross(na, nb)
    coplanar = np.linalg.norm(line) < 1e-8
    area, length = 0., 0.
    if coplanar:
        if np.max(np.abs((b - a[0]) @ na)) > geometric_epsilon:
            return dict(kind='separate', unexpected=False)
        points, area = _coplanar_polygon(a, b, na, geometric_epsilon)
        if not len(points):
            return dict(kind='separate', unexpected=False)
    else:
        line /= np.linalg.norm(line)
        pa, pb = _section(a, b[0], nb, geometric_epsilon), _section(b, a[0], na, geometric_epsilon)
        if not len(pa) or not len(pb):
            return dict(kind='separate', unexpected=False)
        origin = a[0]
        ta, tb = (pa - origin) @ line, (pb - origin) @ line
        lower, upper = max(ta.min(), tb.min()), min(ta.max(), tb.max())
        if lower > upper + geometric_epsilon:
            return dict(kind='separate', unexpected=False)
        length = max(0., float(upper - lower))
        points = pa[0] + (np.array([lower, upper]) - ta[0])[:, None] * line
    if shared:
        common = np.asarray([a[list(first_ids).index(i)] for i in shared])
        if len(shared) == 1:
            expected = np.all(np.linalg.norm(points - common[0], axis=1) <= tolerance * 4.)
        else:
            edge = common[1] - common[0]
            along = np.clip((points - common[0]) @ edge / (edge @ edge), 0., 1.)
            expected = np.all(np.linalg.norm(points - common[0] - along[:, None] * edge, axis=1) <= tolerance * 4.)
        if expected and area <= tolerance ** 2:
            return dict(kind='expected_shared_feature', unexpected=False)
    kind = 'coplanar_area_overlap' if area > tolerance ** 2 else ('surface_crossing' if length > tolerance else 'unshared_contact')
    return dict(kind=kind, unexpected=True, shared_vertices=len(shared),
                intersection_length_mm=length * 1000., intersection_area_mm2=area * 1e6,
                witness=points.mean(0).tolist())

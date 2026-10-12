"""CPU broad/narrow phase for complete triangulated hand self crossings.

Variable-size bounding spheres generate candidates, then triangle AABBs and
SAT filter them. Expected shared features are checked geometrically, including
foldovers of adjacent faces. No finger, palm or forearm region is exempted.
"""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from .triangle_contact import triangle_overlap, intersection_relation


def candidate_pairs(vertices, triangles, tolerance=1e-7):
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0
            or triangles.dtype.kind not in 'iu' or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('Finite vertices and indexed triangles are required')
    geometry = vertices[triangles]
    center = geometry.mean(1)
    radius = np.linalg.norm(geometry - center[:, None], axis=2).max(1)
    lower, upper = geometry.min(1), geometry.max(1)
    buckets = np.floor(np.log2(max(radius.max(), 1e-30) / np.maximum(radius, 1e-30))).astype(int)
    groups = [np.flatnonzero(buckets == i) for i in np.unique(buckets)]
    trees = [cKDTree(center[ids]) for ids in groups]
    result = []
    for i, ids in enumerate(groups):
        for j in range(i, len(groups)):
            other = groups[j]
            reach = float(radius[ids].max() + radius[other].max() + tolerance)
            if i == j:
                pairs = ids[trees[i].query_pairs(reach, output_type='ndarray')]
            else:
                neighbors = trees[i].query_ball_tree(trees[j], reach)
                counts = np.array([len(n) for n in neighbors])
                if not counts.sum():
                    continue
                pairs = np.column_stack([np.repeat(ids, counts), other[np.concatenate([n for n in neighbors if n])]])
            if not len(pairs):
                continue
            a, b = pairs.T
            valid = (np.linalg.norm(center[a] - center[b], axis=1) <= radius[a] + radius[b] + tolerance)
            valid &= np.all(lower[a] <= upper[b] + tolerance, axis=1) & np.all(lower[b] <= upper[a] + tolerance, axis=1)
            if valid.any():
                result.append(np.sort(pairs[valid], axis=1))
    return np.concatenate(result) if result else np.empty((0, 2), dtype=int)


def self_intersections(vertices, triangles, tolerance=1e-7, witness_limit=20):
    return _self_intersections(vertices, triangles, tolerance, witness_limit)


def _self_intersections(vertices, triangles, tolerance, witness_limit, mixed_quad_count=None):
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles)
    geometry = vertices[triangles]
    edges = np.roll(geometry, -1, axis=1) - geometry
    normal = np.cross(edges[:, 0], edges[:, 1])
    length = np.linalg.norm(normal, axis=1)
    degenerate = np.flatnonzero(length <= tolerance * np.linalg.norm(edges, axis=2).max(1))
    valid_ids = np.flatnonzero(length > tolerance * np.linalg.norm(edges, axis=2).max(1))
    pairs = valid_ids[candidate_pairs(vertices, triangles[valid_ids], tolerance)] if len(valid_ids) else np.empty((0, 2), int)
    if mixed_quad_count is not None:
        # Alternative triangles of one face cannot coexist. Alternatives on
        # DIFFERENT faces can coexist, and Blender can rotate corner order
        # independently per face. Uniform global diagonal tests miss these.
        first, second = pairs.T
        cross_choice = (first < 2 * mixed_quad_count) != (second < 2 * mixed_quad_count)
        different_face = first % mixed_quad_count != second % mixed_quad_count
        pairs = pairs[cross_choice & different_face]
    normals = normal / np.maximum(length[:, None], 1e-30)
    unexpected, witnesses, kinds = [], [], Counter()
    expected = 0
    for start in range(0, len(pairs), 4096):
        subset = pairs[start:start+4096]
        hit = triangle_overlap(geometry[subset[:, 0]], geometry[subset[:, 1]], tolerance)
        for a, b in subset[hit]:
            shared = set(triangles[a]) & set(triangles[b])
            # Exact noncoplanar shared-edge triangles intersect only along
            # that edge. Shared-vertex faces get this fast exemption only
            # when their other vertices lie strictly on one side of a plane.
            sine = np.linalg.norm(np.cross(normals[a], normals[b]))
            if len(shared) == 2 and sine >= 1e-8:
                expected += 1
                continue
            if len(shared) == 1 and sine >= 1e-8:
                nonshared = [i for i, v in enumerate(triangles[a]) if v not in shared]
                distance = (geometry[a, nonshared] - geometry[b, 0]) @ normals[b]
                if np.all(distance > tolerance) or np.all(distance < -tolerance):
                    expected += 1
                    continue
                nonshared = [i for i, v in enumerate(triangles[b]) if v not in shared]
                distance = (geometry[b, nonshared] - geometry[a, 0]) @ normals[a]
                if np.all(distance > tolerance) or np.all(distance < -tolerance):
                    expected += 1
                    continue
            relation = intersection_relation(geometry[a], geometry[b], triangles[a], triangles[b],
                                             tolerance, sat_confirmed=True)
            if relation['unexpected']:
                unexpected.append([int(a), int(b)])
                kinds[relation['kind']] += 1
                if len(witnesses) < witness_limit:
                    witnesses.append(dict(first_triangle=int(a), second_triangle=int(b), **relation))
            elif relation['kind'] == 'expected_shared_feature':
                expected += 1
    return dict(surface_intersection_clear=not unexpected and not len(degenerate),
        unexpected_surface_intersections=len(unexpected), pairs=unexpected, kinds=dict(kinds),
        degenerate_triangles=len(degenerate), candidate_pairs=len(pairs), expected_shared_features=expected,
        witnesses=witnesses, complete_mesh_clearance_proven=False,
        exclusions=['volume containment', 'positive minimum distance', 'other geometry', 'continuous time'])


def quad_self_intersections(vertices, faces, tolerance=1e-7):
    """Audit all coexisting choices, including mixed choices on different faces.

    Alternative triangles belonging to the same quad cannot coexist and must
    not create fake self contacts. Both within-face diagonal choices are still
    checked independently for genuine bow-tie/folded quads.
    """
    results, alternatives = [], []
    for first, second in (((0, 1, 2), (0, 2, 3)), ((1, 2, 3), (1, 3, 0))):
        triangles = np.concatenate([faces[:, first], faces[:, second]])
        alternatives.append(triangles)
        results.append(self_intersections(vertices, triangles, tolerance))
    mixed = _self_intersections(vertices, np.concatenate(alternatives), tolerance, 20,
                               mixed_quad_count=len(faces))
    mixed['pair_namespace'] = 'Concatenated diagonal-0 then diagonal-1 triangles; different quad faces only'
    all_results = results + [mixed]
    return dict(surface_intersection_clear=all(r['surface_intersection_clear'] for r in all_results),
        unexpected_surface_intersections=max(r['unexpected_surface_intersections'] for r in all_results),
        possible_triangulation_crossing_pairs=sum(r['unexpected_surface_intersections'] for r in all_results),
        degenerate_triangles=max(r['degenerate_triangles'] for r in results),
        triangulations=results, cross_diagonal=mixed,
        diagonal_choice_complete=True, complete_mesh_clearance_proven=False)

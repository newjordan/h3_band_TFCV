"""Audit self/interhand crossings of Blender's actual level-2 skin triangles.

Run in Blender for its CPU BVH, without rendering:
  blender -b --factory-startup -P lab/skin_mesh_audit.py -- INPUT.json OUT.json

All same-hand faces, palm/webbing and forearms participate. Only intersection
confined to a genuinely shared topological feature is exempted. A clear surface
intersection result is not a volume-containment, positive-distance or CCD proof;
body/costume/drum meshes still need inclusion in the complete rig gate.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.triangle_contact import triangle_overlap, intersection_relation


def audit(path, tolerance=1e-7):
    started = time.monotonic()
    with np.load(path, allow_pickle=False) as data:
        vertices, triangles = data['vertices'], data['triangles']
        sides, regions = data['vertex_side'], data['vertex_region']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices)
            or sides.shape != (len(vertices),) or regions.shape != sides.shape):
        raise ValueError('Invalid actual surface geometry/label mapping')
    geometry = vertices[triangles]
    edges = np.roll(geometry, -1, axis=1) - geometry
    area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    degenerate = np.flatnonzero(area <= tolerance * np.linalg.norm(edges, axis=2).max(axis=1))
    valid = np.flatnonzero(area > tolerance * np.linalg.norm(edges, axis=2).max(axis=1))
    tree = BVHTree.FromPolygons(vertices.tolist(), triangles[valid].tolist(), all_triangles=True, epsilon=tolerance)
    pairs = np.asarray([(valid[i], valid[j]) for i, j in tree.overlap(tree) if i < j], dtype=np.int64).reshape(-1, 2)
    kinds, scopes, region_pairs = Counter(), Counter(), Counter()
    witnesses, unexpected_pairs, unexpected, expected, separated = [], [], 0, 0, 0
    same_hand, interhand, adjacent_crossings = 0, 0, 0
    for start in range(0, len(pairs), 4096):
        subset = pairs[start:start + 4096]
        overlap = triangle_overlap(geometry[subset[:, 0]], geometry[subset[:, 1]], tolerance)
        separated += int(np.count_nonzero(~overlap))
        for i, j in subset[overlap]:
            result = intersection_relation(geometry[i], geometry[j], triangles[i], triangles[j],
                                           tolerance, sat_confirmed=True)
            if not result['unexpected']:
                expected += int(result['kind'] == 'expected_shared_feature')
                separated += int(result['kind'] == 'separate')
                continue
            unexpected += 1
            unexpected_pairs.append([int(i), int(j)])
            kinds[result['kind']] += 1
            a_sides, b_sides = set(sides[triangles[i]]), set(sides[triangles[j]])
            if len(a_sides) != 1 or len(b_sides) != 1:
                raise ValueError('A rendered triangle spans separately driven hands')
            cross = a_sides != b_sides
            scope = 'interhand' if cross else next(iter(a_sides)) + '_self'
            scopes[scope] += 1
            interhand += int(cross)
            same_hand += int(not cross)
            adjacent_crossings += int(bool(set(triangles[i]) & set(triangles[j])))
            owners = sorted(['/'.join(sorted(set(regions[triangles[k]]))) for k in (i, j)])
            region_pairs[' ↔ '.join(owners)] += 1
            if len(witnesses) < 60:
                witnesses.append(dict(first_triangle=int(i), second_triangle=int(j),
                    first_vertex_ids=triangles[i].tolist(), second_vertex_ids=triangles[j].tolist(),
                    regions=owners, scope=scope, **result))
        print(json.dumps(dict(checked_pairs=min(start+4096, len(pairs)), candidate_pairs=len(pairs),
                              unexpected=unexpected, seconds=time.monotonic()-started)), flush=True)
    return dict(mesh=str(path.resolve()), mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        vertices=len(vertices), triangles=len(triangles), tolerance_mm=tolerance * 1000.,
        degenerate_triangles=len(degenerate), degenerate_triangle_ids=degenerate[:100].tolist(),
        candidate_pairs=len(pairs), separated_candidates=separated, expected_shared_feature_contacts=expected,
        unexpected_surface_intersections=unexpected, same_hand_intersections=same_hand, interhand_intersections=interhand,
        adjacent_face_foldovers=adjacent_crossings, kinds=dict(kinds), scopes=dict(scopes), region_pairs=dict(region_pairs),
        witnesses=witnesses, surface_intersection_clear=unexpected == 0 and not len(degenerate),
        unexpected_pairs=unexpected_pairs,
        complete_mesh_clearance_proven=False, continuous_time_proven=False, seconds=time.monotonic()-started)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    source, output = map(Path, argv[:2])
    manifest = json.loads(source.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='actual-skin-mesh-intersection-audit/v1', source_manifest=str(source.resolve()),
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source=manifest['source'],
        hand_scale=manifest['hand_scale'], scope=__doc__, records=[], performance_accepted=False,
        exclusions=['volume containment of open skin surfaces', 'positive minimum mesh distance',
                    'upper arm/body/costume mesh', 'force/contact mechanics', 'continuous time'],
        code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ('blockout/triangle_contact.py', 'lab/skin_mesh_audit.py')})
    for item in manifest['records']:
        path = Path(item['mesh'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['mesh_sha256']:
            raise ValueError('Actual mesh input changed after preparation')
        report['records'].append(dict(frame=item['frame'], **audit(path)))
        output.write_text(json.dumps(report, indent=2) + '\n')
    report['surface_intersection_clear'] = all(r['surface_intersection_clear'] for r in report['records'])
    report['status'] = 'complete'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], surface_intersection_clear=report['surface_intersection_clear'],
        records=[{k:r[k] for k in ('frame', 'degenerate_triangles', 'unexpected_surface_intersections',
                                 'same_hand_intersections', 'interhand_intersections', 'adjacent_face_foldovers')}
                 for r in report['records']]), indent=2))


if __name__ == '__main__':
    main()

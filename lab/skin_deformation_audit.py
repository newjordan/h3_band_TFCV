"""Independent actual-mesh audit of the frozen Blender deformation ablation."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.skin import Skin, ASSET
from blockout.subdivision import catmull_clark
from blockout.mesh_self import quad_self_intersections
from blockout.piano_geometry import scene_from_animation
from blockout.skin_contacts import key_support
from lab.skin_audit import hand_audit, quad_triangles


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_geometry(record, scale, linear):
    level = record['subdivision_levels']
    if record['context'] == 'bind_control':
        with np.load(ASSET, allow_pickle=False) as data:
            assert np.all(data['face_sizes'] == 4)
            operator, faces = catmull_clark(len(data['vertices']), data['face_vertices'].reshape(-1, 4), level, limit=level > 0)
            points = operator @ data['vertices']
            fields = operator @ data['weights']
            dominant = data['bone_names'][fields.argmax(1)]
            return points, faces, np.array([n.rsplit('.', 1)[1] for n in dominant]), dominant, {}
    points, faces, sides, regions, fields, offset = [], [], [], [], {}, 0
    for side in ('L', 'R'):
        skin = Skin(side, scale=scale)
        operator, quad = catmull_clark(len(skin.vertices), skin.faces, level, limit=level > 0)
        points.append(operator @ linear[side])
        faces.append(quad + offset)
        weights = np.asarray(operator @ skin.weights)
        fields[side] = (np.arange(offset, offset + len(weights)), weights, skin.bone_index)
        sides.extend([side] * len(weights))
        regions.extend([skin.names[n] + '.' + side for n in weights.argmax(1)])
        offset += len(weights)
    return np.concatenate(points), np.concatenate(faces), np.array(sides), np.array(regions), fields


def face_set(faces):
    return {tuple(row) for row in np.sort(faces, axis=1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = json.loads(args.snapshots.read_text())
    assert source['status'] == 'complete' and source['source_unchanged_during_run']
    assert not source['candidate_exported']
    code_files = ('lab/skin_deformation_audit.py', 'blockout/mesh_self.py', 'blockout/triangle_contact.py',
        'blockout/skin.py', 'blockout/subdivision.py', 'blockout/piano_geometry.py', 'blockout/key_shapes.py',
        'blockout/skin_contacts.py', 'lab/skin_audit.py', 'blockout/hand_model/skin_bind.npz')
    hashes = {p: sha(ROOT / p) for p in code_files}
    report = dict(schema='frozen-skin-deformation-audit/v1', snapshots=source,
        snapshots_sha256=sha(args.snapshots), code_sha256=hashes, records=[],
        full_performance_accepted=False, candidate_exported=False,
        scope='All actual same/interhand quad surfaces, both diagonals, at one frozen instant; '
              'instrument/assigned distal support on production poses. Bind control is an asset-only test.',
        exclusions=['positive distance', 'volume containment', 'body/costume', 'full music/trajectory',
                    'forces/action', 'continuous time', 'naturalness'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    mappings = {}
    for record in source['records']:
        path = Path(record['mesh'])
        assert sha(path) == record['mesh_sha256']
        with np.load(path, allow_pickle=False) as data:
            vertices, faces = data['vertices'], data['faces']
            key = record['context'], record['subdivision_levels']
            if record['mode'] == 'linear':
                expected, expected_faces, sides, regions, fields = expected_geometry(record, source['hand_scale'], data)
                error, mapping = cKDTree(expected).query(vertices)
                assert len(expected) == len(vertices) and np.array_equal(np.sort(mapping), np.arange(len(expected)))
                assert error.max() * 1000. < .01
                assert face_set(mapping[faces]) == face_set(expected_faces)
                mappings[key] = (mapping, sides[mapping], regions[mapping], fields, faces.copy(), float(error.max() * 1000.))
            mapping, sides, regions, fields, linear_faces, mapping_error = mappings[key]
            assert np.array_equal(faces, linear_faces)
        audit = quad_self_intersections(vertices, faces)
        counts, region_counts = [], []
        choices = [np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]]),
                   np.concatenate([faces[:, [1, 2, 3]], faces[:, [1, 3, 0]]])]
        for triangulation, triangles in zip(audit['triangulations'] + [audit['cross_diagonal']],
                                            choices + [np.concatenate(choices)]):
            counter, regional = Counter(), Counter()
            for first, second in triangulation['pairs']:
                a, b = triangles[first], triangles[second]
                first_side, second_side = set(sides[a]), set(sides[b])
                assert len(first_side) == 1 and len(second_side) == 1
                counter['interhand' if first_side != second_side else next(iter(first_side)) + '_self'] += 1
                regional[' ↔ '.join(sorted(['/'.join(sorted(set(regions[t]))) for t in (a, b)]))] += 1
            counts.append(dict(counter))
            region_counts.append(dict(regional))
        hands, supports = {}, {}
        if record['source'] is not None:
            animation = json.loads(Path(record['source']).read_text())
            assert sha(Path(record['source'])) == record['source_sha256']
            frame = animation['frames'][record['frame']]
            if 'key_commands' in record:
                frame = copy.deepcopy(frame)
                frame['keys'] = copy.deepcopy(record['key_commands'])
            scene = scene_from_animation(animation, frame, music_desk=True)
            for side, hand in frame['hands'].items():
                ids = np.flatnonzero(sides == side)
                selected = np.flatnonzero(np.all(sides[faces] == side, axis=1))
                # Keep full global vertex indices in triangles and witnesses.
                local_map = np.full(len(vertices), -1, int)
                local_map[ids] = np.arange(len(ids))
                hands[side] = hand_audit(vertices[ids], quad_triangles(local_map[faces[selected]]), scene)
                field_ids, weights, names = fields[side]
                supports[side] = []
                for finger, pitch in enumerate(hand['assigned_keys'], 1):
                    if pitch is None:
                        continue
                    cpu_distal = field_ids[weights[:, names[f'finger{finger}-3']] > .2]
                    actual_distal = np.flatnonzero(np.isin(mapping, cpu_distal))
                    supports[side].append(dict(finger=finger, pitch=pitch,
                        activation=hand['activation'][finger-1],
                        **key_support(vertices, actual_distal, pitch, scene)))
        result = dict(record)
        result.update(mapping_from_identical_linear_topology=True, lbs_parity_maximum_error_mm=mapping_error,
            scope_count_buckets=['uniform_diagonal_0', 'uniform_diagonal_1', 'mixed_choices_different_faces'],
            diagonal_choice_complete=True,
            surface_intersection_clear=audit['surface_intersection_clear'],
            self_intersections=audit['unexpected_surface_intersections'], degenerate_triangles=audit['degenerate_triangles'],
            scope_counts=counts, region_counts=region_counts, mesh_audit=audit,
            instrument=hands, assigned_support=supports,
            candidate_exported=False, full_performance_accepted=False, seconds=time.monotonic()-started)
        report['records'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k:result[k] for k in ('context', 'mode', 'subdivision_levels', 'self_intersections',
            'degenerate_triangles', 'scope_counts', 'seconds')} | dict(instrument={s:dict(depth_mm=h['maximum_penetration_mm'],
                triangles=h['triangle_intersections']) for s,h in hands.items()})), flush=True)
    report['source_unchanged_during_run'] = all(sha(ROOT / p) == value for p,value in hashes.items())
    assert report['source_unchanged_during_run']
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Complete actual-mesh ablation; full performance remains unaccepted')


if __name__ == '__main__':
    main()

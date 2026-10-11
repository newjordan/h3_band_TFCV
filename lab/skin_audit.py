"""Audit renderer-matched hand surfaces against all piano solids.

Static sampled frames only. Both diagonals of each nonplanar rendered quad are
audited conservatively, so Blender's tessellation choice cannot hide a contact.
This checks hand/forearm skin versus instrument; self/body/interhand collision
and continuous time need additional gates before a performance can pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.skin import Skin
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import scene_from_animation
from blockout.rig.pose_io import animation_hand_scale


def quad_triangles(faces):
    return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]],
                           faces[:, [1, 2, 3]], faces[:, [1, 3, 0]]])


def hand_audit(vertices, triangles, scene, tolerance=1e-5):
    boxes = scene.candidates(vertices, padding=.003)
    distance, nearest = scene.closest(vertices, boxes)
    penetrating = np.flatnonzero(distance < -tolerance)
    contacts = scene.triangle_contacts(vertices, triangles, tolerance)
    worst = int(distance.argmin())
    by_box = {}
    for box in np.unique(nearest[penetrating]):
        ids = penetrating[nearest[penetrating] == box]
        by_box[scene.specs[box].name] = dict(vertices=len(ids),
            maximum_penetration_mm=float(-distance[ids].min() * 1000.))
    return dict(penetrating_vertices=len(penetrating),
                maximum_penetration_mm=float(max(0., -distance[worst]) * 1000.),
                minimum_clearance_mm=float(distance[worst] * 1000.),
                worst_vertex=worst, worst_position=vertices[worst].tolist(),
                worst_solid=None if nearest[worst] < 0 else scene.specs[nearest[worst]].name,
                vertex_contacts=by_box,
                triangle_contacts={scene.specs[box].name: len(ids) for box, ids in contacts},
                triangle_intersections=sum(len(ids) for _, ids in contacts),
                instrument_clear=not len(penetrating) and not contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('animation', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--frames', default='all')
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--levels', type=int, default=2)
    args = parser.parse_args()
    anim = json.loads(args.animation.read_text())
    args.scale = animation_hand_scale(anim, requested=args.scale)
    ids = list(range(len(anim['frames']))) if args.frames == 'all' else [int(i) for i in args.frames.split(',')]
    skins = {side: Skin(side, scale=args.scale) for side in ('L', 'R')}
    surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, args.levels, limit=True)
                for side, skin in skins.items()}
    triangles = {side: quad_triangles(surface[1]) for side, surface in surfaces.items()}
    records, start = [], time.monotonic()
    for index in ids:
        frame = anim['frames'][index]
        scene = scene_from_animation(anim, frame, music_desk=True)
        hands = {}
        for side, skin in skins.items():
            vertices = surfaces[side][0] @ skin.from_frame(frame['hands'][side])
            hands[side] = hand_audit(vertices, triangles[side], scene)
        records.append(dict(frame=index, time=frame.get('t'), hands=hands))
        print(json.dumps(dict(frame=index, hands={side: dict(depth_mm=round(r['maximum_penetration_mm'], 3),
            triangles=r['triangle_intersections'], worst=r['worst_solid']) for side, r in hands.items()})), flush=True)
    report = dict(source=str(args.animation.resolve()), source_sha256=hashlib.sha256(args.animation.read_bytes()).hexdigest(),
                  hand_scale=args.scale, subdivision_levels=args.levels, limit_surface=True, tolerance_mm=.01,
                  scope='static rendered hand/forearm skin versus keys, casework, desk and floor; both quad diagonals',
                  exclusions=['self collision', 'interhand collision', 'upper arm/body mesh', 'continuous time'],
                  full_timeline=ids == list(range(len(anim['frames']))), seconds=time.monotonic() - start,
                  instrument_clear=all(r['instrument_clear'] for fr in records for r in fr['hands'].values()),
                  affected_frames=sum(any(not r['instrument_clear'] for r in fr['hands'].values()) for fr in records),
                  maximum_penetration_mm=max(r['maximum_penetration_mm'] for fr in records for r in fr['hands'].values()),
                  records=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()

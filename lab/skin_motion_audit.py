"""Oversample anatomical hand motion, with explicit unproved CCD scope.

This detects collisions/contact loss between exported frames. A clean result is
sampled evidence only; it cannot establish continuous-time collision freedom.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('animation', type=Path)
parser.add_argument('report', type=Path)
parser.add_argument('--frames', required=True)
parser.add_argument('--substeps', type=int, default=10)
parser.add_argument('--scale', type=float, required=True)
args = parser.parse_args()
if args.substeps < 1:
    raise ValueError('At least one time substep is required')
os.environ['HAND_SCALE'] = str(args.scale)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from blockout.rig.hand import HandRig
from blockout.rig.skin_contact import pose_vector
from blockout.rig.motion import interpolate_frame, rotation_angle, bend_plane_angle
from blockout.skin import Skin
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import PianoScene
from lab.skin_audit import hand_audit, quad_triangles

anim = json.loads(args.animation.read_text())
start, stop = map(int, args.frames.split(':'))
rigs = {side: HandRig(side) for side in anim['frames'][start]['hands']}
skins = {side: Skin(side, scale=args.scale) for side in rigs}
surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, 2, limit=True) for side, skin in skins.items()}
triangles = {side: quad_triangles(faces) for side, (_, faces) in surfaces.items()}
records, motion = [], []
for index in range(start, stop - 1):
    first, second = anim['frames'][index:index + 2]
    dt = second['t'] - first['t']
    for side, rig in rigs.items():
        a, b = first['hands'][side], second['hands'][side]
        motion.append(dict(frame=index, side=side,
            joint_speed_rad_s=float(np.max(np.abs(pose_vector(rig, b)[6:] - pose_vector(rig, a)[6:])) / dt),
            wrist_speed_m_s=float(np.linalg.norm(np.asarray(b['root']['pos']) - a['root']['pos']) / dt),
            wrist_angular_speed_rad_s=rotation_angle(a['root']['rot'], b['root']['rot']) / dt,
            elbow_plane_speed_rad_s=bend_plane_angle(second['body']['shoulders'][side], b['wrist'], b['elbow'], a['elbow']) / dt))
    steps = range(args.substeps + int(index == stop - 2))
    for step in steps:
        fraction = step / args.substeps
        frame = interpolate_frame(first, second, fraction, rigs)
        scene = PianoScene(anim['keyboard'], frame['keys'], music_desk=True)
        hands = {}
        for side, skin in skins.items():
            vertices = surfaces[side][0] @ skin.from_frame(frame['hands'][side])
            result = hand_audit(vertices, triangles[side], scene)
            x = pose_vector(rigs[side], frame['hands'][side])
            result['joint_bounds_excess_rad'] = float(max(0., (rigs[side].lo - x[6:]).max(),
                                                         (x[6:] - rigs[side].hi).max()))
            hands[side] = result
        records.append(dict(frame=index, fraction=fraction, time=frame['t'], hands=hands))
        print(json.dumps(dict(frame=index, fraction=fraction,
            maximum_penetration_mm=max(r['maximum_penetration_mm'] for r in hands.values()))), flush=True)
report = dict(source=str(args.animation.resolve()), window=[start, stop], sample_fps=anim['fps'] * args.substeps,
              interpolation='linear anatomical joints/roots/key depths; SO(3) wrist; exact two-bone elbow circle',
              continuous_time_proven=False, exclusions=['exact self/interhand/body/costume', 'CCD proof', 'score note completeness'],
              sampled_instrument_clear=all(r['instrument_clear'] for f in records for r in f['hands'].values()),
              samples=len(records), affected_samples=sum(any(not r['instrument_clear'] for r in f['hands'].values()) for f in records),
              maximum_penetration_mm=max(r['maximum_penetration_mm'] for f in records for r in f['hands'].values()),
              motion=motion, records=records)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('records', 'motion')}, indent=2))

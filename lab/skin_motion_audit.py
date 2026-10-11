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
parser.add_argument('--score-midi', type=Path, help='optional MIDI for independently measuring sustained score contacts')
parser.add_argument('--score-feeling', type=Path, help='matching performance-time chart; requires --score-midi')
args = parser.parse_args()
if args.substeps < 1:
    raise ValueError('At least one time substep is required')
if args.score_feeling and not args.score_midi:
    raise ValueError('A performance chart requires its score MIDI')
os.environ['HAND_SCALE'] = str(args.scale)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from blockout.rig.hand import HandRig
from blockout.rig.skin_contact import pose_vector
from blockout.rig.motion import interpolate_frame, rotation_angle, bend_plane_angle
from blockout.skin import Skin
from blockout.skin_contacts import key_support
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import scene_from_animation
from lab.skin_audit import hand_audit, quad_triangles

anim = json.loads(args.animation.read_text())
from blockout.rig.pose_io import animation_hand_scale
args.scale = animation_hand_scale(anim, requested=args.scale)
score_notes = None
if args.score_midi:
    from blockout import midi, feeling
    score_notes = midi.read(args.score_midi)
    if args.score_feeling:
        _, downbeats = midi.beats(args.score_midi)
        score_notes, _ = feeling.perform(score_notes, feeling.Chart(args.score_feeling, downbeats), 0)
start, stop = map(int, args.frames.split(':'))
rigs = {side: HandRig(side) for side in anim['frames'][start]['hands']}
skins = {side: Skin(side, scale=args.scale) for side in rigs}
surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, 2, limit=True) for side, skin in skins.items()}
triangles = {side: quad_triangles(faces) for side, (_, faces) in surfaces.items()}
distal = {side: {finger: np.flatnonzero(np.asarray(surfaces[side][0] @ skin.weights[:, skin.bone_index[f'finger{finger}-3']]) > .2)
                 for finger in range(1, 6)} for side, skin in skins.items()}
records, motion = [], []
previous_sample = None
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
        scene = scene_from_animation(anim, frame, music_desk=True)
        hands, support = {}, {}
        absolute = anim.get('start', 0.) + frame['t']
        required = ({int(n.pitch) for n in score_notes if n.start + .080 <= absolute <= n.end}
                    if score_notes is not None else set())
        sampled_motion = {}
        for side, skin in skins.items():
            vertices = surfaces[side][0] @ skin.from_frame(frame['hands'][side])
            result = hand_audit(vertices, triangles[side], scene)
            x = pose_vector(rigs[side], frame['hands'][side])
            result['joint_bounds_excess_rad'] = float(max(0., (rigs[side].lo - x[6:]).max(),
                                                         (x[6:] - rigs[side].hi).max()))
            from blockout.performer import UPPER_ARM, FOREARM
            hand = frame['hands'][side]
            root, elbow, shoulder = map(np.asarray, (hand['wrist'], hand['elbow'], frame['body']['shoulders'][side]))
            result['arm_length_error_mm'] = float(max(abs(np.linalg.norm(root - elbow) - FOREARM),
                                                      abs(np.linalg.norm(elbow - shoulder) - UPPER_ARM)) * 1000.)
            result['proxy_self_clearance_mm'] = float(rigs[side].finger_clearances(x, np.asarray(hand['root']['rot'])).min() * 1000.)
            for finger, pitch in enumerate(hand['assigned_keys'], 1):
                if pitch is None or int(pitch) not in required:
                    continue
                contact = key_support(vertices, distal[side][finger], pitch, scene)
                contact.update(side=side, finger=finger, key_depth=float(frame['keys'].get(str(pitch), 0.)))
                support.setdefault(int(pitch), []).append(contact)
            if previous_sample is not None:
                elapsed = frame['t'] - previous_sample['t']
                old = previous_sample['hands'][side]
                sampled_motion[side] = dict(
                    joint_speed_rad_s=float(np.max(np.abs(x[6:] - pose_vector(rigs[side], old)[6:])) / elapsed),
                    wrist_speed_m_s=float(np.linalg.norm(root - old['wrist']) / elapsed),
                    wrist_angular_speed_rad_s=rotation_angle(old['root']['rot'], hand['root']['rot']) / elapsed,
                    elbow_plane_speed_rad_s=bend_plane_angle(shoulder, root, elbow, old['elbow']) / elapsed)
            hands[side] = result
        score = dict(required_pitches=sorted(required), contacts=support,
                     unsupported_pitches=sorted(p for p in required if not any(c['supported'] and c['key_depth'] >= .5 for c in support.get(p, []))))
        records.append(dict(frame=index, fraction=fraction, time=frame['t'], hands=hands,
                            sampled_motion=sampled_motion, held_score=score if score_notes is not None else None))
        previous_sample = frame
        print(json.dumps(dict(frame=index, fraction=fraction,
            maximum_penetration_mm=max(r['maximum_penetration_mm'] for r in hands.values()))), flush=True)
report = dict(source=str(args.animation.resolve()), window=[start, stop], sample_fps=anim['fps'] * args.substeps,
              interpolation='linear anatomical joints/roots/key depths; SO(3) wrist; exact two-bone elbow circle',
              continuous_time_proven=False, exclusions=['exact self/interhand/body/costume', 'CCD proof', 'score note completeness'],
              sampled_instrument_clear=all(r['instrument_clear'] for f in records for r in f['hands'].values()),
              samples=len(records), affected_samples=sum(any(not r['instrument_clear'] for r in f['hands'].values()) for f in records),
              maximum_penetration_mm=max(r['maximum_penetration_mm'] for f in records for r in f['hands'].values()),
              motion=motion, records=records)
report['held_score_checked'] = score_notes is not None
report['held_score_scope'] = 'skin support and key depth >= 0.5 after 80 ms settling; onset/accidental-note completeness and forces excluded'
report['held_score_unsupported_observations'] = (sum(len(f['held_score']['unsupported_pitches']) for f in records)
                                               if score_notes is not None else None)
report['held_score_required_observations'] = (sum(len(f['held_score']['required_pitches']) for f in records)
                                            if score_notes is not None else None)
report['sampled_motion_maxima'] = {name: max((m[name] for f in records for m in f['sampled_motion'].values()), default=0.)
                                  for name in ('joint_speed_rad_s', 'wrist_speed_m_s', 'wrist_angular_speed_rad_s', 'elbow_plane_speed_rad_s')}
report['sampled_proxy_self_intersections'] = sum(r['proxy_self_clearance_mm'] < -1e-3 for f in records for r in f['hands'].values())
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('records', 'motion')}, indent=2))

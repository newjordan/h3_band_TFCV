"""Couple dense anatomical poses through bounded neighboring contact solves.

This is an empirical trajectory experiment. Endpoints/control poses are frozen
by default; forward/reverse sweeps share corrected neighbors. No partly failed
path is exported as a candidate. A node pass still needs independent motion,
full mesh/mesh/body checks and CCD before it can become a performance.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('animation', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--frames', default='23:26')
parser.add_argument('--scale', type=float, required=True)
parser.add_argument('--substeps', type=int, default=10)
parser.add_argument('--sweeps', type=int, default=2)
parser.add_argument('--sides', default='LR')
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--passes', type=int, default=4)
parser.add_argument('--feasibility-iterations', type=int, default=80)
parser.add_argument('--move-knots', action='store_true', help='also replan interior source poses; endpoints remain fixed')
parser.add_argument('--resume', type=Path, help='unvalidated numeric checkpoint from exactly the same experiment')
parser.add_argument('--score-midi', type=Path, required=True)
parser.add_argument('--score-feeling', type=Path)
args = parser.parse_args()
if args.substeps < 1 or args.sweeps < 1:
    raise ValueError('Positive sampling/sweep budgets are required')
os.environ['HAND_SCALE'] = str(args.scale)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from blockout import midi, feeling
from blockout.rig.hand import HandRig
from blockout.rig.motion import interpolate_frame, rotation_angle, bend_plane_angle
from blockout.rig.skin_contact import restore_pose, pose_vector, musical_anchors
from blockout.skin import Skin
from blockout.skin_contacts import key_support
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import scene_from_animation
from blockout.performer import UPPER_ARM, FOREARM
from blockout.rig.path_validation import sampled_path_violations
from blockout.rig.path_checkpoint import save_checkpoint, load_checkpoint
from lab.skin_audit import hand_audit, quad_triangles
from lab.piano_interference_audit import interference

source = json.loads(args.animation.read_text())
from blockout.rig.pose_io import animation_hand_scale
args.scale = animation_hand_scale(source, requested=args.scale)
start, stop = map(int, args.frames.split(':'))
if not 0 <= start < stop - 1 < len(source['frames']):
    raise ValueError('The path needs at least two source frames')
rigs = {side: HandRig(side) for side in source['frames'][start]['hands']}
skins = {side: Skin(side, scale=args.scale) for side in rigs}
surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, 2, limit=True) for side, skin in skins.items()}
triangles = {side: quad_triangles(faces) for side, (_, faces) in surfaces.items()}
distal = {side: {finger: np.flatnonzero(np.asarray(surfaces[side][0] @ skin.weights[:, skin.bone_index[f'finger{finger}-3']]) > .2)
                 for finger in range(1, 6)} for side, skin in skins.items()}
keys = {key['pitch']: key for key in source['keyboard']}
notes = midi.read(args.score_midi)
if args.score_feeling:
    _, downbeats = midi.beats(args.score_midi)
    notes, _ = feeling.perform(notes, feeling.Chart(args.score_feeling, downbeats), 0)

frames = []
for index in range(start, stop - 1):
    for step in range(args.substeps):
        frames.append(copy.deepcopy(source['frames'][index]) if step == 0 else
                      interpolate_frame(source['frames'][index], source['frames'][index + 1],
                                        step / args.substeps, rigs))
frames.append(copy.deepcopy(source['frames'][stop - 1]))
offset = frames[0]['t']
required, scenes, anchors = [], [], {}
for i, frame in enumerate(frames):
    absolute = source.get('start', 0.) + frame['t']
    required.append({int(n.pitch) for n in notes if n.start + .080 <= absolute <= n.end})
    assigned = {int(p) for h in frame['hands'].values() for p in h['assigned_keys'] if p is not None}
    for pitch in required[-1] & assigned:
        frame['keys'][str(pitch)] = 1.
    scenes.append(scene_from_animation(source, frame, music_desk=True))
    for side, skin in skins.items():
        operator = surfaces[side][0]
        anchors[i, side] = musical_anchors(skin, operator, operator @ skin.from_frame(frame['hands'][side]),
            frame['hands'][side], scenes[-1], keys, required_pitches=required[-1])
report = dict(source=str(args.animation.resolve()), source_sha256=hashlib.sha256(args.animation.read_bytes()).hexdigest(),
    code_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
                 ('blockout/rig/skin_contact.py', 'blockout/rig/motion.py', 'blockout/rig/path_validation.py',
                  'blockout/rig/path_checkpoint.py', 'blockout/skin.py', 'blockout/subdivision.py',
                  'blockout/skin_contacts.py', 'blockout/piano_geometry.py', 'blockout/key_shapes.py',
                  'blockout/hand_model/skin_bind.npz', 'lab/skin_audit.py',
                  'blockout/rig/hand.py', 'blockout/rig/skeleton.py', 'blockout/rig/comfort.json',
                  'blockout/rig/furelise_cal.json', 'blockout/performer.py',
                  'lab/piano_interference_audit.py', 'lab/restore_skin_path.py')},
    frame_window=[start, stop], hand_scale=args.scale, substeps=args.substeps, sweeps=args.sweeps,
    control_poses_frozen=not args.move_knots, endpoints_frozen=True,
    geometry=source.get('geometry', {'key_shape': 'rectangular'}),
    score_sha256=hashlib.sha256(args.score_midi.read_bytes()).hexdigest(),
    feeling_sha256=hashlib.sha256(args.score_feeling.read_bytes()).hexdigest() if args.score_feeling else None,
    solve_sides=args.sides,
    audit_configuration=dict(top_gap_mm=.1, geometry_tolerance_mm=.01, required_key_depth=.5,
                             score_settling_seconds=.080, arm_length_tolerance_mm=.001,
                             proxy_spacing_tolerance_mm=.001),
    solver_budget=dict(iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations),
    continuous_time_proven=False, candidate_exported=False,
    scope='rendered hand/instrument, held score support, anatomical and engineering speed bounds; sampled nodes only',
    exclusions=['exact self/interhand/body/costume', 'onset/accidental-note completeness', 'force dynamics', 'human form acceptance', 'CCD'],
    records=[])
args.output.parent.mkdir(parents=True, exist_ok=True)
identity = {key: report[key] for key in ('source_sha256', 'code_sha256', 'frame_window', 'hand_scale',
            'substeps', 'control_poses_frozen', 'geometry', 'score_sha256', 'feeling_sha256',
            'solve_sides', 'audit_configuration')}
resume_metadata = None
if args.resume:
    frames, resume_metadata = load_checkpoint(args.resume, frames, rigs, identity)
    report['resumed_from'] = str(args.resume.resolve())
    report['previous_sweeps_completed'] = resume_metadata['sweeps_completed']

def persist():
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')

def audit():
    observations = []
    for index, frame in enumerate(frames):
        support, hands = {}, {}
        for side, skin in skins.items():
            hand = frame['hands'][side]
            vertices = surfaces[side][0] @ skin.from_frame(hand)
            result = hand_audit(vertices, triangles[side], scenes[index])
            vector = pose_vector(rigs[side], hand)
            result['proxy_self_clearance_mm'] = float(rigs[side].finger_clearances(vector, np.asarray(hand['root']['rot'])).min() * 1000.)
            result['joint_bounds_excess_rad'] = float(max(0., (rigs[side].lo - vector[6:]).max(), (vector[6:] - rigs[side].hi).max()))
            wrist, elbow, shoulder = map(np.asarray, (hand['wrist'], hand['elbow'], frame['body']['shoulders'][side]))
            result['arm_length_error_mm'] = float(max(abs(np.linalg.norm(wrist - elbow) - FOREARM),
                                                      abs(np.linalg.norm(elbow - shoulder) - UPPER_ARM)) * 1000.)
            result['wrist_root_error_mm'] = float(np.linalg.norm(wrist - hand['root']['pos']) * 1000.)
            for finger, pitch in enumerate(hand['assigned_keys'], 1):
                if pitch is not None and int(pitch) in required[index]:
                    contact = key_support(vertices, distal[side][finger], pitch, scenes[index])
                    contact['key_depth'] = frame['keys'].get(str(pitch), 0.)
                    support.setdefault(int(pitch), []).append(contact)
            if index:
                previous = frames[index - 1]['hands'][side]
                dt = frame['t'] - frames[index - 1]['t']
                result['motion'] = dict(
                    joint_speed_rad_s=float(np.max(np.abs(vector[6:] - pose_vector(rigs[side], previous)[6:])) / dt),
                    wrist_speed_m_s=float(np.linalg.norm(np.asarray(hand['wrist']) - previous['wrist']) / dt),
                    wrist_angular_speed_rad_s=rotation_angle(hand['root']['rot'], previous['root']['rot']) / dt,
                    elbow_plane_speed_rad_s=bend_plane_angle(frame['body']['shoulders'][side], hand['wrist'], hand['elbow'], previous['elbow']) / dt)
            hands[side] = result
        unsupported = sorted(p for p in required[index] if not any(c['supported'] and c['key_depth'] >= .5 for c in support.get(p, [])))
        depths = np.asarray(list(frame['keys'].values()), float)
        travel = float(max(0., -depths.min(), depths.max() - 1.)) if np.isfinite(depths).all() else float('inf')
        observations.append(dict(node=index, time=frame['t'], hands=hands, required_pitches=sorted(required[index]),
                                 unsupported_pitches=unsupported, key_travel_excess=travel,
                                 mechanical_contacts=interference(scenes[index], include_case=True)))
    return observations

report['initial_observations'] = audit()
persist()
mutable = [i for i in range(1, len(frames) - 1) if args.move_knots or i % args.substeps]
for sweep in range(args.sweeps):
    order = mutable if sweep % 2 == 0 else mutable[::-1]
    for index in order:
        for side in args.sides:
            frame, rig, skin = frames[index], rigs[side], skins[side]
            previous, following = (frames[n]['hands'][side] for n in (index - 1, index + 1))
            dt = min(frame['t'] - frames[index - 1]['t'], frames[index + 1]['t'] - frame['t'])
            begun = time.monotonic()
            result, corrected = restore_pose(rig, skin, surfaces[side], frame['hands'][side], scenes[index],
                frame['body']['shoulders'][side], pose_vector(rig, previous), pose_vector(rig, following),
                dt, anchors=anchors[index, side], previous_hand=previous, following_hand=following,
                iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations)
            result.update(sweep=sweep, node=index, side=side, seconds=time.monotonic() - begun)
            if result['accepted']:
                frame['hands'][side] = corrected
                frame['body']['elbows'][side] = corrected['elbow']
            report['records'].append(result)
            persist()
            save_checkpoint(args.output.with_suffix('.checkpoint.npz'), frames, rigs, identity, sweep)
            print(json.dumps(dict(sweep=sweep, node=index, side=side, accepted=result['accepted'],
                                 penetration_mm=result.get('final_penetration_mm'), seconds=result['seconds'])), flush=True)
    report['final_observations'] = audit()
    persist()
    save_checkpoint(args.output.with_suffix('.checkpoint.npz'), frames, rigs, identity, sweep + 1)
    if not sampled_path_violations(report['final_observations']):
        break

observations = report['final_observations']
report['violations'] = sampled_path_violations(observations)
report['accepted'] = not report['violations']
report['affected_nodes'] = sum(any(not h['instrument_clear'] for h in f['hands'].values()) for f in observations)
report['maximum_penetration_mm'] = max(h['maximum_penetration_mm'] for f in observations for h in f['hands'].values())
report['unsupported_held_observations'] = sum(len(f['unsupported_pitches']) for f in observations)
report['mechanical_interference_nodes'] = sum(bool(f['mechanical_contacts']) for f in observations)
report['candidate_exported'] = bool(report['accepted'])
if report['accepted']:
    candidate = copy.deepcopy(source)
    candidate['start'] = source.get('start', 0.) + offset
    candidate['fps'] = source['fps'] * args.substeps
    for frame in frames:
        frame['t'] -= offset
    candidate['frames'] = frames
    candidate['duration'] = frames[-1]['t']
    candidate['diagnostic'] = 'Coupled sampled nodes passed; independent motion/full body/CCD/naturalness remain unverified.'
    args.output.with_suffix('.anim.json').write_text(json.dumps(candidate))
persist()
print(json.dumps({key: value for key, value in report.items() if key not in ('initial_observations', 'final_observations', 'records')}, indent=2))

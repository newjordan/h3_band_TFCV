"""Restore adjacent poses with fixed musical anchors and physical neighbors.

No failed candidate replaces an existing frame. Exported results are diagnostic
until independent full geometry, musical fidelity and time-motion checks pass.
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
parser.add_argument('--sides', default='LR')
parser.add_argument('--scale', type=float, required=True)
parser.add_argument('--sweeps', type=int, default=2)
parser.add_argument('--passes', type=int, default=5)
parser.add_argument('--feasibility-iterations', type=int, default=0, help='bounded trust-region initialization budget; 0 keeps the baseline')
parser.add_argument('--score-holds', action='store_true', help='require actual skin support for assigned sustained score notes')
args = parser.parse_args()
os.environ['HAND_SCALE'] = str(args.scale)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.rig.hand import HandRig
from blockout.rig.skin_contact import restore_pose, pose_vector, musical_anchors
from blockout.skin import Skin
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import scene_from_animation

source = json.loads(args.animation.read_text())
from blockout.rig.pose_io import animation_hand_scale
args.scale = animation_hand_scale(source, requested=args.scale)
anim = copy.deepcopy(source)
start, stop = map(int, args.frames.split(':'))
if not 0 < start < stop < len(source['frames']):
    raise ValueError('Window requires an unchanged neighbor at each end')
rigs = {side: HandRig(side) for side in args.sides}
skins = {side: Skin(side, scale=args.scale) for side in args.sides}
surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, 2, limit=True) for side, skin in skins.items()}
keys = {key['pitch']: key for key in source['keyboard']}
required_by_frame = {}
if args.score_holds:
    from blockout import midi, feeling
    notes = midi.read(ROOT / 'examples/piano/knights_suite.mid')
    _, downbeats = midi.beats(ROOT / 'examples/piano/knights_suite.mid')
    notes, _ = feeling.perform(notes, feeling.Chart(ROOT / 'examples/piano/knights_suite.feeling.json', downbeats), 0)
    for index in range(start, stop):
        frame = source['frames'][index]
        absolute = source['start'] + frame['t']
        # Existing QA's 80 ms settling interval: a sustained score note must
        # remain down after the attack. Geometry/skin anchors must independently
        # prove support; merely commanding the key down is not a passing result.
        required = {int(note.pitch) for note in notes if note.start + .080 <= absolute <= note.end}
        assigned = {int(pitch) for hand in frame['hands'].values() for pitch in hand['assigned_keys'] if pitch is not None}
        required_by_frame[index] = required
        for pitch in required & assigned:
            frame['keys'][str(pitch)] = 1.
            anim['frames'][index]['keys'][str(pitch)] = 1.
anchors = {}
for index in range(start, stop):
    frame = source['frames'][index]
    scene = scene_from_animation(source, frame, music_desk=True)
    for side, skin in skins.items():
        anchors[index, side] = musical_anchors(skin, surfaces[side][0],
            surfaces[side][0] @ skin.from_frame(frame['hands'][side]), frame['hands'][side], scene, keys,
            required_pitches=required_by_frame.get(index, ()))
report = dict(source=str(args.animation.resolve()), hand_scale=args.scale, window=[start, stop],
              sweeps=args.sweeps, trajectory_validated=False, score_holds=args.score_holds, records=[])
report['feasibility_iterations'] = args.feasibility_iterations
report['source_sha256'] = hashlib.sha256(args.animation.read_bytes()).hexdigest()
report['code_sha256'] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                        for path in ('blockout/rig/skin_contact.py', 'blockout/rig/motion.py',
                                     'blockout/piano_geometry.py', 'blockout/skin.py',
                                     'blockout/subdivision.py', 'lab/restore_skin_window.py')}
if args.score_holds:
    report['score_sha256'] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                            for path in ('examples/piano/knights_suite.mid', 'examples/piano/knights_suite.feeling.json')}
args.output.parent.mkdir(parents=True, exist_ok=True)
for sweep in range(args.sweeps):
    indices = range(start, stop) if sweep % 2 == 0 else range(stop - 1, start - 1, -1)
    for index in indices:
        frame = source['frames'][index]
        scene = scene_from_animation(source, frame, music_desk=True)
        for side, rig in rigs.items():
            previous, following = (anim['frames'][n]['hands'][side] for n in (index - 1, index + 1))
            begun = time.monotonic()
            result, corrected = restore_pose(rig, skins[side], surfaces[side], frame['hands'][side], scene,
                frame['body']['shoulders'][side], pose_vector(rig, previous), pose_vector(rig, following),
                1. / source['fps'], anchors=anchors[index, side], passes=args.passes,
                feasibility_iterations=args.feasibility_iterations,
                previous_hand=previous, following_hand=following)
            result.update(sweep=sweep, frame=index, side=side, seconds=time.monotonic() - begun,
                          requested_anchors=[dict(finger=a.finger, pitch=a.pitch, vertex=a.vertex)
                                             for a in anchors[index, side]])
            report['records'].append(result)
            if result['accepted']:
                anim['frames'][index]['hands'][side] = corrected
                anim['frames'][index]['body']['elbows'][side] = corrected['elbow']
            args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
            anim['diagnostic'] = __doc__
            anim['restoration'] = dict(trajectory_validated=False, window=[start, stop])
            args.output.with_suffix('.anim.json').write_text(json.dumps(anim))
            print(json.dumps(result), flush=True)

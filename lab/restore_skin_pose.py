"""Ablate fixed/movable wrists with actual skin contact and hard bounds.

Outputs are explicitly diagnostic poses. They must not be used as a passing
continuous animation: elbow temporal control and full mesh/mesh clearance are
not established by this instrument-focused experiment.
"""
import argparse
import copy
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
parser.add_argument('--frames', default='24,68')
parser.add_argument('--sides', default='RL')
parser.add_argument('--fixed-wrist', action='store_true')
parser.add_argument('--iterations', type=int, default=140)
parser.add_argument('--passes', type=int, default=4)
parser.add_argument('--feasibility-iterations', type=int, default=0)
parser.add_argument('--scale', type=float, required=True)
parser.add_argument('--anchor-mode', choices=('support', 'nearest_pad'), default='support')
args = parser.parse_args()
os.environ['HAND_SCALE'] = str(args.scale)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blockout.rig.hand import HandRig
from blockout.rig.skin_contact import pose_vector, musical_anchors, restore_pose
from blockout.skin import Skin
from blockout.subdivision import catmull_clark
from blockout.piano_geometry import PianoScene

source = json.loads(args.animation.read_text())
anim = copy.deepcopy(source)
keys = {key['pitch']: key for key in source['keyboard']}
report = dict(source=str(args.animation.resolve()), fixed_wrist=args.fixed_wrist,
              warning=__doc__, hand_scale=args.scale, anchor_mode=args.anchor_mode, records=[])
report['feasibility_iterations'] = args.feasibility_iterations
for side in args.sides:
    rig, skin = HandRig(side), Skin(side, scale=args.scale)
    surface = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    for index in map(int, args.frames.split(',')):
        frame = source['frames'][index]
        hand = frame['hands'][side]
        scene = PianoScene(source['keyboard'], frame['keys'], music_desk=True)
        vertices = surface[0] @ skin.from_frame(hand)
        anchors = musical_anchors(skin, surface[0], vertices, hand, scene, keys, mode=args.anchor_mode)
        start = time.monotonic()
        result, corrected = restore_pose(rig, skin, surface, hand, scene, frame['body']['shoulders'][side],
            pose_vector(rig, source['frames'][index - 1]['hands'][side]),
            pose_vector(rig, source['frames'][index + 1]['hands'][side]), 1. / source['fps'], anchors=anchors,
            allow_wrist=not args.fixed_wrist, iterations=args.iterations, passes=args.passes,
            feasibility_iterations=args.feasibility_iterations)
        result.update(frame=index, side=side, seconds=time.monotonic() - start,
                      requested_anchors=[dict(finger=a.finger, pitch=a.pitch, vertex=a.vertex) for a in anchors])
        report['records'].append(result)
        anim['frames'][index]['hands'][side] = corrected
        anim['frames'][index]['body']['elbows'][side] = corrected['elbow']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(result), flush=True)
anim['diagnostic'] = report['warning']
anim['restoration'] = dict(trajectory_validated=False, records=[dict(frame=r['frame'], side=r['side'],
                           accepted=r['accepted'], candidate_exported=r.get('candidate_exported', False))
                           for r in report['records']])
args.output.with_suffix('.anim.json').write_text(json.dumps(anim))

"""Matched 2x2 placement/form experiment on one fresh measured-size pose.

This is a local feasibility pilot with frozen neighbors, not a performance or
human naturalness pass. Rejected iterates are reported but never exported.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('animation', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', choices=('source', 'form', 'placement', 'combined'), required=True)
    parser.add_argument('--frame', type=int, default=24)
    parser.add_argument('--side', choices=('L', 'R'), default='R')
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--passes', type=int, default=4)
    parser.add_argument('--feasibility-iterations', type=int, default=80)
    parser.add_argument('--human-weight', type=float, default=4.)
    args = parser.parse_args()
    source = json.loads(args.animation.read_text())
    if 'hand_scale' not in source or abs(source['hand_scale'] - args.scale) > 1e-9:
        raise ValueError('Fresh pilot requires an explicit matching animation hand scale')
    if not 0 < args.frame < len(source['frames']) - 1:
        raise ValueError('Two unchanged neighboring frames are required')
    os.environ['HAND_SCALE'] = str(args.scale)
    from blockout import midi, feeling
    from blockout.rig.hand import HandRig
    from blockout.rig.human_form import PianistFormPrior
    from blockout.rig.skin_contact import musical_anchors, pose_vector, restore_pose
    from blockout.skin import Skin
    from blockout.subdivision import catmull_clark
    from blockout.piano_geometry import scene_from_animation

    files = ('blockout/rig/skin_contact.py', 'blockout/rig/contact_placement.py',
        'blockout/rig/human_form.py', 'blockout/rig/pianist_reference.json',
        'blockout/rig/hand.py', 'blockout/rig/skeleton.py', 'blockout/rig/motion.py',
        'blockout/rig/comfort.json', 'blockout/rig/furelise_cal.json',
        'blockout/skin.py', 'blockout/subdivision.py', 'blockout/skin_contacts.py',
        'blockout/piano_geometry.py', 'blockout/key_shapes.py',
        'blockout/hand_model/skin_bind.npz', 'lab/human_contact_pilot.py')
    fingerprint = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    identities = fingerprint()
    mid, chart = (ROOT / ('examples/piano/knights_suite' + suffix) for suffix in ('.mid', '.feeling.json'))
    notes = midi.read(mid)
    _, downbeats = midi.beats(mid)
    notes, _ = feeling.perform(notes, feeling.Chart(chart, downbeats), 0)
    frame = copy.deepcopy(source['frames'][args.frame])
    absolute = source['start'] + frame['t']
    required = {int(n.pitch) for n in notes if n.start + .080 <= absolute <= n.end}
    assigned = {int(p) for h in frame['hands'].values() for p in h['assigned_keys'] if p is not None}
    changes = {}
    for pitch in required & assigned:
        changes[pitch] = dict(before=frame['keys'].get(str(pitch), 0.), after=1.)
        frame['keys'][str(pitch)] = 1.
    rig, skin = HandRig(args.side), Skin(args.side, scale=args.scale)
    surface = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    scene = scene_from_animation(source, frame, music_desk=True)
    hand = frame['hands'][args.side]
    previous, following = [source['frames'][i]['hands'][args.side] for i in (args.frame - 1, args.frame + 1)]
    placement = 'calibrated' if args.variant in ('placement', 'combined') else 'source'
    weight = args.human_weight if args.variant in ('form', 'combined') else 0.
    anchors = musical_anchors(skin, surface[0], surface[0] @ skin.from_frame(hand), hand, scene,
        {k['pitch']: k for k in source['keyboard']}, required_pitches=required,
        placement=placement, active_pitches=required)
    begun = time.monotonic()
    result, corrected = restore_pose(rig, skin, surface, hand, scene, frame['body']['shoulders'][args.side],
        pose_vector(rig, previous), pose_vector(rig, following), 1. / source['fps'],
        previous_hand=previous, following_hand=following, anchors=anchors,
        iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations,
        human_prior=PianistFormPrior(), human_weight=weight, anchor_objective=True)
    result.update(schema='human-contact-pose-pilot/v1', variant=args.variant, frame=args.frame, side=args.side,
        source=str(args.animation.resolve()), source_sha256=hashlib.sha256(args.animation.read_bytes()).hexdigest(),
        code_sha256=identities, source_unchanged_during_run=fingerprint() == identities,
        score_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (mid, chart)},
        hand_scale=args.scale, geometry=source.get('geometry'), seconds=time.monotonic()-begun,
        required_pitches=sorted(required), common_key_commands=changes,
        requested_anchors=[dict(finger=a.finger, pitch=a.pitch, vertex=a.vertex, target=a.target.tolist(),
                                component=scene.specs[a.box].name, placement_prior=a.placement) for a in anchors],
        budget=dict(iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations),
        exclusions=['other hand geometry', 'actual mesh self/body/costume', 'continuous time',
                    'onsets/accidentals', 'muscle/force dynamics', 'human visual acceptance'])
    if not result['source_unchanged_during_run']:
        raise RuntimeError('Solver changed during the pilot; result must not be accepted')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['accepted']:
        args.output.with_suffix('.pose.json').write_text(json.dumps(dict(diagnostic=__doc__, frame=args.frame,
            side=args.side, hand=corrected, trajectory_validated=False), indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('variant', 'accepted', 'final_penetration_mm',
        'triangle_intersections', 'proxy_self_clearance_mm', 'anchors', 'human_form', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()

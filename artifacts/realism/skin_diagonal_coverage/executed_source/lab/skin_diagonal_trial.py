"""Matched uniform versus complete different-face triangle-choice witnesses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism/skin_diagonal_coverage'
sys.path.insert(0, str(ROOT))
from lab.skin_initialization_trial import SOURCE, NOTES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return sorted([p for p in (ROOT / 'blockout').rglob('*')
        if p.is_file() and p.suffix in ('.py', '.json', '.npz', '.obj')]
        + [SOURCE, NOTES, Path(__file__), ROOT / 'lab/skin_initialization_trial.py',
           ROOT / 'lab/SKIN_DIAGONAL_COVERAGE_PROTOCOL.md'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('uniform', 'complete'))
    args = parser.parse_args()
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    source = json.loads(SOURCE.read_text())
    from blockout.rig.pose_io import animation_hand_scale
    scale = animation_hand_scale(source)
    os.environ['HAND_SCALE'] = str(scale)
    from blockout.rig.hand import HandRig
    from blockout.rig.skin_contact import pose_vector, musical_anchors, restore_pose
    from blockout.rig.human_form import PianistFormPrior
    from blockout.skin import Skin
    from blockout.piano_geometry import scene_from_animation
    from blockout.subdivision import catmull_clark

    identities = {str(p.relative_to(ROOT)): sha(p) for p in inputs()}
    frame, following = source['frames'][:2]
    absolute = source['start'] + frame['t']
    notes = json.loads(NOTES.read_text())
    required = {int(n[2]) for n in notes if n[0] + .080 <= absolute < n[1]}
    hand, next_hand = frame['hands']['R'], following['hands']['R']
    rig, skin = HandRig('R'), Skin('R', scale=scale)
    surface = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    scene = scene_from_animation(source, frame, music_desk=True)
    anchors = musical_anchors(skin, surface[0], surface[0] @ skin.from_frame(hand), hand, scene,
        {k['pitch']: k for k in source['keyboard']}, required_pitches=required,
        placement='calibrated_soft_union', active_pitches=required)
    first = json.loads((ROOT / 'artifacts/realism/skin_initialization/both_source_R.json').read_text())
    assert [(a.finger, a.pitch, a.vertex, a.target.tolist()) for a in anchors] == [
        (a['finger'], a['pitch'], a['vertex'], a['target']) for a in first['requested_anchors']]
    began = time.monotonic()
    report, output = restore_pose(rig, skin, surface, hand, scene, frame['body']['shoulders']['R'],
        None, pose_vector(rig, next_hand), 1. / source['fps'], following_hand=next_hand,
        anchors=anchors, wrist_range=.035, iterations=100, passes=4, feasibility_iterations=80,
        human_prior=PianistFormPrior(), human_weight=4., anchor_objective=True,
        skin_pair_mode='complete_interior' if args.mode == 'complete' else 'both_interior', skin_pair_inset=.01)
    assert report['accepted'] or output == hand
    assert all(sha(ROOT / p) == h for p, h in identities.items())
    report.update(schema='mixed-diagonal-restoration/v1', mode=args.mode, side='R', frame=0,
        absolute_time=absolute, hand_scale=scale, input_sha256=identities, source_unchanged_during_run=True,
        seconds=time.monotonic() - began, source_key_commands=frame['keys'],
        requested_anchors=first['requested_anchors'], original_source_compatibility_duties=sorted(required),
        full_performance_accepted=False, full_starting_state_accepted=False, default_adoption=False,
        budget=dict(iterations=100, passes=4, feasibility_iterations=80))
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f'{args.mode}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', mode=args.mode, accepted=report['accepted'],
        self_pairs=report['mesh_self']['unexpected_surface_intersections'],
        mixed_pairs=report['mesh_self']['cross_diagonal']['unexpected_surface_intersections'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()

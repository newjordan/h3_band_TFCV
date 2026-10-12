"""Frozen actual-skin restoration at the first physical warmup pose."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism/skin_initialization'
SOURCE = ROOT / 'artifacts/realism/physical_key_actual_legacy/internal.anim.json'
NOTES = ROOT / 'artifacts/realism/performance_source_v19/notes.json'
CELLS = ('legacy_source', 'legacy_quiet', 'both_source', 'both_quiet')
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return sorted([p for p in (ROOT / 'blockout').rglob('*')
        if p.is_file() and p.suffix in ('.py', '.json', '.npz', '.obj')]
        + [SOURCE, NOTES, Path(__file__), ROOT / 'lab/run_skin_initialization.py',
           ROOT / 'lab/SKIN_INITIALIZATION_PROTOCOL.md', ROOT / 'lab/skin_initialization_graph.toml',
           ROOT / 'lab/research_skin_initialization_task.txt'])


def quiet_seed(rig, hand, required):
    import numpy as np
    from blockout.rig.skin_contact import pose_vector
    seed = np.r_[np.zeros(6), pose_vector(rig, hand)[6:]]
    changed = []
    for index, (pitch, activity) in enumerate(zip(hand['assigned_keys'], hand['activation'])):
        if activity < .05 and pitch not in required:
            start, end = rig.sl[index:index + 2]
            seed[6 + start:6 + end] = rig.q_rest[start:end]
            changed.append(index + 1)
    return seed, changed


def diagnostic_hand(rig, hand, shoulder, coordinates):
    import numpy as np
    from blockout.rig.skin_contact import continuous_elbow
    z = np.asarray(coordinates, float)
    if z.shape != (26,) or not np.isfinite(z).all():
        raise ValueError('Cannot render nonfinite diagnostic coordinates')
    x = np.r_[np.asarray(hand['root']['pos']) + z[:3] / 1000., z[3:]]
    rotation, position, points = rig.points(x, np.asarray(hand['root']['rot']))
    pads = np.asarray(rig.effectors(points, rig._rots))
    elbow = continuous_elbow(shoulder, position, hand['elbow'])
    result = copy.deepcopy(hand)
    result.update(wrist=position.tolist(), angles=rig.angles(x), elbow=elbow.tolist(),
        pads=pads.tolist(), fingers=np.asarray(points).tolist())
    result['root'].update(pos=position.tolist(), rot=rotation.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', choices=CELLS)
    parser.add_argument('side', choices=('L', 'R'))
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
    from blockout.mesh_self import quad_self_intersections

    identities = {str(p.relative_to(ROOT)): sha(p) for p in inputs()}
    frame, following = source['frames'][:2]
    absolute = source['start'] + frame['t']
    assert absolute == 316.9 and source['fps'] == 24 and scale == 1.
    notes = json.loads(NOTES.read_text())
    required_ids = [i for i, n in enumerate(notes) if n[0] + .080 <= absolute < n[1]]
    required = {int(notes[i][2]) for i in required_ids}
    assigned = {int(p) for h in frame['hands'].values() for p in h['assigned_keys'] if p is not None}
    hand, next_hand = frame['hands'][args.side], following['hands'][args.side]
    rig, skin = HandRig(args.side), Skin(args.side, scale=scale)
    surface = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    scene = scene_from_animation(source, frame, music_desk=True)
    vertices = surface[0] @ skin.from_frame(hand)
    source_audit = quad_self_intersections(vertices, surface[1])
    anchors = musical_anchors(skin, surface[0], vertices, hand, scene,
        {k['pitch']: k for k in source['keyboard']}, required_pitches=required,
        placement='calibrated_soft_union', active_pitches=required)
    expected = required & {p for p in hand['assigned_keys'] if p is not None}
    assert {a.pitch for a in anchors} >= expected
    seed, changed = quiet_seed(rig, hand, required) if args.cell.endswith('_quiet') else (None, [])
    begun = time.monotonic()
    report, output = restore_pose(rig, skin, surface, hand, scene, frame['body']['shoulders'][args.side],
        None, pose_vector(rig, next_hand), 1. / source['fps'], following_hand=next_hand,
        anchors=anchors, initial_seed=seed, wrist_range=.035,
        iterations=100, passes=4, feasibility_iterations=80,
        human_prior=PianistFormPrior(), human_weight=4., anchor_objective=True,
        skin_pair_mode='both_interior' if args.cell.startswith('both_') else 'legacy', skin_pair_inset=.01)
    assert report['accepted'] or output == hand
    assert all(sha(ROOT / p) == h for p, h in identities.items())
    report.update(schema='actual-skin-initialization/v1', cell=args.cell, side=args.side,
        source=str(SOURCE), source_sha256=sha(SOURCE), frame=0, absolute_time=absolute,
        input_sha256=identities, source_unchanged_during_run=True, source_self_audit=source_audit,
        seconds=time.monotonic() - begun, hand_scale=scale, source_key_commands=frame['keys'],
        source_held_diagnostic_ids=required_ids, source_held_diagnostic_pitches=sorted(required),
        unassigned_source_held_pitches=sorted(required - assigned), changed_seed_fingers=changed,
        held_source_compatibility_scope='Original source gates retained as diagnostic obligations; physical offset meaning unverified',
        requested_anchors=[dict(finger=a.finger, pitch=a.pitch, vertex=a.vertex, target=a.target.tolist(),
            component=scene.specs[a.box].name, placement=a.placement) for a in anchors],
        temporal_bounds_enforced=True, trajectory_validated=False, full_performance_accepted=False,
        budget=dict(iterations=100, passes=4, feasibility_iterations=80),
        full_starting_state_accepted=False, source_key_commands_unchanged=True,
        initial_surface_evaluation='Solver initialization after optional seed clipping; source_self_audit is untouched source')
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / f'{args.cell}_{args.side}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', cell=args.cell, side=args.side, accepted=report['accepted'],
        self_pairs=report['mesh_self']['unexpected_surface_intersections'],
        instrument_depth_mm=report['final_penetration_mm'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()

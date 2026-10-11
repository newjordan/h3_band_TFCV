"""Search for a contact waypoint before planning its bounded approach path.

This keeps anatomy, fixed shoulder/arm lengths, actual skin/contact and proxy
finger spacing. Neighbor speed constraints are deliberately absent because no
trajectory is being exported or accepted. Wrist range is a planning workspace,
not an allowed per-frame jump. The waypoint must still pass mesh/mesh and
naturalness review and then be connected within the original motion bounds.
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
    parser.add_argument('--frame', type=int, default=24)
    parser.add_argument('--side', choices=('L', 'R'), default='R')
    parser.add_argument('--seed', choices=('source', 'comfort', 'form_uncurl_lift'), default='source')
    parser.add_argument('--placement', choices=('calibrated', 'calibrated_soft', 'calibrated_soft_union'), default='calibrated_soft')
    parser.add_argument('--no-self-constraints', action='store_true')
    parser.add_argument('--wrist-range', type=float, default=.12)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--passes', type=int, default=5)
    parser.add_argument('--feasibility-iterations', type=int, default=120)
    parser.add_argument('--human-weight', type=float, default=4.)
    args = parser.parse_args()
    source_bytes = args.animation.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    from blockout.rig.pose_io import animation_hand_scale
    scale = animation_hand_scale(source)
    os.environ['HAND_SCALE'] = str(scale)
    from blockout.rig.hand import HandRig
    from blockout.rig.skin_contact import musical_anchors, restore_pose
    from blockout.rig.human_form import PianistFormPrior
    from blockout.skin import Skin
    from blockout.piano_geometry import scene_from_animation
    from blockout.subdivision import catmull_clark
    from blockout import midi, feeling

    paths = ('blockout/rig/skin_contact.py', 'blockout/rig/contact_placement.py',
        'blockout/rig/human_form.py', 'blockout/rig/pianist_reference.json', 'blockout/human_score.py',
        'blockout/rig/hand.py', 'blockout/rig/skeleton.py', 'blockout/rig/comfort.json',
        'blockout/rig/furelise_cal.json', 'blockout/rig/motion.py', 'blockout/rig/pose_io.py',
        'blockout/contact.py', 'blockout/performer.py', 'blockout/skin.py', 'blockout/skin_contacts.py',
        'blockout/subdivision.py', 'blockout/piano_geometry.py', 'blockout/key_shapes.py',
        'blockout/hand_model/skin_bind.npz', 'blockout/midi.py', 'blockout/feeling.py',
        'blockout/mesh_self.py', 'blockout/triangle_contact.py', 'lab/plan_contact_waypoint.py')
    fingerprint = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    identities = fingerprint()
    mid, chart = [ROOT / ('examples/piano/knights_suite' + suffix) for suffix in ('.mid', '.feeling.json')]
    notes = midi.read(mid)
    _, downbeats = midi.beats(mid)
    notes, _ = feeling.perform(notes, feeling.Chart(chart, downbeats), 0)
    frame = copy.deepcopy(source['frames'][args.frame])
    absolute = source['start'] + frame['t']
    required = {int(n.pitch) for n in notes if n.start + .080 <= absolute <= n.end}
    assigned = {int(p) for h in frame['hands'].values() for p in h['assigned_keys'] if p is not None}
    for pitch in required & assigned:
        frame['keys'][str(pitch)] = 1.
    rig, skin = HandRig(args.side), Skin(args.side, scale=scale)
    surface = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    scene = scene_from_animation(source, frame, music_desk=True)
    hand = frame['hands'][args.side]
    anchors = musical_anchors(skin, surface[0], surface[0] @ skin.from_frame(hand), hand, scene,
        {k['pitch']: k for k in source['keyboard']}, required_pitches=required,
        placement=args.placement, active_pitches=required)
    seed = None
    seed_identity = None
    if args.seed == 'comfort':
        import numpy as np
        seed = np.r_[np.zeros(6), rig.q_rest]
    elif args.seed == 'form_uncurl_lift':
        import numpy as np
        from scipy.spatial.transform import Rotation
        if args.side != 'R' or args.frame != 24:
            raise ValueError('This diagnostic seed is defined only for the measured R frame 24')
        seed_file = ROOT / 'artifacts/realism/human_pilot_form.render.anim.json'
        seed_animation = json.loads(seed_file.read_text())
        if animation_hand_scale(seed_animation) != scale:
            raise ValueError('Initialization may not resize the frozen source')
        original = seed_animation['frames'][args.frame]['hands'][args.side]
        q = np.array([original['angles'].get(b, {}).get(d, 0.) for ch in rig.chains for b,d,*_ in ch.dofs])
        dofs = [(b,d) for ch in rig.chains for b,d,*_ in ch.dofs]
        pip = dofs.index(('finger4-2', 'flex'))
        q[pip] = rig.q_rest[pip]
        position = np.asarray(original['root']['pos']) + [0., 0., .008]
        orientation = Rotation.from_matrix(np.asarray(original['root']['rot']) @ np.asarray(hand['root']['rot']).T).as_rotvec()
        seed = np.r_[(position - hand['root']['pos']) * 1000., orientation, q]
        seed_identity = dict(source=str(seed_file), source_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
            changed_ring_pip_rad=float(q[pip]), additional_wrist_lift_mm=8.,
            purpose='Initialization only; original objective, hard gates and solver budget remain unchanged')
    begun = time.monotonic()
    result, corrected = restore_pose(rig, skin, surface, hand, scene, frame['body']['shoulders'][args.side],
        None, None, 1. / source['fps'], anchors=anchors, initial_seed=seed, wrist_range=args.wrist_range,
        iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations,
        human_prior=PianistFormPrior(), human_weight=args.human_weight, anchor_objective=True,
        mesh_self_constraints=not args.no_self_constraints)
    frozen_input_unchanged = hashlib.sha256(args.animation.read_bytes()).hexdigest() == source_sha
    if seed_identity is not None:
        frozen_input_unchanged &= hashlib.sha256(Path(seed_identity['source']).read_bytes()).hexdigest() == seed_identity['source_sha256']
    result.update(schema='contact-planning-waypoint/v1', seed=args.seed, seed_identity=seed_identity, frame=args.frame, side=args.side,
        source=str(args.animation.resolve()), source_sha256=source_sha,
        code_sha256=identities, source_unchanged_during_run=fingerprint() == identities and frozen_input_unchanged,
        frozen_input_unchanged_during_run=frozen_input_unchanged,
        score_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (mid, chart)},
        hand_scale=scale, geometry=source.get('geometry'), required_pitches=sorted(required),
        key_commands=frame['keys'], seconds=time.monotonic()-begun, wrist_workspace_m=args.wrist_range,
        budget=dict(iterations=args.iterations, passes=args.passes, feasibility_iterations=args.feasibility_iterations),
        requested_anchors=[dict(finger=a.finger, pitch=a.pitch, vertex=a.vertex, target=a.target.tolist(),
                                component=scene.specs[a.box].name, placement_prior=a.placement) for a in anchors],
        temporal_bounds_enforced=False, trajectory_validated=False, performance_accepted=False,
        placement=args.placement,
        scope=__doc__)
    if not result['source_unchanged_during_run']:
        raise RuntimeError('Solver code changed during waypoint search')
    # A static waypoint is not an animation candidate, including when its local
    # instrument gate passes. Preserve this distinction in both file and report.
    result['animation_candidate_exported'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['accepted']:
        args.output.with_suffix('.waypoint.json').write_text(json.dumps(dict(hand=corrected,
            frame=args.frame, side=args.side, scope=__doc__, hand_scale=scale, key_commands=frame['keys'],
            trajectory_validated=False, performance_accepted=False), indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('seed', 'accepted', 'final_penetration_mm',
        'triangle_intersections', 'proxy_self_clearance_mm', 'anchors', 'human_form', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()

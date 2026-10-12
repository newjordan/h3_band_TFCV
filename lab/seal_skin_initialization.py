"""Seal both preregistered phases, complete audits and honest rendered figures."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.skin_initialization_trial import ART as FIRST, CELLS, SOURCE
from lab.skin_diagonal_trial import ART as SECOND


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def motion_bounds(source, frame, side):
    import numpy as np
    from blockout.rig.hand import HandRig
    from blockout.rig.skin_contact import pose_vector
    from blockout.rig.motion import rotation_angle, bend_plane_angle
    from blockout.rig.orientation import split_reference_rotations
    from blockout.performer import UPPER_ARM, FOREARM
    rig = HandRig(side)
    hand, following = frame['hands'][side], source['frames'][1]['hands'][side]
    original = source['frames'][0]['hands'][side]
    q, next_q = pose_vector(rig, hand)[6:], pose_vector(rig, following)[6:]
    root, shoulder, elbow = map(lambda v: np.asarray(v, float),
        (hand['root']['pos'], frame['body']['shoulders'][side], hand['elbow']))
    dt = 1. / source['fps']
    bounds = dict(joint_speed_rad_s=float(np.abs(q - next_q).max() / dt),
        wrist_speed_m_s=float(np.linalg.norm(root - following['root']['pos']) / dt),
        root_speed_rad_s=rotation_angle(hand['root']['rot'], following['root']['rot']) / dt,
        elbow_plane_speed_rad_s=bend_plane_angle(shoulder, root, elbow, following['elbow']) / dt,
        anatomy_violation_rad=float(max(0., (rig.lo - q).max(), (q - rig.hi).max())),
        wrist_workspace_maximum_mm=float(np.abs(root - original['root']['pos']).max() * 1000.),
        exact_arm_error_mm=1000. * max(abs(np.linalg.norm(elbow - shoulder) - UPPER_ARM),
                                       abs(np.linalg.norm(root - elbow) - FOREARM)))
    rotation = np.asarray(hand['root']['rot'])
    original_rotation = np.asarray(original['root']['rot'])
    next_rotation = np.asarray(following['root']['rot'])
    # Use the existing exported-reference policy, preserving the small fixed
    # stretch in the actual FK and render. A new 1e-8 gate on raw references
    # would incorrectly reject even the unchanged source (~6.8e-7 error).
    proper, stretch = split_reference_rotations(rotation)
    original_proper, original_stretch = split_reference_rotations(original_rotation)
    next_proper, next_stretch = split_reference_rotations(next_rotation)
    from scipy.spatial.transform import Rotation
    relative = (Rotation.from_matrix(proper) * Rotation.from_matrix(original_proper).inv()).as_rotvec()
    bounds['root_speed_rad_s'] = rotation_angle(proper, next_proper) / dt
    bounds['rotation_workspace_component_rad'] = float(np.abs(relative).max())
    bounds['wrist_root_consistency_mm'] = float(np.linalg.norm(root - hand['wrist']) * 1000.)
    bounds['proxy_minimum_spacing_mm'] = float(rig.finger_clearances(np.r_[root, np.zeros(3), q], rotation).min() * 1000.)
    bounds['root_reference_precision'] = dict(
        raw_determinant=float(np.linalg.det(rotation)),
        raw_orthogonality_maximum_error=float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        source_orthogonality_maximum_error=float(np.max(np.abs(original_rotation.T @ original_rotation - np.eye(3)))),
        fixed_stretch_maximum_difference=float(max(np.max(np.abs(stretch - original_stretch)),
            np.max(np.abs(stretch - next_stretch)))),
        policy='Existing split_reference_rotations: positive determinant, raw near-rotation error <=1e-5; proper polar factor checked at 1e-7; fixed stretch equality at 1e-10. Rendered geometry unchanged.')
    bounds['proper_root_rotation'] = bool(np.allclose(stretch, original_stretch, atol=1e-10, rtol=0)
        and np.allclose(stretch, next_stretch, atol=1e-10, rtol=0))
    bounds['passed'] = bool(bounds['joint_speed_rad_s'] <= 12. + 1e-6
        and bounds['wrist_speed_m_s'] <= .8 + 1e-6 and bounds['root_speed_rad_s'] <= 4. + 1e-6
        and bounds['elbow_plane_speed_rad_s'] <= 4. + 1e-6 and bounds['anatomy_violation_rad'] < 1e-9
        and bounds['wrist_workspace_maximum_mm'] <= 35. + 1e-6
        and bounds['rotation_workspace_component_rad'] <= .3 + 1e-6
        and bounds['wrist_root_consistency_mm'] < 1e-6 and bounds['proxy_minimum_spacing_mm'] >= -.001
        and bounds['exact_arm_error_mm'] < .001 and bounds['proper_root_rotation'])
    bounds['scope'] = 'Independent final-coordinate checks to original next pose; fixed shoulders; no continuous-path proof'
    return bounds


def actual_summary(path):
    audit = read(path)
    assert audit['status'] == 'complete' and audit['source_unchanged_during_run']
    assert all(sha(ROOT / p) == h for p, h in audit['code_sha256'].items())
    row = audit['records'][0]
    assert row['diagonal_choice_complete'] and row['lbs_parity_maximum_error_mm'] < .01
    result = dict(audit=str(path), audit_sha256=sha(path), parity_error_mm=row['lbs_parity_maximum_error_mm'],
        surface_intersection_clear=row['surface_intersection_clear'],
        degenerate_triangles=row['degenerate_triangles'], hands={},
        scope='Maximum diagnostic triangle-pair bucket across pure0, pure1 and mixed different-face choices; not collision-event count')
    for side in ('L', 'R'):
        result['hands'][side] = dict(
            self_pair_buckets=[r.get(side + '_self', 0) for r in row['scope_counts']],
            maximum_self_pair_bucket=max(r.get(side + '_self', 0) for r in row['scope_counts']),
            instrument=row['instrument'][side], assigned_support=row['assigned_support'][side])
    result['interhand_pair_buckets'] = [r.get('interhand', 0) for r in row['scope_counts']]
    return result


def requested_anchor_checks(directory, numerical):
    import numpy as np
    from scipy.spatial import cKDTree
    from lab.skin_deformation_audit import expected_geometry
    from blockout.piano_geometry import scene_from_animation
    from blockout.skin_contacts import KeyTop, key_support
    animation = read(directory / 'diagnostic.anim.json')
    record = read(directory / 'snapshots.json')['records'][0]
    with np.load(record['mesh'], allow_pickle=False) as data:
        vertices = data['vertices']
        expected, _, _, _, fields = expected_geometry(record, animation['hand_scale'], data)
    _, mapping = cKDTree(expected).query(vertices)
    assert np.array_equal(np.sort(mapping), np.arange(len(expected)))
    scene = scene_from_animation(animation, animation['frames'][0], music_desk=True)
    result = []
    for anchor in numerical['requested_anchors']:
        cpu_vertex = fields['R'][0][anchor['vertex']]
        actual_vertex = int(np.flatnonzero(mapping == cpu_vertex)[0])
        point = vertices[actual_vertex]
        box = next(i for i, spec in enumerate(scene.specs) if spec.name == anchor['component'])
        local = (point - scene.origins[box]) @ scene.rotations[box]
        top_gap = float((local[2] - scene.upper[box, 2]) * 1000.)
        margin = float(KeyTop(scene, anchor['pitch']).margin(point) * 1000.)
        supported = key_support(vertices, [actual_vertex], anchor['pitch'], scene)['supported']
        result.append(dict(finger=anchor['finger'], pitch=anchor['pitch'], actual_vertex=actual_vertex,
            top_gap_mm=top_gap, finite_union_margin_mm=margin, exposed_top_support=bool(supported),
            passed=bool(.009 <= top_gap <= .101 and margin >= .009 and supported),
            tolerance_scope='Same 10-µm target gap and 1-µm constraint tolerance; 100-µm top-gap ceiling'))
    return result


def main():
    from PIL import Image, ImageDraw, ImageFont
    source = read(SOURCE)
    os.environ['HAND_SCALE'] = str(source['hand_scale'])
    first_runs, first_verify = read(FIRST / 'runs.json'), read(FIRST / 'actual_verification.json')
    second_verify = read(SECOND / 'verification.json')
    assert first_runs['status'] == first_verify['status'] == second_verify['status'] == 'complete'
    assert second_verify['source_unchanged_during_run']
    assert all(sha(Path(p)) == h for p, h in second_verify['input_sha256'].items())
    for record in second_verify['records']:
        assert sha(Path(record['source'])) == record['source_sha256']
        assert sha(Path(record['audit'])) == record['audit_sha256']
        assert sha(Path(record['image'])) == record['image_sha256']
    # Historical executed sources are never rewritten to current hashes.
    for relative, digest in first_runs['input_sha256'].items():
        current = ROOT / relative
        frozen = FIRST / 'executed_source' / relative
        assert sha(current) == digest or (frozen.exists() and sha(frozen) == digest)
    first_results = {}
    for cell in ('original',) + CELLS:
        actual = actual_summary(FIRST / cell / 'audit_complete_diagonals.json')
        records = {side: read(FIRST / f'{cell}_{side}.json') for side in ('L', 'R')} if cell != 'original' else {}
        first_results[cell] = dict(actual=actual, numerical=records,
            historical_uniform_only_acceptance_invalidated=cell == 'both_source')
    second_results = {}
    for mode in ('uniform', 'complete', 'complete_seeded'):
        numerical = read(SECOND / f'{mode}.json')
        for p, h in numerical['input_sha256'].items():
            archived = SECOND / 'executed_source' / p
            assert sha(ROOT / p) == h or (archived.exists() and sha(archived) == h)
        actual = actual_summary(SECOND / mode / 'audit.json')
        frame = read(SECOND / mode / 'diagnostic.anim.json')['frames'][0]
        bounds = motion_bounds(source, frame, 'R')
        assert frame['keys'] == source['frames'][0]['keys']
        assert frame['hands']['L'] == source['frames'][0]['hands']['L']
        required = {(a['finger'], a['pitch']) for a in numerical['requested_anchors']}
        supported = {(a['finger'], a['pitch']) for a in actual['hands']['R']['assigned_support'] if a['supported']}
        actual_anchors = requested_anchor_checks(SECOND / mode, numerical)
        local_pass = bool(numerical['accepted'] and bounds['passed'] and actual['degenerate_triangles'] == 0
            and actual['hands']['R']['maximum_self_pair_bucket'] == 0
            and actual['hands']['R']['instrument']['instrument_clear'] and required <= supported
            and all(a['passed'] for a in actual_anchors) and not any(actual['interhand_pair_buckets']))
        second_results[mode] = dict(numerical=numerical, actual=actual, independent_motion_bounds=bounds,
            independently_verified_local_right_hand=local_pass, full_starting_state_accepted=False,
            actual_requested_anchor_checks=actual_anchors,
            experimental_phase='exploratory staged seed' if mode == 'complete_seeded' else 'matched cold start',
            total_work_seconds=numerical['seconds'] + (numerical.get('seed_identity') or {}).get('preparation_seconds', 0.))
    old = first_results['both_source']['numerical']['R']['numerical_iterate']['solver_coordinates']
    control = second_results['uniform']['numerical']['numerical_iterate']['solver_coordinates']
    import numpy as np
    replay_difference = float(np.max(np.abs(np.asarray(old) - control)))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    first_grid = Image.new('RGB', (1024, 680), '#10151d')
    draw = ImageDraw.Draw(first_grid)
    for index, cell in enumerate(CELLS):
        x, y = (index % 2) * 512, (index // 2) * 340
        picture = Image.open(FIRST / cell / 'render/f_00000.png').convert('RGB')
        assert picture.size == (512, 288)
        first_grid.paste(picture, (x, y + 28))
        draw.text((x + 8, y + 5), cell.replace('_', ' ') + ' — failed diagnostic', font=font, fill='white')
        a = first_results[cell]['actual']['hands']
        draw.text((x + 8, y + 320), f"Complete buckets: R {a['R']['maximum_self_pair_bucket']}, L {a['L']['maximum_self_pair_bucket']}", font=small, fill='white')
    first_grid.save(FIRST / 'initialization_grid.png')
    comparison = Image.new('RGB', (1536, 356), '#10151d')
    draw = ImageDraw.Draw(comparison)
    for index, mode in enumerate(('uniform', 'complete', 'complete_seeded')):
        x = index * 512
        picture = Image.open(SECOND / mode / 'render/f_00000.png').convert('RGB')
        assert picture.size == (512, 288)
        comparison.paste(picture, (x, 28))
        result = second_results[mode]
        label = 'R local checks pass; performance unfinished' if result['independently_verified_local_right_hand'] else 'R local checks fail; diagnostic only'
        draw.text((x + 8, 5), mode + ' witnesses — ' + label, font=small, fill='white')
        draw.text((x + 8, 320), f"R complete bucket {result['actual']['hands']['R']['maximum_self_pair_bucket']}; original L unchanged and rejected", font=small, fill='white')
    draw.text((8, 340), 'Static starting-pose comparison, 316.9 s. No continuous-motion or naturalness approval.', font=small, fill='white')
    comparison.save(SECOND / 'comparison.png')
    log = ROOT / 'artifacts/realism/skin_diagonal_tests.log'
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log.read_text())
    assert match and int(match[1]) == 195
    episode_paths = list((ROOT / 'artifacts/realism/skin_diagonal_research').rglob('episodes/*.json'))
    assert len(episode_paths) == 1
    episode = read(episode_paths[0])
    import shutil
    episode_target = ROOT / 'lab/realism_results/skin_diagonal_review_episode.json'
    shutil.copy2(episode_paths[0], episode_target)
    review_result = read(ROOT / 'artifacts/realism/skin_diagonal_review_result.json')
    assert review_result['status'] == 'completed'
    method_accepted = episode['termination']['kind'] == 'completed' and any(
        row['outcome'] == 'accepted' for t in episode['traces'] for row in t.get('control_credit', []))
    report = dict(schema='skin-initialization-complete-choice-study/v1', status='awaiting_visual_review',
        source=str(SOURCE), source_sha256=sha(SOURCE), first_phase=first_results, second_phase=second_results,
        complete_diagonal_coverage=True, full_performance_accepted=False, default_adoption=False,
        independent_right_hand_passes=sum(r['independently_verified_local_right_hand'] for r in second_results.values()),
        original_source_held_pitches=[49, 56, 61, 64, 80], unassigned_source_held_pitches=[56, 64],
        fixed_source_key_commands=source['frames'][0]['keys'],
        uniform_control_replay_maximum_solver_coordinate_difference=replay_difference,
        source_offset_scope='Original source gates are diagnostic compatibility duties, not verified physical key-off truth',
        original_soundtrack_unchanged=True,
        unit_tests=dict(passed=195, seconds=float(match[2]), log=str(log), log_sha256=sha(log)),
        native_review=dict(method_accepted=bool(method_accepted), episode=episode['episode_id'],
            method_scope='Corrected geometry and two matched cold trials; exploratory staged seed not separately reviewed',
            episode_path=str(episode_target), episode_sha256=sha(episode_target), empirical_approval=False,
            result_path=str(ROOT / 'artifacts/realism/skin_diagonal_review_result.json')),
        figures={str(p): sha(p) for p in [FIRST / 'initialization_grid.png', SECOND / 'comparison.png']},
        limitations=['One frozen window start', 'Source assignments/offset meaning and key/foot mechanics unresolved',
            'Left hand remains rejected', 'Positive mesh distance/containment/body/costume/forces/naturalness unverified',
            'Original next pose and nonlinear continuous trajectory not certified', 'Drummer still rejected on stress cases'],
        previous_goal_turn='Full-suite media delivery verified; no new rig improvement. This turn implements and measures restoration/triangle-coverage corrections.')
    path = ROOT / 'lab/realism_results/skin_initialization_20261012.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], independent_right_hand_passes=report['independent_right_hand_passes'],
        control_replay_difference=replay_difference, native_method_accepted=method_accepted)), flush=True)


if __name__ == '__main__':
    main()

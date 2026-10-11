"""Archive the complete deformation, joint-isolation and key-union studies."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / 'artifacts/realism', ROOT / 'lab/realism_results'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


deform = read(ART / 'deformation_audit_20261011.json')
sweep = read(ART / 'pose_fold_sweep_20261011.json')
verify = read(ART / 'fold_verify_audit_20261011.json')
assert all(r['status'] == 'complete' and r['source_unchanged_during_run'] for r in (deform, sweep, verify))
assert len(deform['records']) == 16 and len(sweep['records']) == 15 and len(verify['records']) == 4
assert all(r['bone_matrix_error'] == 0. for r in deform['records'])
assert all(r['self_intersections'] == 0 for r in deform['records'] if r['context'] == 'bind_control')
trials = {f'{surface}_{seed}': read(ART / f'waypoint_union_{surface}_{seed}.json')
          for surface in ('fixed', 'union') for seed in ('source', 'lift')}
base = trials['fixed_source']
for trial in trials.values():
    for key in ('source_sha256', 'code_sha256', 'score_sha256', 'hand_scale', 'geometry',
                'required_pitches', 'key_commands', 'budget', 'wrist_workspace_m'):
        assert trial[key] == base[key], key
    assert trial['source_unchanged_during_run'] and trial['frozen_input_unchanged_during_run']
    assert not trial['animation_candidate_exported'] and not trial['performance_accepted']
    assert not trial['temporal_bounds_enforced']
    if trial['seed_identity'] is not None:
        assert sha(Path(trial['seed_identity']['source'])) == trial['seed_identity']['source_sha256']

summary = {n: dict(accepted=t['accepted'], instrument_depth_mm=t['final_penetration_mm'],
    instrument_triangles=t['triangle_intersections'], self_pairs=t['mesh_self']['unexpected_surface_intersections'],
    supported=sum(a['exposed_top_support'] for a in t['anchors']), requested=len(t['anchors']),
    proxy_gap_mm=t['proxy_self_clearance_mm'], soft_form_cost=t['human_form']['numerical_iterate']['cost'],
    animation_candidate_exported=t['animation_candidate_exported'], seconds=t['seconds']) for n,t in trials.items()}
deform_summary = [{k:r[k] for k in ('context','mode','subdivision_levels','self_intersections','degenerate_triangles','scope_counts',
                    'instrument','assigned_support','lbs_parity_maximum_error_mm')} for r in deform['records']]
selected = [{k:r[k] for k in ('context','pose_changes','changed_side','self_intersections','scope_counts','instrument')} for r in verify['records']]
repeats = {name:read(ART / f'waypoint_union_replay_{name}.json') for name in ('source','lift')}
actual_union = read(ART / 'union_verify_matched_audit_20261011.json')
assert actual_union['status'] == 'complete' and actual_union['source_unchanged_during_run']
assert len(actual_union['records']) == 2 and not actual_union['candidate_exported']
for name, repeat in repeats.items():
    primary = trials[f'union_{name}']
    for key in ('source_sha256','score_sha256','hand_scale','geometry','required_pitches','key_commands',
                'budget','wrist_workspace_m','final_penetration_mm','triangle_intersections',
                'proxy_self_clearance_mm','anchors','mesh_self'):
        assert repeat[key] == primary[key], (name,key)
    assert repeat['source_unchanged_during_run'] and repeat['frozen_input_unchanged_during_run']
    changed_code = [p for p in primary['code_sha256'] if primary['code_sha256'][p] != repeat['code_sha256'][p]]
    assert changed_code == ['blockout/rig/skin_contact.py'], changed_code
    row = next(r for r in actual_union['records'] if r['context'] == f'union_{name}')
    assert row['key_commands'] == primary['key_commands']
    assert row['pose_changes']['rejected_checkpoint_sha256'] == sha(ART / f'waypoint_union_replay_{name}.json')
actual_union_summary = {r['context']:dict(
    right_hand_instrument_depth_mm=r['instrument']['R']['maximum_penetration_mm'],
    right_hand_instrument_triangles=r['instrument']['R']['triangle_intersections'],
    right_hand_self_pairs=max(c.get('R_self',0) for c in r['scope_counts']),
    unchanged_left_instrument_depth_mm=r['instrument']['L']['maximum_penetration_mm'],
    unchanged_left_self_pairs=max(c.get('L_self',0) for c in r['scope_counts']),
    maximum_cpu_blender_parity_error_mm=r['lbs_parity_maximum_error_mm'],
    accepted=False) for r in actual_union['records']}
episode = read(OUT / 'deformation_review_episode.json')
review = read(OUT / 'deformation_review_result.json')
assert episode['termination']['kind'] == 'completed' and review['status'] == 'completed'
assert {t['role'] for t in episode['traces']} == {'evidence','counterevidence','constraints','synthesize','verify'}
assert all(t['termination']['kind'] == 'answer' for t in episode['traces'])
assert 'PASS for the method only' in review['answer']
log = (ART / 'tests_final_deformation.log').read_text()
match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK', log)
assert match and int(match[1]) == 135

fig, axes = plt.subplots(3, 2, figsize=(12.5, 12), layout='constrained')
contexts = ('fresh', 'contact', 'form', 'bind_control')
for mode,color in (('linear','#c86a3a'), ('quaternion','#2b849f')):
    rows = [next(r for r in deform['records'] if r['context']==n and r['mode']==mode and r['subdivision_levels']==2) for n in contexts]
    axes[0,0].plot(range(4), [r['self_intersections'] for r in rows], 'o-', label=mode, color=color)
    axes[0,1].plot(range(3), [max(h['maximum_penetration_mm'] for h in r['instrument'].values()) for r in rows[:3]], 'o-', label=mode,color=color)
axes[0,0].set(xticks=range(4),xticklabels=['Fresh','Contact correction','Form guidance','Bind control'],ylabel='Both-hand crossing pairs (max of 2 diagonals)')
axes[0,1].set(xticks=range(3),xticklabels=['Fresh','Contact correction','Form guidance'],ylabel='Actual skin / instrument penetration (mm)')
axes[0,0].set_title('Identical bones: quaternion deformation does not clear playing poses')
axes[0,1].set_title('Instrument clearance remains a separate gate')
ring = [r for r in sweep['records'] if r['side']=='R']
angles = [r['form']['angles_deg'][2][1] for r in ring]
axes[1,0].plot(angles,[r['self_crossing_pairs'] for r in ring],'o-',color='#2b849f')
axes[1,1].plot(angles,[r['instrument']['maximum_penetration_mm'] for r in ring],'o-',color='#c86a3a')
for ax in axes[1]: ax.set(xlabel='Measured ring PIP flexion (degrees)')
axes[1,0].set(ylabel='Right-hand crossing pairs', title='Uncurling alone removes self folds')
axes[1,1].set(ylabel='Instrument penetration (mm)',title='The same change drives skin through keys')
for seed,color in (('source','#c86a3a'),('lift','#2b849f')):
    rows = [summary[f'{s}_{seed}'] for s in ('fixed','union')]
    axes[2,0].plot([0,1],[r['self_pairs'] for r in rows],'o-',color=color,label=seed+' initialization')
    axes[2,1].plot([0,1],[r['instrument_depth_mm'] for r in rows],'o-',color=color,label=seed+' initialization')
for ax in axes[2]: ax.set(xticks=[0,1],xticklabels=['Fixed head/stem component','Connected physical key top'])
axes[2,0].set(ylabel='Right-hand crossing pairs',title='Matched initialization × contact-surface experiment')
axes[2,1].set(ylabel='Instrument penetration (mm)',title='Same hard gates and budget; no animation acceptance')
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.tick_params(axis='x',labelsize=8)
for ax in (axes[0,0],axes[0,1],axes[2,0],axes[2,1]): ax.legend(fontsize=8)
fig.suptitle('Piano realism diagnosis: bind geometry is clear; playing contact and self geometry must be planned together')
fig.savefig(ART / 'deformation_and_union.png',dpi=150)
plt.close(fig)

OUT.mkdir(exist_ok=True)
files = ('deformation_audit_20261011.json','pose_fold_sweep_20261011.json','fold_verify_audit_20261011.json',
    'waypoint_seed_source_control.json','waypoint_seed_uncurl_lift.json','pose_fold_sweep_firstpartial.json',
    'waypoint_union_replay_source.json','waypoint_union_replay_lift.json','union_verify_matched_audit_20261011.json',
    'union_verify_audit_20261011.json')
for name in files: shutil.copy2(ART/name, OUT/name)
for source,target in (('tests_final_deformation.log','deformation_and_union_tests_20261011.txt'),
                     ('deformation_audit_base_projection_failure.log','deformation_base_projection_firstfailure.txt'),
                     ('pose_fold_sweep_firstfailure.log','pose_fold_sweep_firstfailure.txt'),
                     ('union_verify_audit_firstfailure.log','union_verify_duplicate_metadata_firstfailure.txt'),
                     ('pose_fold_sweep_firstattempt.py','pose_fold_sweep_firstattempt.txt')):
    shutil.copy2(ART/source,OUT/target)
report = dict(schema='deformation-and-contact-union-progress/v1', date='2026-10-11',
    hypothesis='Distinguish deformation artifacts from pose/contact incompatibility at identical bone states; '
               'allow physically connected white-key top sliding without artificial component seams.',
    deformation_ablation=deform['snapshots']['ablation'], deformation_results=deform_summary,
    joint_sweep_scope='One ring PIP coordinate and a 3×3 left thumb CMC grid; wrist, elbow and all other q are frozen. '
                      'Selected changed poses reconstructed and independently audited in Blender.',
    selected_actual_mesh_results=selected,
    static_union_ablation='Same measured fresh R frame24, anatomy, score/key commands, 120 mm static wrist workspace, '
        'human weight4 and 100×4 plus80 feasibility budget. Fixed component vs physical key-top union × source vs '
        'form-guided uncurl+8mm lift initialization. Original objective is unchanged by initialization. '
        'Finite outer-boundary margin, 10–100µm contact gap, instrument triangles, self mesh, proxy spacing and exact arms remain hard.',
    union_summary=summary, union_trials=trials,
    union_verification_repeats=repeats, actual_union_verification=actual_union_summary,
    verification_repeat_scope='Only diagnostic checkpoint metadata changed after the four matched primary runs. '
        'The two union repeats reproduce their primary numerical gates exactly; actual Blender uses the identical '
        'score-held key commands. Right-hand instrument contacts clear, but both self surfaces and the unchanged left hand reject.',
    reviewed_method=dict(episode_id=episode['episode_id'], declared_receipt_sha256=episode['receipt_sha256'],
        episode_file_sha256=sha(OUT/'deformation_review_episode.json'), terminal_traces=len(episode['traces']),
        unique_nodes=5, synthesis_review_retries=1, scope='Methodological review using supplied primary notes; not empirical rig acceptance',
        completed_connected_planning_weights_deformation_factorial=False),
    unit_tests=dict(passed=int(match[1]),seconds=float(match[2]),log_sha256=sha(ART/'tests_final_deformation.log')),
    source_precision_issue='Unchanged legacy left pinky abduction has 2.0152775e−6 rad excess from 5-decimal export. '
        'It is reported, not silently clamped. New production exports already retain solver precision.',
    retained_failed_attempts=['Base cage was wrongly compared to a limit projection; corrected, regression tested, full audit rerun.',
        'First joint sweep rejected legacy left q rounding excess after completing R rows; preserved, full diagnostic rerun with explicit bounds ledger.',
        'Generic audit first duplicated candidate metadata; report merge corrected and failure log retained.',
        'First independent union replay used source key depths instead of trial score-held commands. That mismatched audit is retained '
        'as union_verify_audit_20261011.json; it is not a trial result. Corrected matched reconstruction/audit is retained separately.'],
    durable_reports={name:sha(OUT/name) for name in files},
    conclusion='Bind control has zero crossings at both levels. Quaternion deformation leaves all playing contexts rejected. '
        'Actual Blender verifies that isolated uncurling clears the changed hand but creates instrument penetration. '
        'Simultaneous key contact, hand posture and wrist planning must be solved together. The connected key-top option '
        'removes an artificial component seam but is not a force, naturalness or continuous-motion certificate.',
    animation_candidate_exported=False, full_performance_accepted=False,
    exclusions=['positive minimum distance and containment','body/costume/drummer meshes','complete musical coverage',
                'forces/muscles/piano action','continuous time','natural motion acceptance'])
(OUT/'deformation_and_union_20261011.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(union_summary=summary, tests=report['unit_tests'], full_performance_accepted=False),indent=2))

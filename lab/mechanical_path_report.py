"""Record completed mechanical/path evidence separately from live trials."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism'
OUT = ROOT / 'lab/realism_results'
read = lambda name: json.loads((ART / (name + '.json')).read_text())
raw = read('skin_motion_exposed_holds_window')
coupled = read('restore_coupled_path')
rectangular = read('piano_mechanical_interference_rectangular')
notched = read('piano_mechanical_interference_notched')
parity = read('skin_parity_notched_probe')
shape_motion = read('skin_motion_notched_probe')
assert not coupled['accepted'] and not coupled['candidate_exported']
assert parity['passed'] and notched['clear']
test_log = (ART / 'tests_path_geometry_drums.log').read_text()
test_count = int(re.search(r'Ran (\d+) tests', test_log)[1])
assert test_log.rstrip().endswith('OK')
drums = json.loads((OUT / 'drummer_crosshand_20261011.json').read_text())
live = {}
for name in ('path_notched_frozen', 'path_notched_movable'):
    path = ART / (name + '.json')
    if path.exists():
        data = json.loads(path.read_text())
        live[name] = dict(status='completed' if 'accepted' in data else 'running',
            accepted=data.get('accepted'), candidate_exported=data.get('candidate_exported', False),
            attempted_local_solves=len(data['records']), control_poses_frozen=data['control_poses_frozen'],
            solver_budget=data['solver_budget'], report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for label, frames, color in (
    ('Exported-pose interpolation', raw['records'], '#c86a3a'),
    ('Coupled frozen knots (rejected)', coupled['final_observations'], '#2b849f')):
    axes[0, 0].plot(range(len(frames)), [max(h['maximum_penetration_mm'] for h in f['hands'].values()) for f in frames],
                    '.-', color=color, label=label)
    axes[0, 1].plot(range(len(frames)), [len(f.get('held_score', f).get('unsupported_pitches', [])) for f in frames],
                    '.-', color=color, label=label)
for ax in axes[0]:
    ax.set_xlabel('240 Hz observation')
    ax.legend(fontsize=8)
axes[0, 0].set_ylabel('Worst actual-skin penetration (mm)')
axes[0, 1].set_ylabel('Required held pitches without support')
for label, data, color in (('Rectangular piano', rectangular, '#c86a3a'), ('Notched prototype', notched, '#2b849f')):
    axes[1, 0].plot(range(4), [max([c['minimum_translation_mm'] for c in f['contacts']] or [0.]) for f in data['records']],
                    '.-', color=color, label=label)
axes[1, 0].set_xticks(range(4), ['Rest', '23', '24', '25'])
axes[1, 0].set_xlabel('Saved key configuration')
axes[1, 0].set_ylabel('Key/key or key/case interference (mm)')
axes[1, 0].legend(fontsize=8)
axes[1, 1].set_axis_off()
axes[1, 1].text(0., .95,
    f'{test_count} unit tests pass; no full rig acceptance.\n\n'
    'Coupled path: 18 → 4 clipped observations.\n'
    'Four held-pitch observations still unsupported.\n'
    'No partly failed animation exported.\n\n'
    'Key-shape change alone leaves hand failures intact.\n'
    f"Blender parity: {max(r['maximum_error_mm'] for r in parity['records']):.4f} mm.\n\n"
    'Drummer: 10 demo / 16 roll hits pass proxy gates.\n'
    '12/12 stress patterns rejected.\n'
    'Full mesh/force/continuous-time checks remain open.',
    va='top', fontsize=11)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.grid(alpha=.2)
figure.suptitle('Rig realism research: improved sampled feasibility, incomplete natural motion')
figure.savefig(ART / 'mechanical_path_experiment.png', dpi=160)
report = dict(schema='rig-mechanical-path-progress/v1', test_count=test_count,
    hypothesis='Coupled neighboring contact solves improve sampled feasibility; physical piano solids must also be nonintersecting.',
    ablation='Mechanical shape is compared on identical key configurations. Shape-only hand motion is unchanged. '
             'The exploratory path also changes assigned held-key commands and shared neighboring poses, so it does not isolate one cause.',
    completed=dict(coupled=coupled, rectangular_mechanics=rectangular, notched_mechanics=notched,
                   notched_blender_parity=parity, shape_only_motion=shape_motion),
    reviewed_protocol=dict(episode='276a908d249378e4be774f41cd01f301b2f2055e5f27a0c2c9d3507950040fb9',
                           approval_scope='experiment method; supplied primary reading notes; not rig acceptance'),
    matched_path_comparison=live,
    drummer_summary=dict(demo=drums['demo']['qa'], roll=drums['roll']['qa'], stress=drums['stress_summary']),
    hand_scale=dict(stress=1., measured_opt_in=.7284, current_default=1.,
                    note='Measured 17.7 cm size is deliberately disabled by default in the existing planner; never silently resize frozen inputs.'),
    conclusion='Mechanical prototype passes tested configurations and rendered parity. Sampled hand motion still fails, '
               'and the reviewed replanning comparison remains pending until both jobs have terminal results.',
    exclusions=['complete repertoire', 'actual mesh self/interhand/body/costume', 'CCD', 'force dynamics', 'human naturalness', 'full drummer renderer'])
(OUT / 'mechanical_path_20261011.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(test_count=test_count, matched_path_comparison=live, completed_path_accepted=coupled['accepted']), indent=2))

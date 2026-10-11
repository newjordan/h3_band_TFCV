"""Record the matched human-contact pilot, including its rejected treatments."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / 'artifacts/realism', ROOT / 'lab/realism_results'
variants = ('source', 'form', 'placement', 'combined')
trials = {v: json.loads((ART / f'human_pilot_{v}.json').read_text()) for v in variants}
baseline = trials['source']
for trial in trials.values():
    for key in ('source_sha256', 'code_sha256', 'score_sha256', 'hand_scale', 'geometry',
                'required_pitches', 'common_key_commands', 'budget', 'contact_objective'):
        assert trial[key] == baseline[key], key
    assert trial['source_unchanged_during_run']
    assert trial['candidate_exported'] == trial['accepted']
assert all(trials[v]['accepted'] for v in ('source', 'form'))
assert not any(trials[v]['accepted'] for v in ('placement', 'combined'))
parity_path = ART / 'skin_parity_measured_form.json'
parity = json.loads(parity_path.read_text())
assert parity['passed'] is True and parity['validation'] == 'complete'
audit = json.loads((OUT / 'skin_audit_measured_notched_20261011.json').read_text())
anim = json.loads((ART / 'measured_notched_overlap.anim.json').read_text())
intervals = {}
for name, predicate in (('warmup', lambda t: t < 248.), ('main', lambda t: 248. <= t <= 250.),
                        ('tail', lambda t: t > 250.)):
    rows = [r for r in audit['records'] if predicate(anim['start'] + r['time'])]
    intervals[name] = dict(frames=len(rows), affected_frames=sum(
        any(not h['instrument_clear'] for h in r['hands'].values()) for r in rows),
        maximum_penetration_mm=max(h['maximum_penetration_mm'] for r in rows for h in r['hands'].values()))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
fingers = np.arange(2, 6)
form = trials['form']['human_form']['numerical_iterate']
iqr = np.asarray(form['target_iqr_deg'])
for column, title in ((0, 'MCP flexion'), (1, 'PIP flexion')):
    ax = axes[0, column]
    ax.fill_between(fingers, iqr[:, 0, column], iqr[:, 1, column], color='#87b5a6', alpha=.25,
                    label='Soft pooled phase IQR')
    for label, data, color in (
        ('Fresh input', baseline['human_form']['before'], '#777777'),
        ('Contact-only correction', baseline['human_form']['numerical_iterate'], '#c86a3a'),
        ('Posture-guided correction', form, '#2b849f')):
        ax.plot(fingers, np.asarray(data['angles_deg'])[:, column], '.-', color=color, label=label)
    ax.set(xlabel='Finger', ylabel=f'{title} (degrees)', xticks=fingers)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
for v, color in (('source', '#c86a3a'), ('form', '#2b849f')):
    anchors = trials[v]['anchors']
    axes[1, 0].plot([a['finger'] for a in anchors], [a['contact_depth_mm'] for a in anchors],
                    '.-', color=color, label=v + ' (accepted pose only)')
anchors = trials['placement']['requested_anchors']
axes[1, 0].plot([a['finger'] for a in anchors], [a['target'][1] * 1000. for a in anchors],
                'o--', color='#777777', label='Requested calibrated depths (both trials rejected)')
axes[1, 0].set(xlabel='Held finger', ylabel='Skin contact depth from key front (mm)', xticks=range(1, 6))
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=.2)
for i, v in enumerate(variants):
    t = trials[v]
    axes[1, 1].plot(i, t['human_form']['numerical_iterate']['cost'], 'o',
        color='#2b849f' if t['accepted'] else '#c86a3a',
        markerfacecolor='#2b849f' if t['accepted'] else 'none')
axes[1, 1].axhline(baseline['human_form']['before']['cost'], color='#777777', linestyle='--', label='Input')
axes[1, 1].set(xticks=range(4), xticklabels=['Source placement', '+ Form', 'Calibrated placement', '+ Both'],
               ylabel='Soft reference deviation (dimensionless)')
axes[1, 1].tick_params(axis='x', labelsize=8)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=.2)
fig.suptitle('Measured-size pilot: posture improves locally; deeper placement and natural motion remain unresolved')
fig.savefig(ART / 'human_contact_pilot.png', dpi=160)

report = dict(schema='human-contact-pilot-summary/v1',
    hypothesis='Soft phase-conditioned anatomical form and finite exposed contact-depth priors can improve posture without weakening hard gates.',
    ablation='One fresh measured-size R-frame-24 input, unchanged neighbors, identical level-2 skin, anatomy/spacing/speed limits, '
             'score/held-key commands, objective form and per-treatment solver budget. Placement and form weight are the two factors.',
    trials=trials, baseline_skin_audit=audit, baseline_intervals=intervals,
    renderer_parity=parity, renderer_parity_sha256=hashlib.sha256(parity_path.read_bytes()).hexdigest(),
    visual_review=dict(camera=dict(view='three4', eye=[.30, -.25, .19], look=[.16, .04, .025], lens_mm=50,
                                   resolution=[1024, 768], follows_hand=False),
        conclusion='The right-hand corrections remain overcurled with thumb/ring/pinky contacts near the fronts. '
                   'The guided image changes posture but is not accepted as natural performance.', passed=False),
    conclusion='Contact-only and posture-guided corrections pass the local gates with all four requested held skin contacts. '
               'Guidance reduces reference deviation from 7.2424 to 3.7351, but the naturalness review fails. '
               'Both calibrated-depth treatments are rejected (2.1703 mm penetration, proxy self overlap, four unsupported anchors). '
               'This local solve cannot establish global infeasibility; broader contact/wrist planning remains necessary.',
    full_performance_accepted=False,
    exclusions=['complete repertoire', 'continuous collision detection', 'actual mesh self/interhand/body/costume',
                'force/muscle/action dynamics', 'onsets/accidentals', 'thumb reference validation'])
OUT.mkdir(exist_ok=True)
(OUT / 'human_contact_pilot_20261011.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(accepted={v: t['accepted'] for v, t in trials.items()}, baseline_intervals=intervals,
                      renderer_parity=parity['passed'], human_naturalness=False), indent=2))

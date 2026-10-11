"""Record held-note restoration experiments, including rejected outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', type=Path, default=Path('artifacts/realism'))
parser.add_argument('--durable', type=Path, default=Path('lab/realism_results/held_contact_20261011.json'))
args = parser.parse_args()
names = ('paired_holds_L25_baseline', 'paired_holds_L25_seeded',
         'paired_holds_v2_L25_baseline', 'paired_holds_v2_L25_seeded',
         'restore_seed_holds_window', 'skin_motion_seed_holds_score_window',
         'restore_exposed_holds_window', 'skin_motion_exposed_holds_window',
         'skin_parity_exposed_holds_window')
reports = {name: json.loads((args.artifacts / (name + '.json')).read_text()) for name in names}
root = Path(__file__).resolve().parents[1]
record = dict(
    scope='held-note and rendered-skin diagnostic experiments; sampled motion only; no complete performance acceptance',
    hypothesis='Bounded initialization can restore held contact; finite exposed top constraints prevent false off-key support.',
    ablation='The v2 isolated left-frame-25 pair differs only in feasibility initialization budget (0 versus 100); all other settings and input/source hashes match.',
    caveats=['The two six-pose windows also differ in angular conditioning and candidate retention, so their comparison cannot isolate footprint constraints alone.',
             'The first initialized frozen-neighbor trial was rejected; initialization alone is not a universal guarantee.',
             'Instrument audits include both quad diagonals. Exact self/interhand/body/costume clearance and CCD remain open.',
             'Held-contact checks exclude attack/onset and accidental-note completeness, force dynamics, full repertoire and calibrated human hand-size validation.',
             'All diagnostic failures remain recorded; failed candidates preserve input poses.'],
    source_checkpoint_sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in
        ('blockout/rig/skin_contact.py', 'blockout/skin_contacts.py', 'lab/skin_motion_audit.py')},
    artifact_sha256={name: hashlib.sha256((args.artifacts / (name + '.json')).read_bytes()).hexdigest() for name in names},
    reports=reports)
a, b = (reports['paired_holds_v2_L25_' + suffix] for suffix in ('baseline', 'seeded'))
assert a['source_sha256'] == b['source_sha256'] and a['code_sha256'] == b['code_sha256'] and a['score_sha256'] == b['score_sha256']
assert a['feasibility_iterations'] == 0 and b['feasibility_iterations'] == 100
args.durable.parent.mkdir(parents=True, exist_ok=True)
args.durable.write_text(json.dumps(record, indent=2) + '\n')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for suffix, color, x in [('baseline', '#b64349', 0), ('seeded', '#267eb8', 1)]:
    result = reports['paired_holds_v2_L25_' + suffix]['records'][0]
    axes[0].scatter([x], [result['final_penetration_mm']], color=color, s=75)
    axes[0].annotate('accepted' if result['accepted'] else 'rejected', (x, result['final_penetration_mm']),
                     xytext=(0, 10), textcoords='offset points', ha='center', color=color)
axes[0].set(xticks=[0, 1], xticklabels=['Original start', 'Bounded start'], ylabel='Candidate penetration (mm)',
            title='Frozen pose, identical physical gates', ylim=(-.5, 7.), xlim=(-.25, 1.25))
for suffix, color, label in [('seed', '#b64349', 'Earlier loose footprint'), ('exposed', '#267eb8', 'Finite exposed top')]:
    motion = reports['skin_motion_' + suffix + '_holds_' + ('score_window' if suffix == 'seed' else 'window')]
    origin = motion['records'][0]['time']
    time = [(f['time'] - origin) * 1000. for f in motion['records']]
    axes[1].plot(time, [len(f['held_score']['unsupported_pitches']) for f in motion['records']], '.-', color=color, label=label)
    axes[2].plot(time, [max(h['maximum_penetration_mm'] for h in f['hands'].values()) for f in motion['records']],
                 '.-', color=color, label=label)
axes[1].set(title='Independent held-score contact check', xlabel='Time from window start (ms)',
            ylabel='Unsupported required pitches per sample', yticks=range(5))
axes[2].set(title='Connected motion still needs a gate', xlabel='Time from window start (ms)', ylabel='Skin penetration (mm)')
for axis in axes:
    axis.grid(alpha=.2)
axes[1].legend(fontsize=8)
fig.suptitle('Held-note support improves; static contact cannot certify natural, clear motion', fontsize=14)
fig.savefig(args.artifacts / 'held_contact_experiment.png', dpi=160)
plt.close(fig)
for name in ('restore_exposed_holds_window', 'skin_motion_exposed_holds_window'):
    report = reports[name]
    print(name, json.dumps({key: value for key, value in report.items() if key not in ('records', 'motion', 'code_sha256', 'score_sha256')}))

"""Line/dot figures for the rendered-surface and temporal ablations."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism'
FILES = ['restore_support_window.json', 'restore_sat_window.json',
         'skin_motion_witness_window.json', 'skin_motion_sat_window.json']
data = {name: json.loads((ART / name).read_text()) for name in FILES}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
labels = ['Clipped-point witnesses', 'Whole-triangle SAT']
counts = [sum(r['accepted'] for r in data[name]['records'])
          for name in FILES[:2]]
axes[0].plot([0, 1], counts, '-o', color='#286f9b', linewidth=2)
axes[0].set(xticks=[0, 1], xticklabels=labels, ylim=(-.2, 6.8), yticks=range(7),
            ylabel='Accepted hand poses / 6', title='Static pose gate')
for i, count in enumerate(counts):
    axes[0].annotate(f'{count}/6', (i, count), xytext=(0, 10), textcoords='offset points', ha='center')
for name, label, style in zip(FILES[2:], labels, ('--', '-')):
    report = data[name]
    times = np.array([record['time'] for record in report['records']])
    depth = np.array([max(hand['maximum_penetration_mm'] for hand in record['hands'].values())
                      for record in report['records']])
    axes[1].plot((times - times[0]) * 1000., depth, style + 'o', markersize=3, label=label)
axes[1].set(xlabel='Time since first corrected pose (ms)', ylabel='Maximum penetration (mm)',
            title='240 Hz motion samples still fail')
axes[1].legend(fontsize=8)
probes = ['overlap', 'debussy', 'wide_chord']
reports = [json.loads((ART / f'skin_audit_v4_{name}_full.json').read_text()) for name in probes]
axes[2].plot(range(3), [r['affected_frames'] for r in reports], 'o', markersize=7, color='#b44040')
axes[2].set(xticks=range(3), xticklabels=['Overlap', 'Debussy', 'Wide chord'], ylim=(0, 96),
            ylabel='Instrument-intersecting frames / 86', title='Complete short saved probes')
for i, report in enumerate(reports):
    axes[2].annotate(f'{report["affected_frames"]}/86', (i, report['affected_frames']),
                     xytext=(0, 10), textcoords='offset points', ha='center')
for axis in axes:
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('Rendered skin constraints improve poses; trajectories and full clips remain unsolved', fontsize=13)
fig.savefig(ART / 'skin_contact_experiment.png', dpi=180)
result = dict(hypothesis='Whole-triangle constraints improve static skin clearance beyond clipped-point witnesses.',
              static_accepted=counts, static_total=6,
              conclusion='SAT accepts six poses, but 18/21 interpolated time samples still collide. Static success is insufficient.',
              full_probe_affected={label: report['affected_frames'] for label, report in zip(probes, reports)},
              scope='Oversized frozen stress probes at hand scale 1; no whole-performance or naturalness claim.',
              source_sha256={name: hashlib.sha256((ART / name).read_bytes()).hexdigest() for name in FILES})
(ART / 'skin_contact_experiment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

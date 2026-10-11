"""Recheck frozen drummer patterns with cross-hand grip collision gates."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.drums import Drummer

ART = ROOT / 'artifacts/realism/drums'
old_roll = json.loads((ART / 'roll_qa.json').read_text())
old_stress = json.loads((ART / 'stress_qa.json').read_text())
demo = json.loads((ART / 'crosshand_demo.json').read_text())
roll = Drummer(old_roll['events'])
roll_result = dict(events=roll.events, qa=roll.audit(np.arange(0., 1.5 + 1/240, 1/240)))
stress = []
for row in old_stress['reports']:
    model = Drummer(row['events'])
    duration = max(e['time'] for e in model.events) + .3
    qa = model.audit(np.arange(0., duration + 1/240, 1/240))
    stress.append(dict(pattern=row['pattern'], events=model.events, previous_qa=row['qa'], qa=qa))
violations = Counter(v for row in stress for v in row['qa']['violations'])
results = dict(schema='drummer-crosshand-contact-experiment/v1',
    hypothesis='Complete cross-hand grip capsules catch interactions omitted by separate hand/stick audits.',
    ablation='Identical frozen stress events and kinematic controller; expanded collision gate. No motion repair claimed.',
    code_sha256=hashlib.sha256((ROOT / 'blockout/drums.py').read_bytes()).hexdigest(),
    input_sha256={name: hashlib.sha256((ART / name).read_bytes()).hexdigest() for name in ('roll_qa.json', 'stress_qa.json', 'crosshand_demo.json')},
    sample_rate_hz=240, demo=dict(events=demo['events'], rig=demo['rig'], kit=demo['kit'], qa=demo['qa']),
    roll=roll_result, stress=stress,
    stress_summary=dict(patterns=len(stress), accepted=sum(row['qa']['passed'] for row in stress),
                        contacts=sum(row['qa']['hits'] for row in stress), violations=dict(violations)),
    conclusion='Demo and roll acceptance is limited to this kinematic proxy scope; retained failures are not performances.',
    scope=demo['qa']['scope'], exclusions=demo['qa']['exclusions'])
destination = ROOT / 'lab/realism_results/drummer_crosshand_20261011.json'
destination.write_text(json.dumps(results, indent=2) + '\n')

# Compare accents and ghost notes using the same single surface and timing.
times = np.linspace(.25, .8, 551)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
for velocity, color, label in ((.15, '#258fac', 'soft'), (.95, '#d4733f', 'accent')):
    model = Drummer([dict(time=.5, drum='snare', hand='R', velocity=velocity)])
    poses = [model.pose('R', float(t)) for t in times]
    axes[0].plot(times, [1000*(p['tip'][2]-.10-model.rig.tip_radius) for p in poses], color=color, label=label)
    axes[1].plot(times, [p['pressure'] for p in poses], color=color, label=label)
for ax in axes:
    ax.axvline(.5, color='#89939e', lw=.8, linestyle=':')
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=.2)
    ax.legend()
axes[0].set_ylabel('Tip clearance above snare (mm)')
axes[1].set_ylabel('Grip activation (dimensionless)')
fig.suptitle('Drummer: accent/soft stroke and retained grip — kinematic proxies, no force validation')
fig.savefig(ART / 'crosshand_stroke_control.png', dpi=150)
print(json.dumps(dict(demo=demo['qa'], roll=roll_result['qa'], stress=results['stress_summary']), indent=2))

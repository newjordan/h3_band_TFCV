"""Line/dot evidence for the four-way controller comparison."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    report = json.loads((ROOT / 'lab/realism_results/composure_20261011.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    colours = dict(baseline='#777777', free='#2271b2', wrist='#d87500', both='#16805d')
    labels = dict(baseline='Current', free='Free control', wrist='Wrist carry', both='Both')
    for name, record in report['records'].items():
        animation = json.loads(Path(record['animation']).read_text())
        times = np.arange(len(animation['frames'])) / animation['fps']
        for side, ax in zip(('R', 'L'), axes[0]):
            values = [np.degrees(frame['hands'][side]['angles']['finger3-2']['flex'])
                      if frame['hands'][side]['assigned_keys'][2] is None else np.nan for frame in animation['frames']]
            ax.plot(times, values, label=labels[name], color=colours[name], linewidth=1.5)
            ax.set(title=f'{side}: middle PIP while unused', xlabel='Seconds in passage', ylabel='Joint flexion (degrees)')
        x = list(report['records']).index(name)
        axes[1,0].plot(x, record['original_score_hold_coverage']*100., 'o', color=colours[name])
        for side, marker in (('R','o'), ('L','s')):
            value=report['common_inactive_middle'][name][side]['middle_rest_deviation_rms_deg']
            if value is not None:
                axes[1,1].plot(x, value, marker, color=colours[name])
    axes[0,0].legend(fontsize=8)
    ticks=list(labels.values())
    axes[1,0].set(title='Original-score held-key depth coverage', ylabel='Required samples with key ≥50% down (%)', xticks=range(4), xticklabels=ticks)
    axes[1,1].set(title='Common unused-middle frames (R circles / L squares)', ylabel='Rest-pose deviation RMS (degrees)', xticks=range(4), xticklabels=ticks)
    fig.suptitle('Composure controller: six-second matched trial, measured size 0.7284\nSampled kinematic evidence; actual skin / complete motion remain separate gates', fontsize=12)
    fig.savefig(ROOT / 'artifacts/realism/composure_metrics.png', dpi=130)
    plt.close(fig)


if __name__=='__main__':
    main()

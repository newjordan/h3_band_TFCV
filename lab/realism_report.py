"""Compare matching realism_probe reports; paired lines/dots, never onset alone.

python lab/realism_report.py --pair Debussy before.qa.json after.qa.json \
    --pair Rach before2.qa.json after2.qa.json --out artifacts/realism/comparison
"""
import argparse
import json
from pathlib import Path


def metrics(report):
    n = report['notes']
    pct = lambda value, total: 100 * value / total if total else None
    return {
        'notes': n['count'],
        'key_onset_pct': pct(n['onset_key_down'], n['count']),
        'physical_onset_pct': pct(n['onset_physical_contact'], n['count']),
        'score_hold_contact_pct': pct(n['hold_contact_frames'], n['hold_note_frames']),
        'score_hold_penetration_frames': n['hold_penetrating_frames'],
        'self_overlap_frames': report['separation']['self_hand_frames']['overlap_over_3mm'],
        'interhand_overlap_frames': report['separation']['interhand_frames']['overlap_over_3mm'],
        'joint_limit_violations': report['bounds']['joint_limit_violations'],
        'joint_speed_max_deg_s': report['motion']['joint_speed_deg_s']['max'],
        'reversal_rate': report['motion']['reversal_rate_per_joint_s'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=3, action='append', required=True, metavar=('LABEL', 'BEFORE', 'AFTER'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, before, after in args.pair:
        reports = [json.loads(Path(p).read_text()) for p in (before, after)]
        a, b = reports
        if any(a[k] != b[k] for k in ('start', 'duration', 'fps')) or a['notes']['count'] != b['notes']['count']:
            raise ValueError(f'{label}: comparisons must use the same scoring interval, fps and notes')
        rows.append({'label': label, 'start': a['start'], 'duration': a['duration'],
                     'reports': [str(Path(p).resolve()) for p in (before, after)],
                     'before': metrics(a), 'after': metrics(b),
                     'provenance': [r.get('provenance', {}) for r in reports]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ('key_onset_pct', 'physical_onset_pct', 'score_hold_contact_pct'),
                                 ('Key depressed near onset', 'Pad contact near onset', 'Contact over score hold')):
        for row in rows:
            ax.plot([0, 1], [row['before'][metric], row['after'][metric]], 'o-', label=row['label'])
        ax.set(xticks=[0, 1], xticklabels=['Baseline', 'Candidate'], ylim=(-2, 102), title=title, ylabel='%')
        ax.grid(alpha=.2)
    axes[0].legend(loc='best', fontsize=8)
    fig.suptitle('Piano realism: matched focused passages (not a whole-suite result)')
    fig.savefig(args.out.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps([{k: v for k, v in r.items() if k != 'provenance'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

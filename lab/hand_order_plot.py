"""Show the measured wrist-order correction with line plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def series(path):
    a=json.loads(Path(path).read_text())
    return np.array([f['t'] for f in a['frames']]),np.array([
        (f['hands']['R']['root']['pos'][0]-f['hands']['L']['root']['pos'][0])*1000. for f in a['frames']])


def main():
    report=json.loads((ROOT/'lab/realism_results/hand_order_20261011.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for label,record,color in [('Earlier minute',report['old_minute'],'#9e3939'),
                               ('Ordered minute',report['minute'],'#287d72')]:
        t,gap=series(record['animation']);axes[0].plot(t,gap,label=label,color=color,linewidth=1.25)
    axes[0].set(xlabel='Playback seconds in the minute',ylabel='Right wrist − left wrist (mm)',
                title='The saved minute crosses near its end; fresh ordered roots stay on their own sides')
    for trial,color in [('baseline','#9e3939'),('ordered','#287d72')]:
        t,gap=series(report['matched'][trial]['animation'])
        axes[1].plot(t+287.,gap,label=trial,color=color,linewidth=1.3)
    axes[1].set(xlabel='Performed score seconds',ylabel='Right wrist − left wrist (mm)',
                title='Matched eight-second stress trial: identical score, size and complete solver budgets')
    for ax in axes:
        ax.axhline(0.,color='#292929',linewidth=.9)
        ax.axhline(60.,color='#596b8a',linestyle='--',linewidth=.8,label='60 mm root bound')
        ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.18)
    fig.suptitle('Wrist order is one gate — note support, skin and motion checks still apply',fontsize=13)
    fig.savefig(ART/'hand_order_metrics.png',dpi=145)
    plt.close(fig)


if __name__=='__main__':
    main()

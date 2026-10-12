"""Standalone all-history direction/speed/idle-middle traces, no acceptance claim."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from lab.orientation_trial import TREATMENTS,sha
from blockout.rig.hand import HandRig
from blockout.rig.orientation import split_reference_rotations
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report=json.loads((ROOT/'lab/realism_results/orientation_20261011.json').read_text())
    assert report['status'] in ('controller_complete','complete')
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True,constrained_layout=True)
    colors=['#64646b','#297fc0','#c97827','#358450']
    paths={}
    for name,color in zip(TREATMENTS,colors):
        source=Path(report['records'][name]['source'])
        assert sha(source)==report['records'][name]['sha256']
        paths[str(source)]=sha(source)
        animation=json.loads(source.read_text())
        frames=animation['warmup']['frames']+animation['frames']
        times=np.array([f['t'] for f in frames])+animation['start']
        for column,side in enumerate(('L','R')):
            rig=HandRig(side)
            angles,speeds,middle=[],[],[]
            previous=None
            rest=rig.q_rest[rig.sl[2]+rig.chains[2].i_pip]
            for frame in frames:
                hand=frame['hands'][side]
                r=np.asarray(hand['root']['rot'])
                proper,_=split_reference_rotations(r)
                forward=r @ rig.sk.R0['wrist'].T @ rig.M_rest[1]
                arm=np.asarray(hand['root']['pos'])-hand['elbow']
                angles.append(np.degrees(np.arccos(np.clip(forward @ arm/(np.linalg.norm(forward)*np.linalg.norm(arm)),-1,1))))
                speeds.append(np.nan if previous is None else
                    (Rotation.from_matrix(proper) * Rotation.from_matrix(previous[0]).inv()).magnitude()/(frame['t']-previous[1]))
                quiet=hand['assigned_keys'][2] is None and hand['activation'][2]<.05
                middle.append(np.degrees(hand['angles']['finger3-2']['flex']) if quiet else np.nan)
                previous=(proper,frame['t'])
            axes[0,column].plot(times,angles,color=color,label=name,linewidth=1.2)
            axes[1,column].plot(times,speeds,color=color,linewidth=1.2)
            axes[2,column].plot(times,middle,color=color,linewidth=1.2)
            if name==TREATMENTS[0]:
                axes[2,column].axhline(np.degrees(rest),color='#202020',linestyle=':',linewidth=1.,label='rest reference')
    for column,side in enumerate(('L','R')):
        axes[0,column].set_title(f'{side} wrist / hand')
        axes[0,column].axhline(90.,color='#ad3030',linestyle=':',linewidth=1.)
        axes[0,column].set_ylabel('Hand-forward / forearm angle (degrees)')
        axes[1,column].set_ylabel('Proper wrist rotation speed (rad/s)')
        axes[2,column].set_ylabel('Truly inactive middle PIP (degrees)')
        axes[2,column].set_xlabel('Original performed score time (seconds)')
        for axis in axes[:,column]:
            axis.axvspan(280.9,282.9,color='#bcbcbc',alpha=.15)
            axis.axvline(282.9,color='#777777',linestyle='--',linewidth=.7)
            axis.grid(alpha=.2)
    axes[0,0].legend(ncol=2,fontsize=9)
    fig.suptitle('Physical orientation and quiet-digit control — sampled kinematics; music/skin still required')
    target=ART/'orientation_traces.png'
    fig.savefig(target,dpi=140)
    fig.savefig(ART/'orientation_traces.svg')
    assert all(sha(Path(p))==h for p,h in paths.items())
    (ART/'orientation_plot.json').write_text(json.dumps(dict(status='complete',
        source_unchanged_during_run=True,input_sha256=paths,figure=str(target),sha256=sha(target),
        full_performance_accepted=False),indent=2)+'\n')
    print(str(target),flush=True)


if __name__=='__main__':
    main()

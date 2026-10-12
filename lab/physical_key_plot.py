"""Line/dot evidence for original gates, requested holds and actual skin."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from lab.physical_key_trial import TREATMENTS,sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'legacy':'#427a9a','conservative':'#b47222','conditional':'#42916c'}
LABELS={'legacy':'Current scheduling','conservative':'Conservative holds','conditional':'Conditional pedal releases'}


def main():
    original=ROOT/'lab/realism_results/physical_keys_20261011.json'
    source=ART/'physical_key_controller_result.json'
    if not source.exists():source.write_bytes(original.read_bytes())
    report=json.loads(source.read_text())
    paths=[source,ART/'key_action_recovered_pedal.json',ART/'performance_source_v19/notes.json']
    paths += [ART/f'physical_key_{n}.anim.json' for n in TREATMENTS]
    paths += [ART/f'physical_key_actual_{n}/audit.json' for n in TREATMENTS]
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in paths}
    notes=json.loads((ART/'performance_source_v19/notes.json').read_text())
    animations={n:json.loads((ART/f'physical_key_{n}.anim.json').read_text()) for n in TREATMENTS}
    times=np.array([f['t']+318.9 for f in animations['legacy']['frames']])
    fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    fig.suptitle('Physical key plans: original music stays fixed; requested holds are experimental',fontsize=15)
    active=np.array([sum(on+.04<=t<off-.02 for on,off,*_ in notes) for t in times])
    axes[0,0].step(times,active,where='post',color='black',lw=2,label='Original source gates')
    for name,a in animations.items():
        actions=[e for h in a['scheduled_note_events'].values() for events in h.values() for e in events]
        requested=np.array([sum(e[0]+.04<=t<e[1]-.02 for e in actions) for t in times])
        axes[0,0].step(times,requested,where='post',color=COLORS[name],label=LABELS[name])
        r=report['records'][name]
        coverage=r['proposed_physical_hold_support']['coverage']
        axes[0,1].plot([0,1],[100*r['metrics']['hold_coverage'],100*coverage],marker='o',
            color=COLORS[name],label=LABELS[name])
        for side,style in (('R','-'),('L','--')):
            wrist=[f['hands'][side]['root']['pos'][0]*1000 for f in a['frames']]
            axes[1,0].plot(times,wrist,style,color=COLORS[name],alpha=.85,label=LABELS[name]+' '+side)
            curls=[np.degrees(f['hands'][side]['angles']['finger3-2']['flex']) for f in a['frames']]
            axes[1,1].plot(times,curls,style,color=COLORS[name],alpha=.8,label=LABELS[name]+' '+side)
        audit=json.loads((ART/f'physical_key_actual_{name}/audit.json').read_text())
        for side,style in (('R','-'),('L','--')):
            xs=[316.9+r['frame']/24 for r in audit['records']]
            ys=[r['instrument'][side]['maximum_penetration_mm'] for r in audit['records']]
            axes[2,0].plot(xs,ys,linestyle='None',marker='o' if side=='R' else '^',markersize=7,
                color=COLORS[name],label=LABELS[name]+' '+side)
    pedal=json.loads((ART/'key_action_recovered_pedal.json').read_text())
    from blockout.key_actions import BinaryPedalTimeline
    timeline=BinaryPedalTimeline([(r['performed_time'],r['raw']['value']) for r in pedal['events']
        if r['performed_time'] is not None],*pedal['domain'],pedal['initial_value'])
    dense=np.linspace(316.9,322.9,1500)
    axes[2,1].step(dense,[int(timeline.state(t)) for t in dense],where='post',color='#555555')
    axes[0,0].set(title='Simultaneous source gates and physical action requests',ylabel='Count',xlabel='Performed time (s)')
    axes[0,1].set(title='Solved key/assignment coverage; actual contact checked separately',ylabel='Supported requested samples (%)',
        xticks=[0,1],xticklabels=['Original source gates','Proposed key holds'],ylim=[0,105])
    axes[1,0].set(title='Wrist motion: right solid, left dashed',ylabel='Wrist x (mm)',xlabel='Performed time (s)')
    axes[1,1].set(title='Middle-finger PIP motion (all states)',ylabel='PIP flexion (degrees)',xlabel='Performed time (s)')
    axes[2,0].set(title='Actual Blender skin: initial pose plus two fixed instants',ylabel='Instrument penetration (mm)',xlabel='Performed time (s)')
    axes[2,1].set(title='Recovered binary sustain estimate — no validated foot motion',ylabel='Estimated pedal down',xlabel='Performed time (s)',yticks=[0,1],ylim=[-.15,1.15])
    for ax in axes.flat:ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8,loc='upper right')
    axes[0,1].legend(fontsize=8,loc='upper right')
    axes[1,0].legend(fontsize=7,ncol=2,loc='upper right')
    axes[2,0].legend(fontsize=7,ncol=2,loc='upper left')
    png=ART/'physical_key_traces.png';svg=ART/'physical_key_traces.svg'
    fig.savefig(png,dpi=130);fig.savefig(svg);plt.close(fig)
    assert all(sha(ROOT/p)==h for p,h in fingerprints.items())
    result=dict(status='complete',input_sha256=fingerprints,source_unchanged_during_run=True,
        png=str(png),png_sha256=sha(png),svg=str(svg),svg_sha256=sha(svg),full_performance_accepted=False)
    (ART/'physical_key_plot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',png=str(png))),flush=True)


if __name__=='__main__':main()

"""Plot original music, legacy action proposals and recovered pedal uncertainty."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from blockout.key_actions import BinaryPedalTimeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def counts(events):
    changes=defaultdict(list)
    for event in events:
        changes[event['start']].append((event['pitch'],1))
        changes[event['end']].append((event['pitch'],-1))
    active=Counter();times=[0.];values=[0]
    for time,items in sorted(changes.items()):
        for pitch,delta in items:
            active[pitch]+=delta
            if not active[pitch]:del active[pitch]
        assert all(value>0 for value in active.values())
        times.append(time);values.append(len(active))
    return times,values


def main():
    paths=[ART/'key_action_lineage.json',ART/'key_action_recovered_pedal.json',
        ART/'key_action_controller_result.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    ledger=json.loads(paths[0].read_text());recovered=json.loads(paths[1].read_text());report=json.loads(paths[2].read_text())
    rows=ledger['records'];source=[r['source'] for r in rows];actions=[a for r in rows for a in r['scheduled_actions']]
    first,last=recovered['domain'];events=[(r['performed_time'],r['raw']['value']) for r in recovered['events'] if r['performed_time'] is not None]
    pedal=BinaryPedalTimeline(events,first,last,recovered['initial_value'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(13,9),gridspec_kw={'height_ratios':[1.1,1.8,.7]},layout='constrained')
    old_x,old_y=counts(source);new_x,new_y=counts(actions)
    axes[0].step(old_x,old_y,where='post',color='#777777',label=f'Original MIDI gates (peak {max(old_y)})')
    axes[0].step(new_x,new_y,where='post',color='#1d72aa',label=f'Scheduled hold proposals (peak {max(new_y)})',linewidth=1.2)
    axes[0].axhline(10,color='#9b5f40',linestyle=':',linewidth=1.,label='Ten-pitch diagnostic; no playability proof')
    axes[0].axvspan(first,last,color='#1d72aa',alpha=.045)
    axes[0].set(xlim=(0.,394.),ylim=(0.,24.),ylabel='Distinct gated pitches',xlabel='Performed suite time (seconds)',
        title='Complete source accounting: 2,046 attacks; 2,044 scheduled proposals; 1,478 shortened holds')
    axes[0].legend(loc='upper left',ncols=2,fontsize=8)
    a,b=318.,325.5
    visible=[r for r in rows if r['source']['start']<b and r['source']['end']>a]
    for row in visible:
        n=row['source'];pitch=n['pitch']
        axes[1].plot([max(a,n['start']),min(b,n['end'])],[pitch,pitch],color='#b0b0b0',linewidth=3,zorder=1)
        for action in row['scheduled_actions']:
            if action['start']<b and action['end']>a:
                axes[1].plot([max(a,action['start']),min(b,action['end'])],[pitch,pitch],color='#1d72aa',linewidth=1.7,zorder=2)
        if a<=n['start']<=b:axes[1].plot(n['start'],pitch,'.',color='#333333',markersize=4,zorder=3)
    axes[1].set(xlim=(a,b),ylabel='MIDI pitch',title='Dense descending passage: gray source gates, blue proposed key holds; neither is physical ground truth')
    axes[1].grid(axis='y',alpha=.15)
    xs=[a]+[t for t,_ in events if a<t<b]+[b]
    ys=[int(pedal.state(t)) for t in xs]
    axes[2].step(xs,ys,where='post',color='#8050a0',linewidth=2,label='Recovered binary CC64 estimate')
    resets=report['sampler_damping_comparison']['reset_times']
    for index,t in enumerate(t for t in resets if a<=t<=b):
        axes[2].axvline(t,color='#888888',linestyle=':',linewidth=1.,label='Old audio reset times; separate mechanism' if index==0 else None)
    axes[2].set(xlim=(a,b),ylim=(-.15,1.35),yticks=[0,1],yticklabels=['Up','Down'],xlabel='Performed suite time (seconds)',
        title='Estimated source pedal and unchanged soundtrack reset schedule; note jitter remains separate')
    axes[2].legend(loc='upper left',fontsize=8)
    fig.suptitle('Pedal recovery and action audit — geometry, forces and natural performance remain unaccepted',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(ART/f'key_action_semantics.{suffix}',dpi=140)
    plt.close(fig)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result=dict(status='complete',input_sha256=hashes,source_unchanged_during_run=True,
        full_original_gate_peak=max(old_y),scheduled_proposal_peak=max(new_y),detail_window=[a,b],
        png_sha256=hashlib.sha256((ART/'key_action_semantics.png').read_bytes()).hexdigest(),
        physical_release_or_pedal_ground_truth_claimed=False)
    (ART/'key_action_plot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

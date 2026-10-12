"""Independent source music and requested physical holds; geometry follows."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from lab.physical_key_trial import TREATMENTS,sha
from lab.hand_order_report import metrics
from lab.continuous_plan_report import digit_metrics
from lab.orientation_report import direction_metrics
from blockout.key_actions import interval_conflicts,legacy_sampler_damping
from blockout import sampler
import numpy as np


def requested_holds(animation):
    frames=animation['frames'];times=animation['start']+np.array([f['t'] for f in frames])
    ledger=animation['key_action_ledger']
    actions={}
    for row in ledger['records']:
        for action in row['scheduled_actions']:
            identity=(row['hand'],action['finger'],action['source_id'])
            actions[identity]=action
    records=[]
    for (hand,finger,group),action in actions.items():
        ids=np.flatnonzero((times>=action['start']+.04)&(times<action['end']-.02))
        if not len(ids):continue
        supported=[]
        for i in ids:
            h=frames[i]['hands'][hand]
            supported.append(h['assigned_keys'][finger]==action['pitch'] and h['activation'][finger]>=.7
                and frames[i]['keys'].get(str(action['pitch']),0.)>=.5)
        records.append(dict(group_id=group,hand=hand,finger=finger,pitch=action['pitch'],
            source_ids=action.get('represented_source_ids',[group]),start=action['start'],end=action['end'],
            required=len(ids),supported=sum(supported),first_missing=next((int(i) for i,ok in zip(ids,supported) if not ok),None)))
    required=sum(r['required'] for r in records);supported=sum(r['supported'] for r in records)
    return dict(records=records,required_samples=required,supported_samples=supported,
        coverage=supported/required if required else None,
        scope='Requested physical action, correct assigned finger and solved key depth; actual finite skin support is separately audited')


def main():
    runs=json.loads((ART/'physical_key_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    animations={n:json.loads((ART/f'physical_key_{n}.anim.json').read_text()) for n in TREATMENTS}
    notes=json.loads((ART/'performance_source_v19/notes.json').read_text())
    resets=sampler.auto_pedal(notes)
    common={s:set.intersection(*[set(i for i,f in enumerate(a['frames'])
        if f['hands'][s]['assigned_keys'][2] is None and f['hands'][s]['activation'][2]<.05)
        for a in animations.values()]) for s in ('R','L')}
    records={}
    for name,animation in animations.items():
        path=ART/f'physical_key_{name}.anim.json'
        assert json.loads(path.with_suffix('.notes.json').read_text())==notes
        internal=copy.deepcopy(animation);internal['start']-=2.
        internal['frames']=copy.deepcopy(animation['warmup']['frames']+animation['frames'])
        for f in internal['frames']:f['t']+=2.
        plan=animation['physical_key_plan'];ledger=animation['key_action_ledger']
        late_cuts=[]
        if plan is not None:
            for row in ledger['records']:
                before=row['stages']['key_action_plan']
                after=row['scheduled_actions']
                if len(before)!=1 or len(after)!=1 or any(before[0][k]!=after[0][k] for k in ('start','end','pitch','velocity')):
                    late_cuts.append(row['source']['source_id'])
            assert not late_cuts
        unique={}
        for row in ledger['records']:
            for action in row['scheduled_actions']:
                unique[(row['hand'],action['source_id'])]=action
        conflicts=interval_conflicts({s:dict(action_trace=dict(reach_release=[a for (h,_),a in unique.items() if h==s]))
            for s in ('R','L')})
        if plan is not None:assert len(conflicts)==len(plan['requested_interval_conflicts'])
        in_window=[r for r in conflicts if r['start']<322.9 and r['end']>318.9]
        damping=[]
        for row in ledger['records']:
            for action in row['scheduled_actions']:
                if action['end']==row['source']['end']:continue
                original_damping=legacy_sampler_damping(row['source']['end'],resets)
                candidate_damping=legacy_sampler_damping(action['end'],resets)
                damping.append(dict(source_id=row['source']['source_id'],original_release=row['source']['end'],
                    proposed_release=action['end'],original_damping=original_damping,candidate_damping=candidate_damping,
                    same_damping_time=original_damping==candidate_damping))
        records[name]=dict(source=str(path),sha256=sha(path),metrics=metrics(animation,notes,[48]),
            internal_metrics=metrics(internal,notes,[48,96]),orientation=direction_metrics(animation),
            internal_orientation=direction_metrics(internal),digit_control=digit_metrics(animation),
            common_inactive_middle=digit_metrics(animation,common),
            proposed_physical_hold_support=requested_holds(animation),
            source_actions=ledger['summary'],post_plan_cuts_or_losses=late_cuts,
            legacy_audio_damping=dict(records=damping,same=sum(r['same_damping_time'] for r in damping),
                different=sum(not r['same_damping_time'] for r in damping),
                scope='Hypothetical reuse of physical release in old sampler; actual soundtrack unchanged, no waveform equivalence'),
            requested_interval_conflicts=dict(total=len(conflicts),visible=len(in_window),visible_records=in_window),
            proposed_release_counts=dict(conditional=sum(g['end']<g['conservative_end'] for g in plan['groups']),
                conservative=sum(g['end']==g['conservative_end'] for g in plan['groups']),
                shared=sum(g['shared_strike'] for g in plan['groups'])) if plan is not None else None,
            provenance=animation['experiment_provenance'])
    report=dict(schema='piano-physical-key-experiment-results/v1',status='controller_complete',runs=runs,
        records=records,common_inactive_middle_indices={s:sorted(v) for s,v in common.items()},
        contrasts=dict(legacy_to_conservative='Bundle: exact unison groups, strict slicing/hold persistence and long-held-event selection; full-score allocation recomputed',
            conservative_to_conditional='Release-policy input changes only; allocation/fingering and resulting trajectory are recomputed',
            equal_per_frame_budget=True,equal_total_work=False),
        actual_mesh_audit_pending=True,candidate_exported=False,default_adoption=False,full_performance_accepted=False,
        limitations=['Shared source voices do not reproduce separate jittered audio impulses',
            'Recovered binary pedals are model estimates; offsets do not identify unique key releases',
            'Strict hold conflicts are retained, not resolved by the experiment',
            'Key depth and skeleton assignment are not actual finite skin support',
            'All initial skin states require independent feasibility',
            'Sampled geometry does not prove positive clearance, containment or nonlinear continuous time',
            'Natural digit form, full body/costume, force/action/foot mechanics and drummer remain open'])
    (ROOT/'lab/realism_results/physical_keys_20261011.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:dict(onsets=r['metrics']['onset_passes'],original_hold_coverage=r['metrics']['hold_coverage'],
        requested_hold_coverage=r['proposed_physical_hold_support']['coverage'],
        releases=r['proposed_release_counts'],conflicts=r['requested_interval_conflicts']['total'])
        for n,r in records.items()},indent=2),flush=True)


if __name__=='__main__':main()

"""Recover upstream CC64 with exact score-export provenance; audit release proposals."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from blockout import feeling,midi,piano,sampler,suite
from blockout.key_actions import BinaryPedalTimeline,legacy_sampler_damping
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_tick(time,bars):
    """Apply the existing note writer's bar quantization to one controller time.

    The complete note export must reproduce original bytes before this is used.
    This mirrors the closure in suite.write, preserving rounded bar tick times.
    """
    starts=np.array([b[0] for b in bars]+[bars[-1][0]+bars[-1][1]])
    lengths=[suite.DIV*4*b[2]//b[3] for b in bars]
    origin=np.r_[0,np.cumsum(lengths)]
    i=int(np.clip(np.searchsorted(starts,time,side='right')-1,0,len(bars)-1))
    return int(round(origin[i]+(time-starts[i])/bars[i][1]*lengths[i]))


def event_key(values):
    start,end,pitch,velocity=values
    return round(start,9),round(end,9),int(pitch),int(velocity)


def main():
    began=time.monotonic()
    paths=[p for p in (ROOT/'blockout').rglob('*.py')]
    paths += [Path(__file__),ROOT/'lab/KEY_ACTION_PROTOCOL.md',ROOT/'lab/research_key_action_task.txt']
    paths += [p for p in (ART/'key_action_sources').iterdir() if p.is_file()]
    paths += list((ROOT/'examples/piano').glob('*.mid'))
    paths += [ROOT/'examples/piano/knights_suite.feeling.json',ART/'performance_source_v19/notes.json',
              ART/'performance_source_v19/ks.wav',ART/'key_action_baseline.json']
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in paths}
    fit=json.loads((ART/'key_action_sources/fit.json').read_text())
    raw_path=ART/'key_action_sources/raw.mid'
    raw_notes=midi.read(raw_path)
    available=Counter(event_key([n.start,n.end,n.pitch,n.velocity]) for n in raw_notes)
    fitted=Counter(event_key(n) for n in fit['notes'])
    assert not (fitted-available),'Fitted note evidence must map to the retained original transcription'
    boundaries=fit['bars'];first,last=boundaries[0],boundaries[-1]
    ns=[midi.Note(max(s-first,0),e-first,p,v,1) for s,e,p,v in fit['notes'] if s>=first-.05]
    bars=[(boundaries[i]-first,boundaries[i+1]-boundaries[i],4,4) for i in range(64)]
    chopin=ART/'key_action_rebuilt_chopin.mid'
    suite.write(chopin,suite.resplit(ns),bars)
    original_chopin=ROOT/'examples/piano/chopin_nocturne_cs_minor_posth.mid'
    assert chopin.read_bytes()==original_chopin.read_bytes()
    all_notes,all_bars,marks=suite.knights_suite(ROOT/'examples/piano/clair_de_lune.mid',
        ROOT/'examples/piano/rach_prelude_op3_no2.mid',chopin)
    rebuilt=ART/'key_action_rebuilt_suite.mid'
    suite.write(rebuilt,all_notes,all_bars)
    original_suite=ROOT/'examples/piano/knights_suite.mid'
    assert rebuilt.read_bytes()==original_suite.read_bytes()
    notes=midi.read(original_suite);_,db=midi.beats(original_suite)
    chart=feeling.Chart(str(original_suite.with_suffix('.feeling.json')),db)
    receipt={};performed,timing=feeling.perform(notes,chart,0,timing_receipt=receipt)
    original=[[n.start,n.end,n.pitch,n.velocity,n.track] for n in performed]
    assert original==json.loads((ART/'performance_source_v19/notes.json').read_text())
    (ART/'key_action_timing.json').write_text(json.dumps(receipt,indent=2)+'\n')
    chopin_seconds=midi._parse(chopin)[2];suite_seconds=midi._parse(rebuilt)[2]
    suite_origin=marks[-1][1]
    domain_start=float(timing(suite_seconds(score_tick(suite_origin,all_bars))))
    domain_end=float(timing(suite_seconds(score_tick(suite_origin+sum(b[1] for b in bars),all_bars))))
    events=midi.control_changes(raw_path,64)
    assert len(events)==282 and {(e.track,e.channel) for e in events}=={(1,0)}
    mapped=[];retained=[];initial=0;carry=None
    for identity,event in enumerate(events):
        record=dict(raw_id=identity,raw=asdict(event))
        if event.time<first:
            initial=event.value;carry=identity
            record.update(status='before excerpt; retained as initial-state evidence',performed_time=None)
        elif event.time>last:
            record.update(status='after excerpt; retained without extrapolated foot state',performed_time=None)
        else:
            shifted=event.time-first
            chopin_tick=score_tick(shifted,bars);chopin_time=chopin_seconds(chopin_tick)
            input_time=chopin_time+suite_origin
            suite_tick=score_tick(input_time,all_bars);suite_time=suite_seconds(suite_tick)
            performance_time=float(timing(suite_time))
            record.update(status='inside recovered Chopin domain',crop_shifted_time=shifted,
                chopin_tick=chopin_tick,chopin_score_time=chopin_time,suite_input_time=input_time,
                suite_tick=suite_tick,suite_score_time=suite_time,performed_time=performance_time)
            retained.append((performance_time,event.value))
        mapped.append(record)
    assert len(mapped)==len(events) and [r['raw_id'] for r in mapped]==list(range(len(events)))
    pedal=BinaryPedalTimeline(retained,domain_start,domain_end,initial)
    recovered=dict(schema='piano-estimated-pedal-recovery/v1',domain=[domain_start,domain_end],
        initial_value=initial,initial_evidence_raw_id=carry,final_value=retained[-1][1],
        events=mapped,source_sha256=sha(raw_path),source_recording_crop=[first,last],
        model_estimate=True,ground_truth=False,physical_foot_motion_validated=False,
        time_mapping='recording crop; Chopin bar/tick export; suite excerpt offset and bar/tick export; shared performance time map',
        note_jitter_applied_to_pedals=False,
        earlier_pieces='unknown; no estimated pedal stream supplied',
        terminal_release_fabricated=False)
    (ART/'key_action_recovered_pedal.json').write_text(json.dumps(recovered,indent=2)+'\n')
    piano.REACH_SPLIT=True
    score=piano.compile_score(performed)
    baseline=json.loads((ART/'key_action_baseline.json').read_text())
    for side,h in score.hands.items():
        assert baseline['hands'][side]['events']=={str(k):v for k,v in h['ev'].items()}
        assert baseline['hands'][side]['fingering']==[[[n.start,n.end,n.pitch,n.velocity,n.track,f] for n,f in row] for row in h['fing']]
    assert baseline['ownership']==score.ownership and baseline['centre_x']==score.centre_x
    ledger=score.action_ledger
    (ART/'key_action_lineage.json').write_text(json.dumps(ledger,indent=2)+'\n')
    resets=sampler.auto_pedal(original) # archived render command supplies no downbeats
    coverage=[];audio=[];ambiguities={str(ms):[] for ms in (0,10,30,50)}
    timing_rows={r['output_id']:r for r in receipt['records']}
    for row in ledger['records']:
        source=row['source'];identity=source['source_id'];linked=timing_rows[identity]
        assert linked['output_note']==[source[k] for k in ('start','end','pitch','velocity','track')]
        is_chopin=linked['input_note'][0]>=suite_origin-1e-9
        if is_chopin and domain_start<=source['start']<=source['end']<=domain_end:
            for ms in (0,10,30,50):
                bounds=pedal.release_hypotheses(source['start'],source['end'],ms/1000)
                ambiguities[str(ms)].append(dict(source_id=identity,candidate_intervals=bounds))
        for action in row['scheduled_actions']:
            a,b=action['end'],source['end']
            if a>=b:continue
            comparison=dict(source_id=identity,hand=row['hand'],finger=action['finger'],release=a,source_offset=b)
            if is_chopin and domain_start<=a<=b<=domain_end:
                comparison.update(status='within estimated pedal domain',**pedal.release_coverage(a,b))
            else:
                comparison.update(status='unknown outside recovered pedal query domain',covered=None)
            coverage.append(comparison)
            old,new=legacy_sampler_damping(b,resets),legacy_sampler_damping(a,resets)
            audio.append(dict(source_id=identity,source_damping=old,candidate_damping=new,
                difference_seconds=new-old,same_damping_time=old==new,
                scope='same damping parameter in existing sampler; not measured acoustic/performance equivalence'))
    stage_cuts=Counter()
    for row in ledger['records']:
        for change in row['changes']:
            if len(change['before'])==len(change['after'])==1 and change['after'][0]['end']<change['before'][0]['end']:
                stage_cuts[change['stage']]+=1
    report=dict(schema='piano-key-action-study/v1',status='controller_and_source_complete',
        input_sha256=fingerprints,code_unchanged_during_run=all(sha(ROOT/p)==h for p,h in fingerprints.items()),
        source_recovery=dict(raw_note_count=len(raw_notes),fitted_note_count=len(fit['notes']),
            all_fitted_notes_identified_in_raw=True,raw_cc64_count=len(events),mapped_count=len(retained),
            all_raw_cc_accounted=True,chopin_bytes_identical=True,suite_bytes_identical=True,
            chopin_sha256=sha(chopin),suite_sha256=sha(rebuilt),domain=[domain_start,domain_end],
            initial_value=initial,initial_evidence_raw_id=carry,final_value=retained[-1][1]),
        source_actions=ledger['summary'],stage_release_cuts=dict(stage_cuts),
        legacy_control=dict(fingering_identical=True,scheduled_events_identical=True,
            ownership_identical=True,body_centre_identical=True,original_performed_notes_identical=True),
        estimated_pedal_release_audit=dict(records=coverage,
            covered=sum(r['covered'] is True for r in coverage),failed=sum(r['covered'] is False for r in coverage),
            unknown=sum(r['covered'] is None for r in coverage),ground_truth=False),
        sampler_damping_comparison=dict(records=audio,reset_times=resets.tolist(),
            same=sum(r['same_damping_time'] for r in audio),different=sum(not r['same_damping_time'] for r in audio),
            waveform_equivalence_verified=False,soundtrack_replaced=False),
        offset_ambiguity=dict(records=ambiguities,
            matched_notes={ms:sum(bool(r['candidate_intervals']) for r in rows) for ms,rows in ambiguities.items()},
            chosen_physical_releases=0,timing_accuracy_claimed=False),
        native_graph_pending=True,default_controller_changed=False,
        source_recording='Frank Levy recording feeds model; preview soundtrack is separately synthesized Salamander audio',
        physical_key_actions_validated=False,full_performance_accepted=False,seconds=time.monotonic()-began)
    assert report['code_unchanged_during_run']
    target=ROOT/'lab/realism_results/key_actions_20261011.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],recovery=report['source_recovery'],
        missing=ledger['summary']['missing_source_ids'],shortened=len(ledger['summary']['shortened_source_ids']),
        stage_cuts=dict(stage_cuts),pedal_counts={k:report['estimated_pedal_release_audit'][k] for k in ('covered','failed','unknown')},
        sampler_counts={k:report['sampler_damping_comparison'][k] for k in ('same','different')},
        ambiguous_offsets=report['offset_ambiguity']['matched_notes'])),flush=True)


if __name__=='__main__':main()

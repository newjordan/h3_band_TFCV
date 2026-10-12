"""Three frozen full-budget source-voice/key-hold trajectory treatments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
TREATMENTS=('legacy','conservative','conditional')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return sorted([p for p in (ROOT/'blockout').rglob('*')
        if p.is_file() and p.suffix in ('.py','.json','.npz','.obj')]
        +[Path(__file__),ROOT/'lab/run_physical_keys.py',ROOT/'lab/PHYSICAL_KEY_PLAN_PROTOCOL.md',
          ROOT/'lab/research_physical_key_plan_task.txt',ROOT/'examples/piano/knights_suite.mid',
          ROOT/'examples/piano/knights_suite.feeling.json',ART/'performance_source_v19/notes.json',
          ART/'performance_source_v19/ks.wav',ART/'key_action_timing.json',
          ART/'key_action_recovered_pedal.json',ART/'key_action_lineage.json',ART/'key_action_baseline.json'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('treatment',choices=TREATMENTS)
    args=ap.parse_args()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['HAND_SCALE']='1'
    overrides=dict(PRESS_AXES=[20.,1.,20.],PRESS_AXES_BLACK=[20.,20.,20.],
        REPAIR_ITERS=2,PRESS_OVERSHOOT=.003,REACH_SPLIT=True,NO_HAND_CROSS=True,
        FREE_CONTROL=False,WRIST_CARRY=False,WRIST_SO3_HISTORY=False,WRIST_SIGNED_ANGLES=False)
    os.environ['PIANO_OVR']=json.dumps(overrides)
    from blockout import midi,feeling,piano
    from blockout.key_actions import propose_key_plan,BinaryPedalTimeline
    from blockout.performance_windows import export_window,rejoin_windows
    import numpy as np
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in inputs()}
    path=ROOT/'examples/piano/knights_suite.mid'
    original_input=midi.read(path)
    tracks,_,_=midi._parse(path)
    channel_attacks=[(i,status&15) for i,tr in enumerate(tracks) for _,status,data in tr
                     if isinstance(status,int) and status&240==144 and data[1]>0]
    assert len(channel_attacks)==len(original_input) and {ch for _,ch in channel_attacks}=={0}
    channels={i:0 for i in range(len(original_input))}
    beats,downbeats=midi.beats(path,1.)
    chart=feeling.Chart(str(path.with_suffix('.feeling.json')),downbeats)
    receipt={};notes,timing=feeling.perform(original_input,chart,0,timing_receipt=receipt)
    assert receipt==json.loads((ART/'key_action_timing.json').read_text())
    original=[[n.start,n.end,n.pitch,n.velocity,n.track] for n in notes]
    assert original==json.loads((ART/'performance_source_v19/notes.json').read_text())
    baseline=json.loads((ART/'key_action_lineage.json').read_text())
    recovered=json.loads((ART/'key_action_recovered_pedal.json').read_text())
    pedal=BinaryPedalTimeline([(r['performed_time'],r['raw']['value']) for r in recovered['events']
        if r['performed_time'] is not None],*recovered['domain'],recovered['initial_value'])
    proposal=None if args.treatment=='legacy' else propose_key_plan(notes,receipt,channels,baseline,pedal,
        conditional_releases=args.treatment=='conditional',offset_tolerance=.030)
    began=time.monotonic()
    score=piano.compile_score(notes,key_action_plan=proposal)
    if args.treatment=='legacy':
        old=json.loads((ART/'key_action_baseline.json').read_text())
        assert old['ownership']==score.ownership and old['centre_x']==score.centre_x
        for side,h in score.hands.items():
            assert old['hands'][side]['events']=={str(f):ev for f,ev in h['ev'].items()}
            assert old['hands'][side]['fingering']==[[[n.start,n.end,n.pitch,n.velocity,n.track,f] for n,f in row] for row in h['fing']]
    plan=piano.prepare_plan(compiled_score=score,start=318.9,dur=4.,fps=24,speed=1.,
        physical_warmup=True,beats=[float(timing(b)) for b in beats],
        downbeats=[float(timing(b)) for b in downbeats],
        schedule=[(float(timing(t)),head) for t,head in chart.heads],
        energy_fn=lambda t:float(np.clip(1.4*chart.at('strength',np.interp(t,timing.tp,timing.ts)),0,1)),seed=0)
    animation=piano.solve_plan(plan)
    parts=[export_window(animation,0,48),export_window(animation,48,96)]
    assert rejoin_windows(parts)['frames']==animation['frames']
    assert len(animation['frames'])==96 and len(animation['warmup']['frames'])==48 and animation['hand_scale']==1.
    assert original==[[n.start,n.end,n.pitch,n.velocity,n.track] for n in notes]
    assert all(sha(ROOT/p)==h for p,h in fingerprints.items())
    animation['geometry']=dict(key_shape='notched',scope='identical production renderer and independent audit')
    animation['experiment_provenance']=dict(schema='piano-physical-key-experiment/v1',
        treatment=args.treatment,overrides=overrides,input_sha256=fingerprints,
        original_score_unchanged=True,original_soundtrack_unchanged=True,code_unchanged_during_run=True,
        verified_input_channels=dict(note_on_count=len(channel_attacks),channels=[0],
            source_sha256=sha(path),timing_receipt_identical=True),seconds=time.monotonic()-began,
        legacy_scheduling_identical=args.treatment=='legacy',candidate_exported=False,
        diagnostic_motion_exported=True,full_performance_accepted=False,
        roundtrip=dict(passed=True,source_poses_keys_body_timestamps_identical=True),
        maximum_per_frame_hand_budget_unchanged=True,equal_total_work=False)
    target=ART/f'physical_key_{args.treatment}.anim.json'
    target.write_text(json.dumps(animation,allow_nan=False)+'\n')
    target.with_suffix('.notes.json').write_text(json.dumps(original)+'\n')
    print(json.dumps(dict(status='complete',treatment=args.treatment,source=str(target),sha256=sha(target),
        seconds=time.monotonic()-began,actions=score.action_ledger['summary'],
        requested_interval_conflicts=len(score.physical_key_plan['requested_interval_conflicts']) if proposal else None)),flush=True)


if __name__=='__main__':main()

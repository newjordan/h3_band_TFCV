"""Frozen full-budget 2x2 absolute-orientation/forearm experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'
sys.path.insert(0, str(ROOT))
TREATMENTS = ('legacy', 'so3', 'forearm', 'combined')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return sorted([p for p in (ROOT/'blockout').rglob('*')
        if p.is_file() and p.suffix in ('.py', '.json', '.npz', '.obj')]
        + [Path(__file__), ROOT/'lab/ORIENTATION_PROTOCOL.md',
           ROOT/'lab/research_orientation_task.txt', ROOT/'examples/piano/knights_suite.mid',
           ROOT/'examples/piano/knights_suite.feeling.json', ART/'performance_source_v19/notes.json'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('treatment', choices=TREATMENTS)
    args = ap.parse_args()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HAND_SCALE'] = '1'
    overrides = dict(PRESS_AXES=[20.,1.,20.], PRESS_AXES_BLACK=[20.,20.,20.],
        REPAIR_ITERS=2, PRESS_OVERSHOOT=.003, REACH_SPLIT=True, NO_HAND_CROSS=True,
        FREE_CONTROL=False, WRIST_CARRY=False,
        WRIST_SO3_HISTORY=args.treatment in ('so3','combined'),
        WRIST_SIGNED_ANGLES=args.treatment in ('forearm','combined'), WRIST_SO3_EVALUATIONS=30)
    os.environ['PIANO_OVR'] = json.dumps(overrides)
    from blockout import midi, feeling, piano
    from blockout.performance_windows import export_window, rejoin_windows
    import numpy as np
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs()}
    path = ROOT/'examples/piano/knights_suite.mid'
    notes = midi.read(path)
    beats, downbeats = midi.beats(path, 1.)
    chart = feeling.Chart(str(path.with_suffix('.feeling.json')), downbeats)
    notes, timing = feeling.perform(notes, chart, 0)
    original = [[n.start,n.end,n.pitch,n.velocity,n.track] for n in notes]
    assert original == json.loads((ART/'performance_source_v19/notes.json').read_text())
    began = time.monotonic()
    score = piano.compile_score(notes)
    plan = piano.prepare_plan(compiled_score=score,start=282.9,dur=4.,fps=24,speed=1.,
        physical_warmup=True,beats=[float(timing(b)) for b in beats],
        downbeats=[float(timing(b)) for b in downbeats],
        schedule=[(float(timing(t)),head) for t,head in chart.heads],
        energy_fn=lambda t: float(np.clip(1.4*chart.at('strength',np.interp(t,timing.tp,timing.ts)),0,1)),seed=0)
    animation = piano.solve_plan(plan)
    parts = [export_window(animation,0,48),export_window(animation,48,96)]
    assert rejoin_windows(parts)['frames'] == animation['frames']
    assert len(animation['frames'])==96 and len(animation['warmup']['frames'])==48
    assert animation['hand_scale']==1.
    assert original == [[n.start,n.end,n.pitch,n.velocity,n.track] for n in notes]
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    baseline = json.loads((ART/'continuous_plan_warm.anim.json').read_text()) if args.treatment=='legacy' else None
    reproduction = None
    if baseline is not None:
        reproduction = dict(source=str(ART/'continuous_plan_warm.anim.json'),
            sha256=sha(ART/'continuous_plan_warm.anim.json'),
            visible_frames_identical=animation['frames']==baseline['frames'],
            physical_warmup_frames_identical=animation['warmup']['frames']==baseline['warmup']['frames'])
        assert reproduction['visible_frames_identical'] and reproduction['physical_warmup_frames_identical']
    animation['geometry'] = dict(key_shape='notched',scope='identical production renderer and independent audit')
    animation['experiment_provenance'] = dict(schema='piano-wrist-orientation/v1',treatment=args.treatment,
        overrides=overrides,input_sha256=hashes,original_score_unchanged=True,
        code_unchanged_during_run=True,seconds=time.monotonic()-began,
        source_baseline_reproduction=reproduction,candidate_exported=False,
        diagnostic_motion_exported=True,full_performance_accepted=False,
        roundtrip=dict(passed=True,source_poses_keys_body_timestamps_identical=True),
        maximum_per_frame_hand_budget_unchanged=True,equal_total_work=False,
        extra_offline_so3_fit=bool(overrides['WRIST_SO3_HISTORY']))
    target=ART/f'orientation_{args.treatment}.anim.json'
    target.write_text(json.dumps(animation,allow_nan=False)+'\n')
    target.with_suffix('.notes.json').write_text(json.dumps(original)+'\n')
    print(json.dumps(dict(status='complete',treatment=args.treatment,source=str(target),
        sha256=sha(target),seconds=time.monotonic()-began,reproduction=reproduction)),flush=True)


if __name__=='__main__':
    main()

"""Frozen full-budget continuity ablation, original performed score unchanged."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return sorted([p for p in (ROOT/'blockout').rglob('*')
        if p.is_file() and p.suffix in ('.py', '.json', '.npz', '.obj')]
        + [ROOT/'lab/continuous_plan_trial.py', ROOT/'examples/piano/knights_suite.mid',
           ROOT/'examples/piano/knights_suite.feeling.json',
           ROOT/'artifacts/realism/performance_source_v19/notes.json'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('treatment', choices=['windowed', 'coherent', 'warm', 'composed'])
    args = ap.parse_args()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HAND_SCALE'] = '1'
    composed = args.treatment == 'composed'
    overrides = dict(PRESS_AXES=[20., 1., 20.], PRESS_AXES_BLACK=[20., 20., 20.],
        REPAIR_ITERS=2, PRESS_OVERSHOOT=.003, REACH_SPLIT=True, NO_HAND_CROSS=True,
        FREE_CONTROL=composed, WRIST_CARRY=composed)
    os.environ['PIANO_OVR'] = json.dumps(overrides)
    from blockout import midi, feeling, piano
    from blockout.performance_windows import export_window, rejoin_windows
    import numpy as np
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in inputs()}
    source = ROOT/'examples/piano/knights_suite.mid'
    notes = midi.read(source)
    beats, downbeats = midi.beats(source, 1.)
    chart = feeling.Chart(str(source.with_suffix('.feeling.json')), downbeats)
    notes, timing = feeling.perform(notes, chart, 0)
    original = [[n.start, n.end, n.pitch, n.velocity, n.track] for n in notes]
    assert original == json.loads((ROOT/'artifacts/realism/performance_source_v19/notes.json').read_text())
    kwargs = dict(fps=24, speed=1., beats=[float(timing(b)) for b in beats],
        downbeats=[float(timing(b)) for b in downbeats],
        schedule=[(float(timing(t)), head) for t, head in chart.heads],
        energy_fn=lambda t: float(np.clip(1.4*chart.at('strength', np.interp(t, timing.tp, timing.ts)), 0, 1)), seed=0)
    began = time.monotonic()
    if args.treatment == 'windowed':
        parts = [piano.animate(notes, start=start, dur=2., **kwargs) for start in (282.9, 284.9)]
        animation = copy.deepcopy(parts[0])
        animation['frames'] = []
        for i, part in enumerate(parts):
            for frame in part['frames']:
                frame = copy.deepcopy(frame)
                frame['t'] += 2.*i
                animation['frames'].append(frame)
        animation['qa'] = {'scope': 'two independent solves; original diagnostics retained in experiment parts'}
        animation['experiment_parts'] = [{k: v for k, v in part.items() if k != 'frames'} for part in parts]
        animation['continuity']['internal_frames'] = sum(p['continuity']['internal_frames'] for p in parts)
        animation['continuity']['history'] = 'two independent physical histories; no join blending'
        roundtrip = None
    else:
        score = piano.compile_score(notes)
        plan = piano.prepare_plan(compiled_score=score, start=282.9, dur=4.,
            physical_warmup=args.treatment in ('warm', 'composed'), **kwargs)
        animation = piano.solve_plan(plan)
        parts = [export_window(animation, 0, 48), export_window(animation, 48, 96)]
        rejoined = rejoin_windows(parts)
        assert rejoined['frames'] == animation['frames']
        roundtrip = dict(passed=True, original_pose_key_body_and_time_identical=True,
            render_windows=2, solve_calls=1)
        for i, part in enumerate(parts):
            path = ROOT/f'artifacts/realism/continuous_plan_{args.treatment}.part{i}.anim.json'
            path.write_text(json.dumps(part, allow_nan=False)+'\n')
    assert len(animation['frames']) == 96
    assert animation['hand_scale'] == 1.
    assert original == [[n.start, n.end, n.pitch, n.velocity, n.track] for n in notes]
    assert all(sha(ROOT/p) == h for p, h in fingerprints.items())
    animation['geometry'] = dict(key_shape='notched', scope='identical renderer and independent audit')
    animation['experiment_provenance'] = dict(schema='piano-continuous-plan/v1', treatment=args.treatment,
        overrides=overrides, input_sha256=fingerprints, original_score_unchanged=True,
        code_unchanged_during_run=True, roundtrip=roundtrip, seconds=time.monotonic()-began,
        candidate_exported=False, diagnostic_motion_exported=True, full_performance_accepted=False,
        warmup_extra_solved_frames=48 if args.treatment in ('warm','composed') else 0,
        comparison_scope='same per-frame budget; full-score context and physical warmup change total work')
    path = ROOT/f'artifacts/realism/continuous_plan_{args.treatment}.anim.json'
    path.write_text(json.dumps(animation, allow_nan=False)+'\n')
    path.with_suffix('.notes.json').write_text(json.dumps(original)+'\n')
    print(json.dumps(dict(status='complete', treatment=args.treatment, source=str(path), sha256=sha(path),
        seconds=time.monotonic()-began, continuity=animation['continuity'])), flush=True)


if __name__ == '__main__':
    main()

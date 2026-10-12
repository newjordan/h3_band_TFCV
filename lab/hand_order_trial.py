"""Full-budget score-preserving trial of the requested no-crossing style."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--start', type=float, default=287.)
    parser.add_argument('--seconds', type=float, default=8.)
    args = parser.parse_args()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HAND_SCALE'] = '1'
    overrides = dict(PRESS_AXES=[20., 1., 20.], PRESS_AXES_BLACK=[20., 20., 20.],
        REPAIR_ITERS=2, PRESS_OVERSHOOT=.003, REACH_SPLIT=True,
        FREE_CONTROL=False, WRIST_CARRY=False, NO_HAND_CROSS=args.ordered)
    os.environ['PIANO_OVR'] = json.dumps(overrides)
    from blockout import midi, feeling, piano
    import numpy as np
    source = ROOT / 'examples/piano/knights_suite.mid'
    feeling_file = source.with_suffix('.feeling.json')
    code = list((ROOT / 'blockout').rglob('*.py')) + [ROOT / 'lab/hand_order_trial.py', source, feeling_file]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in code}
    notes = midi.read(source)
    beats, downbeats = midi.beats(source, 1.)
    chart = feeling.Chart(str(feeling_file), downbeats)
    notes, timing = feeling.perform(notes, chart, 0)
    original = [[n.start, n.end, n.pitch, n.velocity, n.track] for n in notes]
    durable = json.loads((ROOT / 'artifacts/realism/performance_source_v19/notes.json').read_text())
    assert original == durable
    schedule = [(float(timing(t)), head) for t, head in chart.heads]
    energy = lambda t: float(np.clip(1.4 * chart.at('strength', np.interp(t, timing.tp, timing.ts)), 0, 1))
    began = time.monotonic()
    animation = piano.animate(notes, 24, args.start, args.seconds, 1.,
        [float(timing(b)) for b in beats], [float(timing(b)) for b in downbeats], schedule, energy, 0)
    assert len(animation['frames']) == round(24 * args.seconds)
    assert animation['hand_scale'] == 1.
    assert original == [[n.start, n.end, n.pitch, n.velocity, n.track] for n in notes]
    assert all(sha(ROOT / path) == value for path, value in fingerprints.items())
    animation['geometry'] = dict(key_shape='notched', scope='identical renderer and independent audit')
    animation['experiment_provenance'] = dict(schema='piano-hand-order/v1', overrides=overrides,
        input_sha256=fingerprints, original_score_unchanged=True, code_unchanged_during_run=True,
        candidate_exported=False, diagnostic_motion_exported=True, full_performance_accepted=False,
        seconds=time.monotonic()-began,
        hypothesis='Persistent ordered note ownership, restrained first approach and bounded wrists prevent crossing.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(animation) + '\n')
    args.output.with_suffix('.notes.json').write_text(json.dumps(original) + '\n')
    print(json.dumps(dict(animation=str(args.output), sha256=sha(args.output),
        seconds=time.monotonic()-began, hand_order=animation['hand_order']['audit'], qa=animation['qa'])), flush=True)


if __name__ == '__main__':
    main()

"""Run frozen controller jobs on CPU; retain every terminal result."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism'


def main():
    code = list((ROOT/'blockout').rglob('*.py')) + [ROOT/'lab/hand_order_trial.py']
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    jobs = [('baseline', 287., 8., False), ('ordered', 287., 8., True)]
    jobs += [(f'minute_{i}', 239.9 + 7.5*i, 7.5, True) for i in range(8)]
    running, records = [], []
    began = time.monotonic()
    for label, start, seconds, ordered in jobs:
        path = ART / f'hand_order_{label}.anim.json'
        command = [sys.executable, str(ROOT/'lab/hand_order_trial.py'), str(path),
                   '--start', str(start), '--seconds', str(seconds)]
        if ordered:
            command.append('--ordered')
        log = open(ART/f'hand_order_{label}.log', 'w')
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        running.append((label, start, seconds, ordered, path, process, log))
        print(json.dumps(dict(started=label, pid=process.pid)), flush=True)
    for label, start, seconds, ordered, path, process, log in running:
        status = process.wait()
        log.close()
        record = dict(trial=label, start=start, duration_seconds=seconds, ordered=ordered,
                      exit_code=status, animation=str(path), output_exists=path.exists())
        records.append(record)
        print(json.dumps(record), flush=True)
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in fingerprints.items())
    report = dict(status='complete' if all(r['exit_code']==0 and r['output_exists'] for r in records) else 'failed',
                  seconds=time.monotonic()-began, records=records, input_sha256=fingerprints,
                  code_unchanged_during_run=unchanged, gpu_used=False)
    (ART/'hand_order_runs.json').write_text(json.dumps(report,indent=2)+'\n')
    assert unchanged
    assert report['status']=='complete'


if __name__=='__main__':
    main()

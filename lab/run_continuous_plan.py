"""Launch four frozen CPU trials and retain each terminal outcome."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'lab'))
from continuous_plan_trial import inputs, sha


def main():
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs()}
    hashes['lab/run_continuous_plan.py'] = sha(Path(__file__))
    jobs, records = [], []
    began = time.monotonic()
    for name in ('windowed', 'coherent', 'warm', 'composed'):
        log = open(ROOT/f'artifacts/realism/continuous_plan_{name}.log', 'w')
        process = subprocess.Popen([sys.executable, str(ROOT/'lab/continuous_plan_trial.py'), name],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((name, process, log))
        print(json.dumps(dict(started=name, pid=process.pid)), flush=True)
    for name, process, log in jobs:
        status = process.wait()
        log.close()
        path = ROOT/f'artifacts/realism/continuous_plan_{name}.anim.json'
        record = dict(treatment=name, exit_code=status, output_exists=path.exists(),
            source=str(path), sha256=sha(path) if path.exists() else None)
        records.append(record)
        print(json.dumps(record), flush=True)
    unchanged = all(sha(ROOT/p) == h for p, h in hashes.items())
    report = dict(status='complete' if all(r['exit_code']==0 and r['output_exists'] for r in records) else 'failed',
        records=records, input_sha256=hashes, code_unchanged_during_run=unchanged,
        seconds=time.monotonic()-began, gpu_used=False, full_performance_accepted=False)
    (ROOT/'artifacts/realism/continuous_plan_runs.json').write_text(json.dumps(report, indent=2)+'\n')
    assert unchanged and report['status']=='complete'


if __name__ == '__main__':
    main()

"""Run all four CPU-only treatments from the same frozen source state."""
import json
from pathlib import Path
import subprocess
import sys
import time

from orientation_trial import ROOT,ART,TREATMENTS,inputs,sha


def main():
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs()}
    hashes['lab/run_orientation.py']=sha(Path(__file__))
    jobs,records=[],[]
    began=time.monotonic()
    for name in TREATMENTS:
        log=open(ART/f'orientation_{name}.log','w')
        process=subprocess.Popen([sys.executable,str(ROOT/'lab/orientation_trial.py'),name],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,process,log))
        print(json.dumps(dict(started=name,pid=process.pid)),flush=True)
    for name,process,log in jobs:
        status=process.wait()
        log.close()
        path=ART/f'orientation_{name}.anim.json'
        record=dict(treatment=name,exit_code=status,output_exists=path.exists(),
            source=str(path),sha256=sha(path) if path.exists() else None)
        records.append(record)
        print(json.dumps(record),flush=True)
    unchanged=all(sha(ROOT/p)==h for p,h in hashes.items())
    report=dict(status='complete' if all(r['exit_code']==0 and r['output_exists'] for r in records) else 'failed',
        records=records,input_sha256=hashes,code_unchanged_during_run=unchanged,
        seconds=time.monotonic()-began,gpu_used=False,full_performance_accepted=False)
    (ART/'orientation_runs.json').write_text(json.dumps(report,indent=2)+'\n')
    assert unchanged and report['status']=='complete'


if __name__=='__main__':
    main()

"""Finish independent skin diagnostics and render after frozen solves complete."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def wait_json(path):
    while True:
        try:
            result=json.loads(path.read_text())
            return result
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(10)


def skin(trial):
    source=ART/f'hand_order_{trial}.anim.json'
    animation=wait_json(source)
    assert animation['experiment_provenance']['code_unchanged_during_run']
    directory=ART/f'hand_order_actual_{trial}'
    directory.mkdir(exist_ok=True)
    environment={**os.environ,'LP_NUM_THREADS':'2','LIBGL_ALWAYS_SOFTWARE':'1','MESA_SHADER_CACHE_DISABLE':'true',
                 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    with open(directory/'snapshot.log','w') as log:
        subprocess.run(['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'lab/hand_order_snapshot.py'),
            '--',str(source),str(directory),'36,72'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
    assert (directory/'snapshots.json').exists()
    assert '"status": "complete"' in (directory/'snapshot.log').read_text()
    with open(directory/'audit.log','w') as log:
        subprocess.run([sys.executable,str(ROOT/'lab/skin_deformation_audit.py'),str(directory/'snapshots.json'),
            str(directory/'audit.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
    audit=json.loads((directory/'audit.json').read_text())
    assert audit['status']=='complete'
    print(json.dumps(dict(skin_complete=trial,records=len(audit['records']))),flush=True)


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(skin,t) for t in ('baseline','ordered')]
        result=wait_json(ART/'hand_order_runs.json')
        assert result['status']=='complete' and result['code_unchanged_during_run']
        for script in ['hand_order_report.py','hand_order_plot.py','hand_order_render.py']:
            print(json.dumps(dict(starting=script)),flush=True)
            subprocess.run([sys.executable,str(ROOT/'lab'/script)],cwd=ROOT,check=True)
        for future in futures:
            future.result()
    print('Complete diagnostics and render; visual review and sealing still required',flush=True)


if __name__=='__main__':
    main()

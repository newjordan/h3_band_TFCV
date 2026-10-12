"""Complete independent actual-skin snapshots and CPU renders after frozen solves."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'


def skin(name):
    source = ART/f'continuous_plan_{name}.anim.json'
    animation = json.loads(source.read_text())
    assert animation['experiment_provenance']['code_unchanged_during_run']
    directory = ART/f'continuous_plan_actual_{name}'
    directory.mkdir(exist_ok=True)
    environment = {**os.environ, 'LP_NUM_THREADS':'2', 'LIBGL_ALWAYS_SOFTWARE':'1', 'MESA_SHADER_CACHE_DISABLE':'true',
                   'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
    with open(directory/'snapshot.log', 'w') as log:
        subprocess.run(['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'lab/hand_order_snapshot.py'),
            '--',str(source),str(directory),'47,48'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=environment, check=True)
    snapshots = json.loads((directory/'snapshots.json').read_text())
    assert snapshots['status']=='complete' and snapshots['source_unchanged_during_run']
    assert [r['frame'] for r in snapshots['records']]==[47,48]
    with open(directory/'audit.log', 'w') as log:
        subprocess.run([sys.executable, str(ROOT/'lab/skin_deformation_audit.py'),str(directory/'snapshots.json'),
            str(directory/'audit.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    audit = json.loads((directory/'audit.json').read_text())
    assert audit['status']=='complete' and audit['source_unchanged_during_run']
    print(json.dumps(dict(actual_skin_complete=name, snapshots=2)), flush=True)


def main():
    while not (ART/'continuous_plan_runs.json').exists():
        time.sleep(10)
    runs = json.loads((ART/'continuous_plan_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    subprocess.run([sys.executable, str(ROOT/'lab/continuous_plan_report.py')], cwd=ROOT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        audits = [pool.submit(skin, name) for name in ('windowed', 'coherent', 'warm', 'composed')]
        subprocess.run([sys.executable, str(ROOT/'lab/continuous_plan_render.py')], cwd=ROOT, check=True)
        for audit in audits:
            audit.result()
    print('Complete numerical diagnostics, actual skin and render; visual review and sealing still required', flush=True)


if __name__ == '__main__':
    main()

"""Audit initial skin plus matched instants and render, after numerical solves."""
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from physical_key_trial import ROOT,ART,TREATMENTS,sha


def skin(name):
    source=ART/f'physical_key_{name}.anim.json'
    animation=json.loads(source.read_text())
    assert animation['experiment_provenance']['code_unchanged_during_run']
    directory=ART/f'physical_key_actual_{name}'
    directory.mkdir(exist_ok=True)
    internal=copy.deepcopy(animation)
    internal['start']-=2.
    internal['frames']=copy.deepcopy(animation['warmup']['frames']+animation['frames'])
    for frame in internal['frames']:
        frame['t']+=2.
    assert len(internal['frames'])==144
    internal['audit_selection']=dict(first_physical_warmup_frame=0,
        matched_visible_frames={'138':90,'139':91},
        initial_check_selection='First physical pose; additionally selected before trial results were observed',
        full_performance_accepted=False)
    path=directory/'internal.anim.json'
    path.write_text(json.dumps(internal,allow_nan=False)+'\n')
    audit_path=directory/'audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='complete' and audit['source_unchanged_during_run']
        assert [r['frame'] for r in audit['records']]==[0,138,139]
        assert all(r['source_sha256']==sha(path) and r['mesh_sha256']==sha(Path(r['mesh'])) for r in audit['records'])
        print(json.dumps(dict(actual_skin_complete=name,reused_identical_completed_source=True)),flush=True)
        return
    environment={**os.environ,'LP_NUM_THREADS':'2','LIBGL_ALWAYS_SOFTWARE':'1',
        'MESA_SHADER_CACHE_DISABLE':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    with open(directory/'snapshot.log','w') as log:
        subprocess.run(['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'lab/hand_order_snapshot.py'),
            '--',str(path),str(directory),'0,138,139'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=environment,check=True)
    snapshots=json.loads((directory/'snapshots.json').read_text())
    assert snapshots['status']=='complete' and snapshots['source_unchanged_during_run']
    assert [r['frame'] for r in snapshots['records']]==[0,138,139]
    with open(directory/'audit.log','w') as log:
        subprocess.run([sys.executable,str(ROOT/'lab/skin_deformation_audit.py'),str(directory/'snapshots.json'),
            str(directory/'audit.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
    audit=json.loads((directory/'audit.json').read_text())
    assert audit['status']=='complete' and audit['source_unchanged_during_run']
    print(json.dumps(dict(actual_skin_complete=name,snapshots=3,initial_surface_checked=True,
        full_warmup_skin_accepted=False)),flush=True)


def main():
    runs=json.loads((ART/'physical_key_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    subprocess.run([sys.executable,str(ROOT/'lab/physical_key_report.py')],cwd=ROOT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        audits=[pool.submit(skin,name) for name in TREATMENTS]
        subprocess.run([sys.executable,str(ROOT/'lab/physical_key_render.py')],cwd=ROOT,check=True)
        for audit in audits:
            audit.result()
    print('Complete numerical diagnostics, actual skin and render; visual review/sealing still required',flush=True)


if __name__=='__main__':
    main()


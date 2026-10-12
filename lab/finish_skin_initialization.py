"""Independent Blender verification of every fixed initialization outcome."""
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from skin_initialization_trial import ROOT, ART, SOURCE, CELLS, sha, diagnostic_hand


def main():
    runs = json.loads((ART / 'runs.json').read_text())
    assert runs['status'] == 'complete' and runs['code_unchanged_during_run'] and len(runs['records']) == 8
    assert all(sha(ROOT / p) == value for p, value in runs['input_sha256'].items())
    for row in runs['records']:
        assert sha(ART / f"{row['cell']}_{row['side']}.json") == row['sha256']
    from blockout.rig.pose_io import animation_hand_scale
    source = json.loads(SOURCE.read_text())
    os.environ['HAND_SCALE'] = str(animation_hand_scale(source))
    from blockout.rig.hand import HandRig
    rigs = {side: HandRig(side) for side in ('L', 'R')}
    records = []
    environment = {**os.environ, 'LP_NUM_THREADS': '2', 'LIBGL_ALWAYS_SOFTWARE': '1',
        'MESA_SHADER_CACHE_DISABLE': 'true', 'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    fingerprints = {str(p): sha(p) for p in [Path(__file__), ROOT / 'lab/skin_initialization_trial.py',
        ROOT / 'lab/hand_order_snapshot.py', ROOT / 'lab/skin_deformation_audit.py',
        ROOT / 'blockout/blender_piano.py']}
    for cell in ('original',) + CELLS:
        directory = ART / cell
        directory.mkdir(exist_ok=True)
        animation = copy.deepcopy(source)
        animation['frames'] = copy.deepcopy(source['frames'][:1])
        animation.pop('warmup', None)
        animation['diagnostic'] = 'Unvalidated numerical initialization snapshot; no animation candidate or trajectory acceptance'
        animation['full_performance_accepted'] = False
        if cell != 'original':
            for side in ('L', 'R'):
                report = json.loads((ART / f'{cell}_{side}.json').read_text())
                hand = diagnostic_hand(rigs[side], source['frames'][0]['hands'][side],
                    source['frames'][0]['body']['shoulders'][side], report['numerical_iterate']['solver_coordinates'])
                animation['frames'][0]['hands'][side] = hand
                animation['frames'][0]['body']['elbows'][side] = hand['elbow']
        assert animation['frames'][0]['keys'] == source['frames'][0]['keys']
        path = directory / 'diagnostic.anim.json'
        path.write_text(json.dumps(animation, allow_nan=False) + '\n')
        records.append(dict(cell=cell, directory=str(directory), source=str(path), source_sha256=sha(path)))

    def verify(record):
        directory, path = Path(record['directory']), Path(record['source'])
        for name, command in [
            ('snapshot', ['/usr/bin/blender', '-b', '--factory-startup', '-t', '2', '-P',
                str(ROOT / 'lab/hand_order_snapshot.py'), '--', str(path), str(directory), '0']),
            ('audit', [sys.executable, str(ROOT / 'lab/skin_deformation_audit.py'),
                str(directory / 'snapshots.json'), str(directory / 'audit.json')]),
            ('render', ['/usr/bin/blender', '-b', '--factory-startup', '-t', '2', '-P',
                str(ROOT / 'blockout/blender_piano.py'), '--', str(path), str(directory / 'render'),
                '--frames', '0:1', '--view', 'overhead', '--eye', '0,-1,1', '--look', '0,.015,.02',
                '--lens', '32', '--hide', 'body', '--quality', 'draft', '--res', '512x288'])]:
            with (directory / f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, check=True)
            if name == 'snapshot':
                snapshots = json.loads((directory / 'snapshots.json').read_text())
                assert snapshots['status'] == 'complete' and len(snapshots['records']) == 1
                assert snapshots['source_unchanged_during_run']
            if name == 'render':
                assert 'done 1 frames' in (directory / 'render.log').read_text()
        audit = json.loads((directory / 'audit.json').read_text())
        assert audit['status'] == 'complete' and audit['source_unchanged_during_run']
        assert len(audit['records']) == 1 and sha(path) == record['source_sha256']
        record.update(audit=str(directory / 'audit.json'), audit_sha256=sha(directory / 'audit.json'),
            image=str(directory / 'render/f_00000.png'), image_sha256=sha(directory / 'render/f_00000.png'))
        print(json.dumps(dict(verified=record['cell'], self_pairs=audit['records'][0]['self_intersections'])), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(verify, records))
    assert all(sha(Path(p)) == value for p, value in fingerprints.items())
    assert all(sha(ROOT / p) == value for p, value in runs['input_sha256'].items())
    report = dict(status='complete', source_unchanged_during_run=True, input_sha256=fingerprints,
        records=records, full_performance_accepted=False, candidate_exported=False)
    (ART / 'actual_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('All five two-hand diagnostic snapshots verified and rendered; full performance remains rejected', flush=True)


if __name__ == '__main__':
    main()

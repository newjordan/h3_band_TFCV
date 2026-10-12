"""Production verification of two mixed-choice trials and all prior snapshots."""
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from skin_diagonal_trial import ROOT, ART, SOURCE, sha
from skin_initialization_trial import ART as FIRST, CELLS, diagnostic_hand


def main():
    reports = {mode: json.loads((ART / f'{mode}.json').read_text()) for mode in ('uniform', 'complete', 'complete_seeded')}
    for report in reports.values():
        assert report['source_unchanged_during_run']
        for p, h in report['input_sha256'].items():
            archived = ART / 'executed_source' / p
            assert sha(ROOT / p) == h or (archived.exists() and sha(archived) == h)
    source = json.loads(SOURCE.read_text())
    os.environ['HAND_SCALE'] = str(source['hand_scale'])
    from blockout.rig.hand import HandRig
    rig = HandRig('R')
    environment = {**os.environ, 'LP_NUM_THREADS': '2', 'LIBGL_ALWAYS_SOFTWARE': '1',
        'MESA_SHADER_CACHE_DISABLE': 'true', 'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    fingerprints = {str(p): sha(p) for p in [Path(__file__), ROOT / 'lab/hand_order_snapshot.py',
        ROOT / 'lab/skin_deformation_audit.py', ROOT / 'blockout/blender_piano.py']}
    records = []
    for mode, report in reports.items():
        directory = ART / mode
        directory.mkdir(exist_ok=True)
        animation = copy.deepcopy(source)
        animation['frames'] = copy.deepcopy(source['frames'][:1])
        animation.pop('warmup', None)
        animation['diagnostic'] = 'One unvalidated numerical initialization snapshot; no trajectory or full-performance approval'
        hand = diagnostic_hand(rig, source['frames'][0]['hands']['R'], source['frames'][0]['body']['shoulders']['R'],
            report['numerical_iterate']['solver_coordinates'])
        animation['frames'][0]['hands']['R'] = hand
        animation['frames'][0]['body']['elbows']['R'] = hand['elbow']
        assert animation['frames'][0]['keys'] == source['frames'][0]['keys']
        path = directory / 'diagnostic.anim.json'
        path.write_text(json.dumps(animation, allow_nan=False) + '\n')
        records.append(dict(mode=mode, source=str(path), source_sha256=sha(path), directory=str(directory)))

    def verify(record):
        directory, path = Path(record['directory']), Path(record['source'])
        commands = [
            ('snapshot', ['/usr/bin/blender', '-b', '--factory-startup', '-t', '2', '-P',
                str(ROOT / 'lab/hand_order_snapshot.py'), '--', str(path), str(directory), '0']),
            ('audit', [sys.executable, str(ROOT / 'lab/skin_deformation_audit.py'),
                str(directory / 'snapshots.json'), str(directory / 'audit.json')]),
            ('render', ['/usr/bin/blender', '-b', '--factory-startup', '-t', '2', '-P',
                str(ROOT / 'blockout/blender_piano.py'), '--', str(path), str(directory / 'render'),
                '--frames', '0:1', '--view', 'overhead', '--eye', '0,-1,1', '--look', '0,.015,.02',
                '--lens', '32', '--hide', 'body', '--quality', 'draft', '--res', '512x288'])]
        for name, command in commands:
            with (directory / f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, check=True)
            if name == 'snapshot':
                snapshot = json.loads((directory / 'snapshots.json').read_text())
                assert snapshot['status'] == 'complete' and snapshot['source_unchanged_during_run']
            if name == 'render':
                assert 'done 1 frames' in (directory / 'render.log').read_text()
        audit = json.loads((directory / 'audit.json').read_text())
        assert audit['status'] == 'complete' and audit['source_unchanged_during_run']
        assert len(audit['records']) == 1 and sha(path) == record['source_sha256']
        record.update(audit=str(directory / 'audit.json'), audit_sha256=sha(directory / 'audit.json'),
            image=str(directory / 'render/f_00000.png'), image_sha256=sha(directory / 'render/f_00000.png'))
        print(json.dumps(dict(verified=record['mode'], counts=audit['records'][0]['scope_counts'])), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(verify, records))
    for cell in ('original',) + CELLS:
        directory = FIRST / cell
        with (directory / 'complete_diagonal_audit.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'lab/skin_deformation_audit.py'),
                str(directory / 'snapshots.json'), str(directory / 'audit_complete_diagonals.json')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        audit = json.loads((directory / 'audit_complete_diagonals.json').read_text())
        assert audit['status'] == 'complete' and audit['source_unchanged_during_run']
        print(json.dumps(dict(reevaluated=cell, counts=audit['records'][0]['scope_counts'])), flush=True)
    assert all(sha(Path(p)) == h for p, h in fingerprints.items())
    assert all(sha(ROOT / p) == h for p, h in reports['complete_seeded']['input_sha256'].items())
    (ART / 'verification.json').write_text(json.dumps(dict(status='complete', records=records,
        input_sha256=fingerprints, source_unchanged_during_run=True, full_performance_accepted=False), indent=2) + '\n')
    print('Both new trials and all five original snapshots verified with complete diagonal-choice coverage', flush=True)


if __name__ == '__main__':
    main()

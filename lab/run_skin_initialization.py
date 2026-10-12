"""Run the eight fixed-input initialization trials without changing sources."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from skin_initialization_trial import ROOT, ART, CELLS, inputs, sha


def main():
    ART.mkdir(parents=True, exist_ok=True)
    identities = {str(p.relative_to(ROOT)): sha(p) for p in inputs()}
    manifest = dict(status='running', pid=os.getpid(), input_sha256=identities,
        cells=list(CELLS), sides=['L', 'R'], workers=3, records=[])
    path = ART / 'runs.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    begun = time.monotonic()
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def run(cell, side):
        output = ART / f'{cell}_{side}.json'
        if output.exists():
            existing = json.loads(output.read_text())
            assert existing['input_sha256'] == identities and existing['source_unchanged_during_run']
            return dict(cell=cell, side=side, sha256=sha(output), reused_identical_finished_trial=True)
        with (ART / f'{cell}_{side}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT / 'lab/skin_initialization_trial.py'), cell, side],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(f'{cell}/{side} failed, see its retained log')
        report = json.loads(output.read_text())
        assert report['input_sha256'] == identities and report['source_unchanged_during_run']
        print(json.dumps(dict(completed=f'{cell}/{side}', accepted=report['accepted'],
            self_pairs=report['mesh_self']['unexpected_surface_intersections'], seconds=report['seconds'])), flush=True)
        return dict(cell=cell, side=side, sha256=sha(output), reused_identical_finished_trial=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, cell, side) for cell in CELLS for side in ('L', 'R')]
        for future in concurrent.futures.as_completed(futures):
            manifest['records'].append(future.result())
            path.write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(sha(ROOT / p) == value for p, value in identities.items())
    manifest.update(status='complete', code_unchanged_during_run=True, seconds=time.monotonic() - begun)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Eight initialization trials complete; independent actual skin verification remains required', flush=True)


if __name__ == '__main__':
    main()

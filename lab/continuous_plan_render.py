"""CPU-only matched four-treatment performance render, with preserved audio."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'
LABELS = [('windowed', 'Two independent solves'), ('coherent', 'One full-score plan'),
          ('warm', 'Plan + physical warmup'), ('composed', 'Warmup + digit / wrist control')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    jobs, records = [], []
    fingerprints = {str(p): sha(p) for p in [ROOT/'blockout/blender_piano.py', ROOT/'blockout/skin.py', Path(__file__)]}
    for name, _ in LABELS:
        source = ART/f'continuous_plan_{name}.anim.json'
        animation = json.loads(source.read_text())
        preview = copy.deepcopy(animation)
        preview['fps'], preview['frames'] = 12, preview['frames'][::2]
        assert len(preview['frames']) == 48 and preview['frames'] == animation['frames'][::2]
        path = ART/f'continuous_plan_{name}.preview.anim.json'
        path.write_text(json.dumps(preview, allow_nan=False)+'\n')
        directory = ART/f'continuous_plan_{name}_frames'
        directory.mkdir(exist_ok=True)
        records.append(dict(treatment=name, source=str(source), source_sha256=sha(source),
            preview=str(path), preview_sha256=sha(path), source_poses_preserved=True))
        for first in (0, 24):
            log = open(ART/f'continuous_plan_render_{name}_{first}.log', 'w')
            env = {**os.environ, 'LP_NUM_THREADS':'2', 'LIBGL_ALWAYS_SOFTWARE':'1', 'MESA_SHADER_CACHE_DISABLE':'true',
                   'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
            command = ['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'blockout/blender_piano.py'),
                '--',str(path),str(directory),'--frames',f'{first}:{first+24}', '--view','overhead',
                '--eye','0,-1,1','--look','-.04,.015,.02','--lens','40','--hide','body',
                '--quality','draft','--res','768x432']
            jobs.append((subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    for process, log in jobs:
        assert process.wait() == 0
        log.close()
        assert 'done 24 frames' in Path(log.name).read_text()
    for record in records:
        directory = ART/f"continuous_plan_{record['treatment']}_frames"
        files = sorted(directory.glob('f_*.png'))
        assert len(files) == 48
        for i, p in enumerate(files):
            assert p.name == f'f_{i:05d}.png'
            with Image.open(p) as picture:
                assert picture.size == (768, 432)
                picture.load()
        record['frames_sha256'] = {p.name: sha(p) for p in files}
        assert sha(Path(record['source'])) == record['source_sha256']
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters = ';'.join(f"[{i}:v]format=gray,drawtext=fontfile={font}:text='{label}':x=12:y=12:fontsize=19:fontcolor=white:box=1:boxcolor=black@0.65[v{i}]"
                       for i, (_, label) in enumerate(LABELS))
    filters += ';[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]'
    command = ['ffmpeg','-y','-v','warning']
    for name, _ in LABELS:
        command += ['-framerate','12','-i',str(ART/f'continuous_plan_{name}_frames/f_%05d.png')]
    video = ART/'continuous_plan_comparison.mp4'
    command += ['-ss','282.9','-i',str(ART/'performance_source_v19/ks.wav'),'-filter_complex',filters,
        '-map','[v]','-map','4:a','-t','4','-c:v','libx264','-preset','fast','-crf','21','-threads','4',
        '-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(video)]
    subprocess.run(command, check=True)
    assert all(sha(Path(p))==h for p,h in fingerprints.items())
    report = dict(status='encoded', records=records, input_sha256=fingerprints, source_unchanged_during_run=True,
        video=str(video), video_sha256=sha(video), frames=48, fps=12, duration_seconds=4., resolution=[1536,864],
        start=282.9, gpu_used=False, full_performance_accepted=False)
    (ART/'continuous_plan_render.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()

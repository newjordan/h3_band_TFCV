"""Render a matched eight-second regression comparison, using CPU only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    jobs,records=[],[]
    for trial in ('baseline','ordered'):
        source=ART/f'hand_order_{trial}.anim.json'
        animation=json.loads(source.read_text())
        preview=copy.deepcopy(animation);preview['fps']=12;preview['frames']=preview['frames'][::2]
        assert len(preview['frames'])==96
        assert preview['frames']==animation['frames'][::2]
        path=ART/f'hand_order_{trial}.preview.anim.json'
        path.write_text(json.dumps(preview)+'\n')
        directory=ART/f'hand_order_{trial}_frames';directory.mkdir(exist_ok=True)
        records.append(dict(trial=trial,source=str(source),source_sha256=sha(source),
            preview=str(path),preview_sha256=sha(path),source_poses_preserved=True))
        for first in range(0,96,24):
            log=open(ART/f'hand_order_compare_{trial}_{first}.log','w')
            environment={**os.environ,'LP_NUM_THREADS':'2','LIBGL_ALWAYS_SOFTWARE':'1','MESA_SHADER_CACHE_DISABLE':'true',
                         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
            command=['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'blockout/blender_piano.py'),
                '--',str(path),str(directory),'--frames',f'{first}:{first+24}','--view','overhead',
                '--eye','0,-1,1','--look','-.04,.015,.02','--lens','40','--hide','body','--quality','draft','--res','768x432']
            jobs.append((subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment),log))
    for process,log in jobs:
        assert process.wait()==0;log.close()
        assert 'done 24 frames' in Path(log.name).read_text()
    for record in records:
        files=sorted((ART/f"hand_order_{record['trial']}_frames").glob('f_*.png'))
        assert len(files)==96
        for i,p in enumerate(files):
            assert p.name==f'f_{i:05d}.png'
            with Image.open(p) as picture:
                assert picture.size==(768,432);picture.load()
        record['frames_sha256']={p.name:sha(p) for p in files}
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters=(f"[0:v]format=gray,drawtext=fontfile={font}:text='Baseline':x=14:y=14:fontsize=21:fontcolor=white:box=1:boxcolor=black@0.65[a];"
             f"[1:v]format=gray,drawtext=fontfile={font}:text='Ordered hands (trial)':x=14:y=14:fontsize=21:fontcolor=white:box=1:boxcolor=black@0.65[b];"
             '[a][b]hstack=inputs=2[v]')
    video=ART/'hand_order_comparison.mp4'
    subprocess.run(['ffmpeg','-y','-v','warning','-framerate','12','-i',str(ART/'hand_order_baseline_frames/f_%05d.png'),
        '-framerate','12','-i',str(ART/'hand_order_ordered_frames/f_%05d.png'),'-ss','287','-i',str(ART/'performance_source_v19/ks.wav'),
        '-filter_complex',filters,'-map','[v]','-map','2:a','-t','8','-c:v','libx264','-preset','fast','-crf','21',
        '-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(video)],check=True)
    proof=dict(status='encoded',records=records,video=str(video),video_sha256=sha(video),
        frames=96,fps=12,duration_seconds=8.,resolution=[1536,432],start=287.,gpu_used=False,
        full_performance_accepted=False)
    (ART/'hand_order_comparison_render.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='records'},indent=2))


if __name__=='__main__':
    main()

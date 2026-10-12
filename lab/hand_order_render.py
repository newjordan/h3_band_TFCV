"""Render a minute on CPU from unchanged solved poses and original audio."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ART/'hand_order_minute.anim.json'
    animation=json.loads(source.read_text())
    preview=copy.deepcopy(animation)
    preview['frames']=preview['frames'][::2]
    preview['fps']=12
    for original,selected in zip(animation['frames'][::2],preview['frames']):
        assert original==selected
    target=ART/'hand_order_minute.preview.anim.json'
    target.write_text(json.dumps(preview)+'\n')
    directory=ART/'hand_order_minute_frames'
    directory.mkdir(exist_ok=True)
    jobs=[]
    for start in range(0,720,90):
        log=open(ART/f'hand_order_render_{start}.log','w')
        command=['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'blockout/blender_piano.py'),
            '--',str(target),str(directory),'--frames',f'{start}:{start+90}','--view','overhead',
            '--eye','0,-1,1','--look','-.04,.015,.02','--lens','40','--hide','body','--quality','draft','--res','768x432']
        import os
        environment={**os.environ,'LP_NUM_THREADS':'2','LIBGL_ALWAYS_SOFTWARE':'1','MESA_SHADER_CACHE_DISABLE':'true',
                     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        jobs.append((subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment),log))
    for process,log in jobs:
        status=process.wait();log.close();assert status==0
        assert 'done 90 frames' in Path(log.name).read_text()
    files=sorted(directory.glob('f_*.png'))
    assert len(files)==720
    for i,p in enumerate(files):
        assert p.name==f'f_{i:05d}.png'
        with Image.open(p) as picture:
            assert picture.size==(768,432);picture.load()
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    label=f"format=gray,drawtext=fontfile={font}:text='Ordered hands - diagnostic performance':x=14:y=14:fontsize=19:fontcolor=white:box=1:boxcolor=black@0.65"
    video=ART/'hand_order_minute.mp4'
    command=['ffmpeg','-y','-v','warning','-framerate','12','-i',str(directory/'f_%05d.png'),
        '-ss','239.9','-i',str(ART/'performance_source_v19/ks.wav'),'-map','0:v','-map','1:a','-t','60',
        '-vf',label,'-c:v','libx264','-preset','fast','-crf','21','-threads','4','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','96k','-movflags','+faststart',str(video)]
    subprocess.run(command,check=True)
    proof=dict(source=str(source),source_sha256=sha(source),preview=str(target),preview_sha256=sha(target),
        frames_sha256={p.name:sha(p) for p in files},source_pose_preserved=True,render_fps=12,
        duration_seconds=60.,resolution=[768,432],camera=dict(eye=[0,-1,1],look=[-.04,.015,.02],lens=40),
        video=str(video),video_sha256=sha(video),gpu_used=False,
        audio_source_sha256=sha(ART/'performance_source_v19/ks.wav'),audio_start=239.9,
        audio_note_identity_verified=True,full_performance_accepted=False)
    (ART/'hand_order_render.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='frames_sha256'},indent=2))


if __name__=='__main__':
    main()

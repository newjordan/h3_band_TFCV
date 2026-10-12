"""CPU grayscale comparison of unchanged three-treatment sampled poses."""
import copy
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

from physical_key_trial import ROOT,ART,sha

LABELS=[('legacy','Current scheduling'),('conservative','Conservative key holds'),
        ('conditional','Conditional pedal releases')]


def main():
    paths=[ROOT/'blockout/blender_piano.py',ROOT/'blockout/skin.py',Path(__file__)]
    fingerprints={str(p):sha(p) for p in paths}
    jobs,records=[],[]
    for name,_ in LABELS:
        source=ART/f'physical_key_{name}.anim.json'
        animation=json.loads(source.read_text())
        preview=copy.deepcopy(animation)
        preview['fps']=12
        preview['frames']=preview['frames'][::2]
        assert preview['frames']==animation['frames'][::2] and len(preview['frames'])==48
        path=ART/f'physical_key_{name}.preview.anim.json'
        path.write_text(json.dumps(preview,allow_nan=False)+'\n')
        directory=ART/f'physical_key_{name}_frames'
        directory.mkdir(exist_ok=True)
        log=open(ART/f'physical_key_render_{name}.log','w')
        env={**os.environ,'LP_NUM_THREADS':'2','LIBGL_ALWAYS_SOFTWARE':'1',
            'MESA_SHADER_CACHE_DISABLE':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        command=['/usr/bin/blender','-b','--factory-startup','-t','2','-P',str(ROOT/'blockout/blender_piano.py'),
            '--',str(path),str(directory),'--frames','0:48','--view','overhead','--eye','0,-1,1',
            '--look','0,.015,.02','--lens','32','--hide','body','--quality','draft','--res','512x288']
        jobs.append((subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env),log))
        records.append(dict(treatment=name,source=str(source),source_sha256=sha(source),
            preview=str(path),preview_sha256=sha(path),source_poses_preserved=True))
    for process,log in jobs:
        assert process.wait()==0
        log.close()
        assert 'done 48 frames' in Path(log.name).read_text()
    for record in records:
        directory=ART/f"physical_key_{record['treatment']}_frames"
        files=sorted(directory.glob('f_*.png'))
        assert len(files)==48
        for i,path in enumerate(files):
            assert path.name==f'f_{i:05d}.png'
            with Image.open(path) as picture:
                assert picture.size==(512,288)
                picture.load()
        record['frames_sha256']={p.name:sha(p) for p in files}
        assert sha(Path(record['source']))==record['source_sha256']
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters=';'.join(f"[{i}:v]format=gray,drawbox=x=0:y=0:w=iw:h=25:color=black:t=fill,drawtext=fontfile={font}:text='{label}':x=8:y=5:fontsize=15:fontcolor=white[v{i}]"
        for i,(_,label) in enumerate(LABELS))
    filters+=';[v0][v1][v2]hstack=inputs=3[v]'
    command=['ffmpeg','-y','-v','warning']
    for name,_ in LABELS:
        command+=['-framerate','12','-i',str(ART/f'physical_key_{name}_frames/f_%05d.png')]
    video=ART/'physical_key_comparison.mp4'
    command+=['-ss','318.9','-i',str(ART/'performance_source_v19/ks.wav'),'-filter_complex',filters,
        '-map','[v]','-map','3:a','-t','4','-c:v','libx264','-preset','fast','-crf','21','-threads','4',
        '-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(video)]
    subprocess.run(command,check=True)
    assert all(sha(Path(p))==h for p,h in fingerprints.items())
    report=dict(status='encoded',records=records,input_sha256=fingerprints,source_unchanged_during_run=True,
        video=str(video),video_sha256=sha(video),frames=48,fps=12,duration_seconds=4.,resolution=[1536,288],
        start=318.9,gpu_used=False,full_performance_accepted=False)
    report['camera']=dict(view='overhead',eye=[0.,-1.,1.],look=[0.,.015,.02],lens=32,
        change_reason='Initial matched camera clipped part of the baseline right hand; numerical poses unchanged')
    (ART/'physical_key_render.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    main()

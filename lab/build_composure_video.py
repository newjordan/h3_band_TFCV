"""Encode matched grayscale controllers with the identical performed audio."""
from pathlib import Path
import subprocess

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def main():
    for trial in ('baseline','both'):
        files=sorted((ART/f'composure_{trial}_frames').glob('f_*.png'))
        assert len(files)==72
        for i,file in enumerate(files):
            assert file.name==f'f_{i:05d}.png'
            with Image.open(file) as image:
                assert image.size==(768,432)
                image.load()
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters=(f"[0:v]format=gray,drawtext=fontfile={font}:text='Current controller':x=16:y=16:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[a];"
             f"[1:v]format=gray,drawtext=fontfile={font}:text='Curled fingers + wrist carry (trial)':x=16:y=16:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[b];"
             '[a][b]hstack=inputs=2[v]')
    command=['ffmpeg','-y','-v','warning','-framerate','12','-i',str(ART/'composure_baseline_frames/f_%05d.png'),
        '-framerate','12','-i',str(ART/'composure_both_frames/f_%05d.png'),
        '-ss','247','-i',str(ART/'performance_source_v19/ks.wav'),
        '-filter_complex',filters,'-map','[v]','-map','2:a','-t','6',
        '-c:v','libx264','-preset','fast','-crf','21','-threads','4','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','96k','-movflags','+faststart',str(ART/'composure_comparison.mp4')]
    subprocess.run(command,check=True)


if __name__=='__main__':
    main()

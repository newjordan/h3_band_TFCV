"""Whole-score archive preview, including the opening omitted by the v19 cut."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import wave
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'
WEB = Path('/home/frosty40/h3/results/band_site/web')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    began = time.monotonic()
    archive = Path('/tmp/claude-1000/suite_v5')
    for name, source in [('full_suite_opening_source.anim.json', archive/'ks_v5.json'),
                         ('full_suite_opening_source.notes.json', archive/'notes.json')]:
        shutil.copy2(source, ART/name)
    opening_source = ART/'full_suite_opening_source.anim.json'
    main_source = WEB/'media/suite_grey_v19.mp4'
    original_notes = ART/'performance_source_v19/notes.json'
    audio = ART/'performance_source_v19/ks.wav'
    fingerprints = {str(p): sha(p) for p in (opening_source, main_source, original_notes, audio,
        ART/'full_suite_opening_source.notes.json', ROOT/'blockout/blender_piano.py', Path(__file__))}
    opening = json.loads(opening_source.read_text())
    old = [n for n in json.loads((ART/'full_suite_opening_source.notes.json').read_text()) if n[0]<39.9]
    new = [n for n in json.loads(original_notes.read_text()) if n[0]<39.9]
    assert len(old)==len(new)==128
    assert all([a[j] for j in (0,1,2,4)]==[b[j] for j in (0,1,2,4)] for a,b in zip(old,new))
    velocity_difference = dict(affected_notes=sum(a[3]!=b[3] for a,b in zip(old,new)),
        maximum_midi_velocity_difference=max(abs(a[3]-b[3]) for a,b in zip(old,new)),
        scope='Earlier opening motion has different touch dynamics; this is an archive preview, not a matched controller test')
    assert opening['start']==0. and opening['fps']==24
    preview = copy.deepcopy(opening)
    preview['frames'] = [copy.deepcopy(f) for f in opening['frames'][::2] if f['t']<39.9]
    preview['fps'], preview['hand_scale'] = 12, 1.
    preview['archive_scope'] = velocity_difference
    assert len(preview['frames'])==479
    path = ART/'full_suite_opening.preview.anim.json'
    path.write_text(json.dumps(preview, allow_nan=False)+'\n')
    directory = ART/'full_suite_opening_frames'
    directory.mkdir(exist_ok=True)
    # Avoid overlapping the 16-thread comparison renderer and independent skin jobs.
    while not (ART/'continuous_plan_render.json').exists():
        time.sleep(10)
    jobs = []
    for first in range(0,479,120):
        last = min(first+120,479)
        log = open(ART/f'full_suite_opening_render_{first}.log','w')
        env = {**os.environ,'HAND_SCALE':'1','LP_NUM_THREADS':'1','LIBGL_ALWAYS_SOFTWARE':'1',
               'MESA_SHADER_CACHE_DISABLE':'true','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        command = ['/usr/bin/blender','-b','--factory-startup','-t','1','-P',str(ROOT/'blockout/blender_piano.py'),
            '--',str(path),str(directory),'--frames',f'{first}:{last}','--view','overhead',
            '--eye','0,-1,1','--look','-.04,.015,.02','--lens','40','--quality','draft','--res','512x288']
        jobs.append((subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env),log,last-first))
    for process,log,count in jobs:
        assert process.wait()==0
        log.close()
        assert f'done {count} frames' in Path(log.name).read_text()
    files = sorted(directory.glob('f_*.png'))
    assert len(files)==479
    for i,p in enumerate(files):
        assert p.name==f'f_{i:05d}.png'
        with Image.open(p) as picture:
            assert picture.size==(512,288)
            picture.load()
    with wave.open(str(audio)) as value:
        duration = value.getnframes()/value.getframerate()
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters = ('[0:v]format=gray,setsar=1,setpts=PTS-STARTPTS[opening];'
        '[1:v]fps=12,scale=512:288:force_original_aspect_ratio=decrease,pad=512:288:(ow-iw)/2:(oh-ih)/2,format=gray,setsar=1,setpts=PTS-STARTPTS[main];'
        '[opening][main]concat=n=2:v=1:a=0,tpad=stop_mode=clone:stop_duration=4,'
        f"drawtext=fontfile={font}:text='Archive motion - controller improvements pending':x=8:y=8:fontsize=12:fontcolor=white:box=1:boxcolor=black@0.7[v]")
    video = ART/'full_suite_archive_preview.mp4'
    subprocess.run(['ffmpeg','-y','-v','warning','-framerate','12','-i',str(directory/'f_%05d.png'),
        '-i',str(main_source),'-i',str(audio),'-filter_complex',filters,'-map','[v]','-map','2:a',
        '-t',str(duration),'-c:v','libx264','-preset','fast','-crf','22','-threads','4','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','96k','-movflags','+faststart',str(video)],check=True)
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    assert stream['width']==512 and stream['height']==288 and stream['avg_frame_rate']=='12/1'
    assert abs(float(stream['duration'])-duration)<=1./12+.001
    assert int(stream['nb_read_frames'])>=math.floor(duration*12)
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    assert all(sha(Path(p))==h for p,h in fingerprints.items())
    report = dict(schema='whole-suite-archive-preview/v1',status='verified', video=str(video),video_sha256=sha(video),
        resolution=[512,288],fps=12,original_audio_duration_seconds=duration,duration_seconds=float(stream['duration']),
        input_sha256=fingerprints,source_unchanged_during_run=True,ffprobe=probe,full_decode_errors=0,
        opening=dict(source=str(opening_source),frames=479,source_start=0.,source_end=39.9,
            note_times_pitches_releases_tracks_identical=True,velocity_differences=velocity_difference),
        main=dict(source=str(main_source),score_start=39.9,score_end=390.525,
            presentation_offset_seconds=479/12.-39.9),
        original_audio_preserved=True,playback_speed=1.,gpu_used=False,seconds=time.monotonic()-began,
        full_performance_accepted=False,latest_controller_integrated=False,
        scope='Complete score audio with archived opening and main performance; final posture held during audio decay',
        limitations=['Earlier opening touch dynamics', 'v19 crossing/clipping and solver joins remain',
            'archive seam is not a continuous solve', 'actual full body/costume and force/action unverified'])
    (ROOT/'lab/realism_results/full_suite_preview_20261011.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','video','video_sha256','duration_seconds','latest_controller_integrated')},indent=2))


if __name__=='__main__':
    main()

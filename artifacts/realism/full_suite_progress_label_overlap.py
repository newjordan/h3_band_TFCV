"""Whole-suite review cut with the completed wrist-order minute in its score position."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ART/'full_suite_archive_preview.mp4'
    minute = ART/'hand_order_minute.mp4'
    archive_path = ROOT/'lab/realism_results/full_suite_preview_20261011.json'
    order_path = ROOT/'lab/realism_results/hand_order_20261011.json'
    audio = ART/'performance_source_v19/ks.wav'
    archive = json.loads(archive_path.read_text())
    order = json.loads(order_path.read_text())
    assert archive['status']=='verified' and sha(source)==archive['video_sha256']
    assert order['status']=='complete' and order['minute']['metrics']['order']['passed']
    assert sha(minute)==order['render']['video_sha256']
    hashes = {str(p):sha(p) for p in (source,minute,archive_path,order_path,audio,Path(__file__))}
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters = ('[0:v]split=2[a][b];[a]trim=end=239.9,setpts=PTS-STARTPTS[before];'
        '[b]trim=start=299.9,setpts=PTS-STARTPTS[after];'
        '[1:v]scale=512:288,setsar=1,format=gray,'
        f"drawtext=fontfile={font}:text='Current wrist order - physics still incomplete':x=8:y=8:fontsize=12:fontcolor=white:box=1:boxcolor=black@0.7[mid];"
        '[before][mid][after]concat=n=3:v=1:a=0[v]')
    video = ART/'full_suite_progress_preview.mp4'
    duration = archive['original_audio_duration_seconds']
    subprocess.run(['ffmpeg','-y','-v','warning','-i',str(source),'-i',str(minute),'-i',str(audio),
        '-filter_complex',filters,'-map','[v]','-map','2:a','-t',str(duration),'-c:v','libx264',
        '-preset','fast','-crf','22','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k',
        '-movflags','+faststart',str(video)],check=True)
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],int(stream['nb_read_frames']))==(512,288,4725)
    assert stream['avg_frame_rate']=='12/1' and abs(float(stream['duration'])-duration)<=1./12+.001
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report = dict(archive, schema='whole-suite-progress-preview/v1', status='verified',video=str(video),
        video_sha256=sha(video),ffprobe=probe,archive_preview_source=str(archive_path),
        archive_preview_sha256=sha(archive_path),progress_cut_input_sha256=hashes,
        latest_controller_integrated='partial: wrist-order minute at 239.9–299.9 seconds',
        latest_segment=dict(score_start=239.9,score_end=299.9,source=str(minute),source_sha256=sha(minute),
            displayed_frames=720,frame_grid_start=math.ceil(239.9*12)/12,
            renderer_samples_preserved=True,original_audio_timeline=True,
            passed_scope='sampled wrist-root order only',failed_chunk_joins=7,
            actual_skin_failure_maximum_mm=6.533311676979064),
        scope='Whole-score audio and actual rendered archived motion, with completed wrist-order minute inserted at its original score position. '
              'Visible controller/camera cuts; no pose blending or continuous-physics acceptance.')
    report['limitations'] += ['Only one minute uses the updated wrist-order controller',
        'Preview cuts and seven internal minute joins are not a continuous solve',
        'Four-second continuous-plan experiment is not integrated into this full review cut']
    (ROOT/'lab/realism_results/full_suite_progress_20261011.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','video','video_sha256','duration_seconds','latest_controller_integrated')},indent=2))


if __name__=='__main__':
    main()

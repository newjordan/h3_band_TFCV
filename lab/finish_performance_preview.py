"""Verify the complete diagnostic render and archive its reproducible identity."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    proof_path = ART / 'performance_60s.anim.preview.json'
    report = json.loads(proof_path.read_text())
    assert sha(Path(report['source'])) == report['source_sha256']
    assert sha(Path(report['animation'])) == report['animation_sha256']
    assert sha(Path(report['audio_source'])) == report['audio_source_sha256']
    copies = ART / 'performance_source_v19'
    preserved = {str(copies / name): sha(copies / name) for name in ('ks_v19.json','ks.wav','run_parts.sh')}
    assert preserved[str(copies / 'ks_v19.json')] == report['source_sha256']
    assert preserved[str(copies / 'ks.wav')] == report['audio_source_sha256']
    assert preserved[str(copies / 'run_parts.sh')] == report['original_scale_run_script_sha256']
    report['preserved_workspace_sources'] = preserved
    report['preview_producer_sha256'] = {p: sha(ROOT / p) for p in (
        'lab/prepare_performance_preview.py','lab/finish_performance_preview.py')}
    assert all(sha(ROOT / p) == s for p,s in report['renderer_input_sha256'].items())
    frames = ART / 'performance_60s_frames'
    paths = sorted(frames.glob('f_*.png'))
    assert [p.name for p in paths] == [f'f_{i:05d}.png' for i in range(report['render_frames'])]
    frame_hashes = []
    for p in paths:
        with Image.open(p) as im:
            assert list(im.size) == report['render']['resolution']
            im.verify()
        frame_hashes.append(sha(p))
    assert len(set(frame_hashes)) > report['render_frames'] * .9, 'Unexpected frozen/duplicate motion'
    report['png_sha256'] = frame_hashes
    report['render_worker_logs'] = {}
    for n in range(report['render']['workers']):
        log = ART / f'performance_60s_render_{n}.log'
        text = log.read_text()
        assert 'Traceback' not in text and 'done 180 frames' in text and 'Blender quit' in text
        report['render_worker_logs'][log.name] = sha(log)
    video = ART / 'performance_60s.mp4'
    probe = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-count_frames','-show_entries',
        'stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration,size',
        '-of','json',str(video)], text=True))
    vs = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    assert vs['codec_name'] == 'h264' and audio['codec_name'] == 'aac'
    assert [vs['width'],vs['height']] == report['render']['resolution']
    assert vs['r_frame_rate'] == f"{report['render_fps']}/1"
    assert int(vs['nb_read_frames']) == report['render_frames']
    assert abs(float(probe['format']['duration']) - report['duration_seconds']) < .1
    decode = ART / 'performance_60s_decode.log'
    assert decode.read_text() == '', 'Encoded video/audio must decode without errors'
    audio_log = ART / 'performance_60s_audio_check.log'
    volume = re.search(r'max_volume: ([\d.-]+) dB', audio_log.read_text())
    assert volume and -80. < float(volume[1]) < 0., 'Audio must contain a finite unclipped signal'
    report.update(status='complete', video=str(video), video_sha256=sha(video), ffprobe=probe,
        decode_errors=0, decode_log_sha256=sha(decode), audio_peak_db=float(volume[1]), audio_check_sha256=sha(audio_log),
        timing='Source frames 4800:6240:2; video at 12 fps; audio trimmed at original absolute time 239.9 s; tempo unchanged',
        visual_samples=[0,360,719], original_pose_key_body_data_preserved=True,
        newest_static_contact_fixes_integrated=False, full_performance_accepted=False,
        scope='One continuous minute of the earlier full v19 planner using current production skinned geometry. '
              'Diagnostic performance with unresolved clipping/posture failures; not a new physics-approved trajectory.')
    dest = ROOT / 'lab/realism_results/performance_preview_20261011.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(video=str(video),duration=report['duration_seconds'],frames=report['render_frames'],
        sha256=report['video_sha256'], full_performance_accepted=False),indent=2))


if __name__ == '__main__':
    main()

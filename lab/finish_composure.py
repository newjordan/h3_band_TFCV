"""Seal completed controller, actual-surface and video evidence without accepting it."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / 'lab/realism_results/composure_20261011.json'
    report = json.loads(destination.read_text())
    report['actual_mesh_audit'] = {}
    for trial in report['records']:
        path = ART / f'composure_actual_{trial}/audit.json'
        audit = json.loads(path.read_text())
        assert audit['status'] == 'complete' and audit['source_unchanged_during_run']
        assert len(audit['records']) == 2
        report['actual_mesh_audit'][trial] = audit
    video = ART / 'composure_comparison.mp4'
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    assert (stream['width'], stream['height'], int(stream['nb_read_frames'])) == (1536, 432, 72)
    assert stream['avg_frame_rate'] == '12/1'
    assert abs(float(stream['duration']) - 6.) < .01
    decoded = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],
                             capture_output=True,text=True)
    assert decoded.returncode == 0 and not decoded.stderr
    frames = {}
    for trial in ('baseline','both'):
        directory = ART / f'composure_{trial}_frames'
        files = sorted(directory.glob('f_*.png'))
        assert len(files) == 72 and files[0].name == 'f_00000.png' and files[-1].name == 'f_00071.png'
        frames[trial] = {file.name:sha(file) for file in files}
        assert len(set(frames[trial].values())) > 60
    proof = json.loads((ART / 'composure_preview.json').read_text())
    report['render'] = dict(proof, verified=True, video=str(video), video_sha256=sha(video),
        ffprobe=probe, frames_sha256=frames, full_decode_errors=0,
        visual_samples_reviewed=[0,36,71], audio_source=str(ART/'performance_source_v19/ks.wav'),
        audio_mapping='Original performed audio, absolute seconds247–253; full note list identity verified separately')
    original_audio_notes = Path('/tmp/claude-1000/suite_v19/notes.json')
    original_notes = json.loads(original_audio_notes.read_text())
    trial_notes = json.loads((ART / 'composure_both.anim.notes.json').read_text())
    assert original_notes == trial_notes
    archive = ART / 'performance_source_v19/notes.json'
    archive.write_text(json.dumps(original_notes)+'\n')
    report['render']['audio_note_identity_verified'] = True
    report['render']['audio_note_source_sha256'] = sha(archive)
    tests = ART/'composure_tests.log'
    count = re.search(r'Ran (\d+) tests',tests.read_text())
    assert count and tests.read_text().rstrip().endswith('OK')
    report['unit_tests']=dict(passed=int(count[1]),log=str(tests),sha256=sha(tests))
    episode=ROOT/'lab/realism_results/composure_review_episode.json'
    receipt=json.loads(episode.read_text())
    assert receipt['termination']['kind']=='completed' and len(receipt['traces'])==5
    report['reviewed_method']=dict(episode_id=receipt['episode_id'],episode_file_sha256=sha(episode),
        native_graph_roles=5,review_scope='methodology only; requested before controller outcomes')
    report['actual_mesh_audit_pending']=False
    report['status']='complete'
    report['conclusion']='Diagnostic controller outcomes only; no complete realistic performance accepted. '
    report['production_defaults_changed']=False
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=report['unit_tests']['passed'],
        video_sha256=sha(video),accepted=report['full_performance_accepted'])))


if __name__=='__main__':
    main()

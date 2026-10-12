"""Seal complete order diagnostics, independent skin snapshots and video."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path=ROOT/'lab/realism_results/hand_order_20261011.json'
    report=json.loads(path.read_text())
    assert report['runs']['status']=='complete' and report['runs']['code_unchanged_during_run']
    notes=json.loads((ART/'performance_source_v19/notes.json').read_text())
    for trial in ['baseline','ordered']+[f'minute_{i}' for i in range(8)]:
        assert json.loads((ART/f'hand_order_{trial}.anim.notes.json').read_text())==notes
    ownership=json.loads((ART/'hand_order_ownership_check.json').read_text())
    assert ownership['events']==2046 and not ownership['whole_score_held_register_inversions']
    report['whole_score_ownership_check']=ownership
    report['actual_mesh_audit']={}
    for trial in ('baseline','ordered','minute'):
        audit_path=ART/f'hand_order_actual_{trial}/audit.json'
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='complete' and audit['source_unchanged_during_run']
        assert len(audit['records'])==(1 if trial=='minute' else 2)
        report['actual_mesh_audit'][trial]=audit
    render=json.loads((ART/'hand_order_render.json').read_text())
    video=Path(render['video'])
    assert sha(video)==render['video_sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],int(stream['nb_read_frames']))==(768,432,720)
    assert stream['avg_frame_rate']=='12/1' and abs(float(stream['duration'])-60.)<.01
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    tests=ART/'hand_order_tests.log';count=re.search(r'Ran (\d+) tests',tests.read_text())
    assert count and tests.read_text().rstrip().endswith('OK')
    report['unit_tests']=dict(passed=int(count[1]),log=str(tests),sha256=sha(tests))
    report['render']=dict(render,verified=True,ffprobe=probe,full_decode_errors=0,
        visual_samples_reviewed=json.loads((ART/'hand_order_visual_review.json').read_text()))
    assert report['render']['visual_samples_reviewed']['reviewed']
    comparison=json.loads((ART/'hand_order_comparison_render.json').read_text())
    comparison_video=Path(comparison['video'])
    assert sha(comparison_video)==comparison['video_sha256']
    comparison_probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(comparison_video)]))
    comparison_stream=next(s for s in comparison_probe['streams'] if s['codec_type']=='video')
    assert (comparison_stream['width'],comparison_stream['height'],int(comparison_stream['nb_read_frames']))==(1536,432,96)
    assert comparison_stream['avg_frame_rate']=='12/1' and abs(float(comparison_stream['duration'])-8.)<.01
    comparison_decode=subprocess.run(['ffmpeg','-v','error','-i',str(comparison_video),'-f','null','-'],capture_output=True,text=True)
    assert comparison_decode.returncode==0 and not comparison_decode.stderr
    report['comparison_render']=dict(comparison,verified=True,ffprobe=comparison_probe,full_decode_errors=0)
    report['actual_mesh_audit_pending']=False
    report['status']='complete'
    report['conclusion']='Requested wrist-root order passes the saved minute. Complete realistic motion remains unaccepted; all music, skin and chunk-boundary failures are retained.'
    assert report['minute']['metrics']['order']['passed']
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],order=report['minute']['metrics']['order'],
        tests=report['unit_tests']['passed'],video_sha256=sha(video),full_performance_accepted=False)))


if __name__=='__main__':
    main()

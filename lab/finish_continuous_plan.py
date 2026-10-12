"""Seal every completed outcome, actual skin witness and reviewed video."""
import json
from pathlib import Path
import re
import subprocess

from continuous_plan_report import TREATMENTS, sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'


def main():
    path = ROOT/'lab/realism_results/continuous_plan_20261011.json'
    report = json.loads(path.read_text())
    assert report['runs']['status']=='complete' and report['runs']['code_unchanged_during_run']
    for name, record in report['records'].items():
        assert sha(Path(record['source']))==record['sha256']
    audits = {}
    for name in TREATMENTS:
        audit = json.loads((ART/f'continuous_plan_actual_{name}/audit.json').read_text())
        assert audit['status']=='complete' and audit['source_unchanged_during_run']
        assert [r['frame'] for r in audit['records']]==[47,48]
        audits[name] = audit
    render = json.loads((ART/'continuous_plan_render.json').read_text())
    video = Path(render['video'])
    assert sha(video)==render['video_sha256'] and render['source_unchanged_during_run']
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],int(stream['nb_read_frames']))==(1536,864,48)
    assert stream['avg_frame_rate']=='12/1' and abs(float(stream['duration'])-4.)<.01
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    visual = json.loads((ART/'continuous_plan_visual_review.json').read_text())
    assert visual['reviewed']
    log = ART/'continuous_plan_tests.log'
    count = re.search(r'Ran (\d+) tests',log.read_text())
    assert count and log.read_text().rstrip().endswith('OK')
    review = json.loads((ROOT/'lab/realism_results/continuous_plan_review_result.json').read_text())
    assert review['status']=='completed'
    report.update(status='complete', actual_mesh_audit_pending=False, actual_mesh_audit=audits,
        render=dict(render,verified=True,ffprobe=probe,full_decode_errors=0,visual_review=visual),
        unit_tests=dict(passed=int(count[1]),log=str(log),sha256=sha(log)),
        native_research=dict(result=str(ROOT/'lab/realism_results/continuous_plan_review_result.json'),
            episode=str(ROOT/'lab/realism_results/continuous_plan_review_episode.json'),
            episode_id='e085d72d3a37db6ce0f96f39477b8a15ea4495c39b2ac99789d6579273fe42b7',
            completed_nodes=5,verdict_scope='Methodological PASS; not empirical rig acceptance',
            recommended_additional_controls=['same-plan history reset at midpoint','controller without warmup'],
            additional_controls_status='pending'),
        conclusion='A single chronological solve removes the sampled join discontinuity in this passage. '
            'Full-score preparation also changes fingering, body placement and context, so attribution remains combined. '
            'Actual skin, original music and full realistic performance remain unaccepted.')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',tests=int(count[1]),video_sha256=sha(video),full_performance_accepted=False)))


if __name__=='__main__':
    main()

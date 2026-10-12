"""Seal complete trial evidence, independent skin and reviewed media."""
import json
from pathlib import Path
import re
import subprocess

from orientation_trial import ROOT,ART,TREATMENTS,sha


def diagnostic_comparisons(report,audits):
    baseline=report['records']['legacy']
    result={}
    for name in TREATMENTS:
        current=report['records'][name]
        m=current['metrics']
        geometric_regressions=[]
        for before,after in zip(audits['legacy']['records'],audits[name]['records']):
            assert before['frame']==after['frame']
            for side in ('R','L'):
                # Comparative tolerance covers the independently measured
                # Blender/CPU precision; it is not a no-clipping allowance.
                if after['instrument'][side]['maximum_penetration_mm']>before['instrument'][side]['maximum_penetration_mm']+.002:
                    geometric_regressions.append(dict(frame=after['frame'],hand=side,kind='instrument depth',
                        baseline_mm=before['instrument'][side]['maximum_penetration_mm'],
                        treatment_mm=after['instrument'][side]['maximum_penetration_mm']))
                a={(x['finger'],x['pitch']):x for x in before['assigned_support'][side]}
                b={(x['finger'],x['pitch']):x for x in after['assigned_support'][side]}
                assert a.keys()==b.keys()
                for identity,value in a.items():
                    if value['activation']>=.7 and value['supported'] and not b[identity]['supported']:
                        geometric_regressions.append(dict(frame=after['frame'],hand=side,
                            kind='lost strongly active finite assigned support',finger=identity[0],pitch=identity[1]))
            for diagonal,(a,b) in enumerate(zip(before['scope_counts'],after['scope_counts'])):
                for scope in set(a)|set(b):
                    if b.get(scope,0)>a.get(scope,0):
                        geometric_regressions.append(dict(frame=after['frame'],diagonal=diagonal,
                            kind=scope,baseline_pairs=a.get(scope,0),treatment_pairs=b.get(scope,0)))
        gates=dict(angular_roughness_reduced_both_hands=all(
            current['orientation']['hands'][s]['spatial_angular_acceleration_rms_rad_s2']
            <baseline['orientation']['hands'][s]['spatial_angular_acceleration_rms_rad_s2'] for s in ('R','L')),
            original_onsets_not_worse=m['onset_passes']>=baseline['metrics']['onset_passes'],
            original_gate_coverage_not_worse=m['hold_coverage']>=baseline['metrics']['hold_coverage']-1e-12,
            sampled_internal_joint_bounds_retained=not current['internal_metrics']['anatomy_violations']
                and current['internal_metrics']['maximum_joint_speed_rad_s']<=12.+1e-8,
            actual_skin_and_active_support_not_worse=not geometric_regressions)
        result[name]=dict(gates=gates,passed=bool(name!='legacy' and all(gates.values())),
            geometric_regressions=geometric_regressions,
            scope='local diagnostic nonregression; baseline itself fails geometry and physical music, no rig acceptance',
            comparison_depth_tolerance_mm=.002)
    return result


def main():
    path=ROOT/'lab/realism_results/orientation_20261011.json'
    report=json.loads(path.read_text())
    assert report['runs']['status']=='complete' and report['runs']['code_unchanged_during_run']
    for name,record in report['records'].items():
        assert sha(Path(record['source']))==record['sha256']
    reproduction=report['records']['legacy']['provenance']['source_baseline_reproduction']
    assert reproduction['visible_frames_identical'] and reproduction['physical_warmup_frames_identical']
    audits={}
    for name in TREATMENTS:
        audit=json.loads((ART/f'orientation_actual_{name}/audit.json').read_text())
        assert audit['status']=='complete' and audit['source_unchanged_during_run']
        assert [r['frame'] for r in audit['records']]==[0,95,96]
        audits[name]=audit
    render=json.loads((ART/'orientation_render.json').read_text())
    video=Path(render['video'])
    assert sha(video)==render['video_sha256'] and render['source_unchanged_during_run']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(video)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],int(stream['nb_read_frames']))==(1024,576,48)
    assert stream['avg_frame_rate']=='12/1' and abs(float(stream['duration'])-4.)<.01
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    visual=json.loads((ART/'orientation_visual_review.json').read_text())
    assert visual['reviewed']
    log=ART/'orientation_tests.log'
    count=re.search(r'Ran (\d+) tests',log.read_text())
    assert count and log.read_text().rstrip().endswith('OK')
    review_path=ROOT/'lab/realism_results/orientation_review_result.json'
    episode_path=ROOT/'lab/realism_results/orientation_review_episode.json'
    review=json.loads(review_path.read_text())
    episode=json.loads(episode_path.read_text())
    assert review['status']=='completed' and episode['termination']['phase']=='done'
    assert {t['role'] for t in episode['traces']}=={'synthesize','verify'}
    initial_episode_path=ROOT/'lab/realism_results/orientation_review_initial_episode.json'
    initial_episode=json.loads(initial_episode_path.read_text())
    assert initial_episode['termination']['phase']=='failed'
    assert {'evidence','counterevidence','constraints'} <= {t['role'] for t in initial_episode['traces']}
    plots=json.loads((ART/'orientation_plot.json').read_text())
    assert plots['status']=='complete' and plots['source_unchanged_during_run']
    recovery=json.loads((ART/'orientation_initial_recovery.json').read_text())
    score=json.loads((ART/'orientation_score_semantics.json').read_text())
    math_domain=json.loads((ART/'orientation_math_domain_and_budget.json').read_text())
    report.update(status='complete',actual_mesh_audit_pending=False,actual_mesh_audit=audits,
        render=dict(render,verified=True,ffprobe=probe,full_decode_errors=0,visual_review=visual),
        plots=plots,unit_tests=dict(passed=int(count[1]),log=str(log),sha256=sha(log)),
        native_research=dict(result=str(review_path),result_sha256=sha(review_path),
            episode=str(episode_path),episode_sha256=sha(episode_path),episode_id=episode['episode_id'],
            completed_nodes=2,initial_graph=dict(episode_id=initial_episode['episode_id'],
                source=str(initial_episode_path),sha256=sha(initial_episode_path),
                status='failed after review-gate rejection and synthesis-retry deadline',
                evidence_roles_with_answers=['evidence','counterevidence','constraints']),
            recovery_scope='Standalone synthesis/review of the supplied brief; prior role outputs are hashed, not reused',
            verdict_scope='Graph completion metadata only; synthesis/reviewer text is not exposed. '
                'The final CLI answer is an independent methodological review of the supplied brief, '
                'not an exposed reviewer approval or empirical rig acceptance'),
        implementation_recovery=recovery,original_score_semantics=score,
        math_domain_and_budget=math_domain,diagnostic_nonregression=diagnostic_comparisons(report,audits),
        default_adoption=False,full_performance_accepted=False,
        conclusion='Absolute wrist history and unaliased forearm direction are matched kinematic factors. '
            'All musical/skin regressions remain visible; no full rig or default controller adoption is accepted. '
            'SO3 combines history transport and rotation smoothing, requiring separate follow-up controls.')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',tests=int(count[1]),video_sha256=sha(video),
        episode_id=episode['episode_id'],full_performance_accepted=False)),flush=True)


if __name__=='__main__':
    main()

"""Seal original music, matched budgets, actual surfaces and reviewed media."""
import json
from pathlib import Path
import re
import subprocess
from physical_key_trial import ROOT,ART,TREATMENTS,sha


def skin_summary(audit):
    records=[]
    for r in audit['records']:
        support=[x for values in r['assigned_support'].values() for x in values if x['activation']>=.7]
        records.append(dict(frame=r['frame'],surface_intersection_clear=r['surface_intersection_clear'],
            scope_counts=r['scope_counts'],instrument_depth_mm={s:r['instrument'][s]['maximum_penetration_mm'] for s in ('R','L')},
            instrument_triangle_intersections={s:r['instrument'][s]['triangle_intersections'] for s in ('R','L')},
            instrument_clear=all(r['instrument'][s]['instrument_clear'] for s in ('R','L')),
            strong_support_required=len(support),strong_support_missing=sum(not x['supported'] for x in support),
            lbs_parity_maximum_error_mm=r['lbs_parity_maximum_error_mm']))
    initial=records[0]
    return dict(records=records,initial_feasible=initial['surface_intersection_clear']
        and initial['instrument_clear'] and initial['strong_support_missing']==0,
        all_sampled_surfaces_clear=all(r['surface_intersection_clear'] and r['instrument_clear'] for r in records),
        scope='First physical pose plus two preregistered visible instants; no positive-clearance, containment or nonlinear trajectory certification')


def main():
    path=ROOT/'lab/realism_results/physical_keys_20261011.json'
    report=json.loads(path.read_text())
    assert report['runs']['status']=='complete' and report['runs']['code_unchanged_during_run']
    assert all(sha(ROOT/p)==h for p,h in report['runs']['input_sha256'].items())
    audits={};summaries={}
    for name,record in report['records'].items():
        assert sha(Path(record['source']))==record['sha256']
        assert record['provenance']['maximum_per_frame_hand_budget_unchanged']
        audit=json.loads((ART/f'physical_key_actual_{name}/audit.json').read_text())
        assert audit['status']=='complete' and audit['source_unchanged_during_run']
        assert [r['frame'] for r in audit['records']]==[0,138,139]
        audits[name]=audit;summaries[name]=skin_summary(audit)
        if name!='legacy':
            assert record['source_actions']['source_voices_scheduled_once'] and not record['post_plan_cuts_or_losses']
    render=json.loads((ART/'physical_key_render.json').read_text());video=Path(render['video'])
    assert sha(video)==render['video_sha256'] and render['source_unchanged_during_run']
    formatting_path=ART/'physical_key_postrun_formatting.json'
    formatting=json.loads(formatting_path.read_text())
    for entry in formatting['records']:
        assert entry['ast_identical'] and sha(Path(entry['executed_archive']))==entry['executed_sha256']
        assert sha(Path(entry['source']))==entry['current_sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(video)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],int(stream['nb_read_frames']))==(1536,288,48)
    assert stream['avg_frame_rate']=='12/1' and abs(float(stream['duration'])-4.)<.01
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr
    visual_path=ART/'physical_key_visual_review.json';visual=json.loads(visual_path.read_text())
    assert visual['reviewed'] and visual['video_sha256']==sha(video)
    plot=json.loads((ART/'physical_key_plot.json').read_text())
    assert plot['status']=='complete' and plot['source_unchanged_during_run']
    assert visual['plot_sha256']==plot['png_sha256']==sha(ART/'physical_key_traces.png')
    log=ART/'physical_key_tests.log';count=re.search(r'Ran (\d+) tests',log.read_text())
    assert count and log.read_text().rstrip().endswith('OK')
    control_path=ART/'physical_key_target_control.json';control=json.loads(control_path.read_text())
    assert control['status']=='complete' and len(control['records'])==4
    assert all(r['all_preexisting_fields_equal'] for r in control['records'])
    review_path=ROOT/'lab/realism_results/physical_keys_review_result.json'
    episode_path=ROOT/'lab/realism_results/physical_keys_review_episode.json'
    result=json.loads(review_path.read_text());episode=json.loads(episode_path.read_text())
    initial_camera_path=ART/'physical_key_render.initial_camera.json'
    initial_camera=json.loads(initial_camera_path.read_text())
    camera_archive=ART/'physical_key_render.initial_camera.py'
    assert sha(camera_archive)==initial_camera['input_sha256'][str(ROOT/'lab/physical_key_render.py')]
    assert sha(ART/'physical_key_comparison.initial_camera.mp4')==initial_camera['video_sha256']
    comparisons={}
    baseline=report['records']['legacy']
    for name,r in report['records'].items():
        comparisons[name]=dict(original_onsets_not_worse=r['metrics']['onset_passes']>=baseline['metrics']['onset_passes'],
            original_gate_coverage_not_worse=r['metrics']['hold_coverage']>=baseline['metrics']['hold_coverage']-1e-12,
            sampled_internal_bounds_retained=not r['internal_metrics']['anatomy_violations']
                and r['internal_metrics']['maximum_joint_speed_rad_s']<=12.+1e-8,
            initial_actual_surface_feasible=summaries[name]['initial_feasible'],
            actual_sampled_surfaces_clear=summaries[name]['all_sampled_surfaces_clear'],
            strong_sampled_support_retained=all(x['strong_support_missing']==0 for x in summaries[name]['records']),
            no_requested_interval_conflicts=r['requested_interval_conflicts']['total']==0,
            full_realism_accepted=False)
    report.update(status='complete',actual_mesh_audit_pending=False,actual_mesh_audit=audits,
        actual_mesh_summary=summaries,diagnostic_gates=comparisons,
        render=dict(render,verified=True,ffprobe=probe,full_decode_errors=0,visual_review=visual),
        plots=plot,unit_tests=dict(passed=int(count[1]),log=str(log),sha256=sha(log)),
        prepared_target_control=dict(control,source=str(control_path),sha256=sha(control_path)),
        original_unison_evidence=dict(source=str(ART/'physical_key_score_semantics.json'),
            sha256=sha(ART/'physical_key_score_semantics.json')),
        implementation_recovery=dict(initial_control_comparison=dict(
            log=str(ART/'physical_key_legacy_control.log'),sha256=sha(ART/'physical_key_legacy_control.log'),
            reason='New preserve_holds metadata was included in old hands comparison; corrected control omits lineage/policy metadata and retains all numerical fields'),
            post_run_source_formatting=dict(formatting,source=str(formatting_path),sha256=sha(formatting_path)),
            camera=dict(initial_metadata=str(initial_camera_path),initial_metadata_sha256=sha(initial_camera_path),
                initial_source_archive=str(camera_archive),initial_source_sha256=sha(camera_archive),
                initial_video=str(ART/'physical_key_comparison.initial_camera.mp4'),
                initial_video_sha256=initial_camera['video_sha256'],
                reason='Baseline right hand partly cropped; all cells rerendered with wider matched camera and unchanged numerical frames')),
        native_research=dict(status='method_rejected' if episode['termination']['phase']!='done' else 'completed',
            task_status=result['status'],episode_id=episode['episode_id'],phase=episode['termination']['phase'],
            result=str(review_path),result_sha256=sha(review_path),episode=str(episode_path),episode_sha256=sha(episode_path),
            rejection_reported_by_driver='Conditional cuts rely on unverified performed-source offsets and clock alignment',
            physical_offset_meaning_verified=False,numerical_clock_mapping_reproduced=True,
            conditional_arm_scope='Unsupported offset-semantics assumption, diagnostic hypotheses only; blocked for adoption',
            validated_conditional_releases=0,
            scope='Graph failed after two verifier rejections. Driver quotes its methodological objection; hidden node bodies are not independently inspected.'),
        original_source_note_count=2046,original_soundtrack_replaced=False,
        drummer_changed=False,physical_key_actions_validated=False,default_adoption=False,full_performance_accepted=False,
        conclusion='Source voices and experimental physical key actions are separated without deleting music. '
            'Conservative and pedal-conditioned requests retain their holds and expose conflicts. '
            'Actual initial/contact/skin and original-music results remain decisive; neither treatment is adopted by default.')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',tests=int(count[1]),skin=summaries,video_sha256=sha(video),
        full_performance_accepted=False)),flush=True)


if __name__=='__main__':main()

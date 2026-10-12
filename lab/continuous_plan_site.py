"""Display completed continuity ablation and retained failures."""


def apply_continuous_plan(section, report):
    assert report['status']=='complete' and report['render']['verified']
    rows = [['Treatment','Onsets','Original hold coverage','Maximum joint speed']]
    for name in ('windowed','coherent','warm','composed'):
        m = report['records'][name]['metrics']
        rows.append([name,f"{m['onset_passes']}/{m['original_score_onsets']}",
            f"{100*m['hold_coverage']:.1f}%",f"{m['maximum_joint_speed_rad_s']:.2f} rad/s"])
    a,b = (report['records'][n]['metrics']['chunk_boundaries'][0]['hands']['R'] for n in ('windowed','coherent'))
    worst = max(h['maximum_penetration_mm'] for audit in report['actual_mesh_audit'].values()
                for record in audit['records'] for h in record['instrument'].values())
    section['items'][0:0] = [
        dict(kind='video',src='media/rig_realism_continuous_plan_comparison.mp4',wide=True,
            title='Continuous motion: four matched controller treatments',
            note='Four seconds at the original tempo, score and hand size, in grayscale with original audio. '
                'Top left: independent segments; top right: one complete-score plan. '
                'Bottom left: actual physical warmup; bottom right: warmup plus restrained digits and wrist carry. '
                f"The right-wrist join jump drops from {a['wrist_displacement_mm']:.2f} to {b['wrist_displacement_mm']:.2f} mm. "
                'The one-plan exported windows rejoin with identical poses, keys, body and timestamps. '
                'Warmup adds 48 solved frames; total computation is unequal. '
                f"Eight independently audited skin snapshots still expose failures (up to {worst:.2f} mm instrument clipping). "
                f"{report['unit_tests']['passed']} tests pass. Full realism remains unfinished. "
                '<a href="media/rig_realism_continuous_plan_results.json">All treatments, original music, warmup and actual-skin evidence</a>.'),
        dict(kind='table',title='Continuity helps the join; physical music and skin still need work',rows=rows)]


def apply_full_suite(section, report):
    assert report['status']=='complete' and report['visual_review']['reviewed']
    section['items'].insert(0, dict(kind='video',src='media/rig_realism_full_suite_progress.mp4',wide=True,
        title='Entire Knights Suite: 6:34 progress preview, with original audio',
        note='The whole piece from its opening through the final audio decay, 512×288 grayscale at 12 fps and normal tempo. '
            'The minute from 4:00–5:00 uses the completed wrist-order controller. Other sections combine the earlier opening motion with the saved v19 performance. '
            'Opening pitches and note times match; its touch dynamics differ. The archive join and audio tail are disclosed in the source record. '
            'The updated minute keeps wrist order but still has skin clipping and seven solver joins. '
            'The four-second continuous-plan experiment below is separate. Older sections retain their crossing, curl and clipping failures. '
            '<a href="media/rig_realism_full_suite_preview_results.json">Preview sources, timing and limitations</a>.'))

"""Publish completed, rejected-or-unproven controller comparisons honestly."""


def apply_composure(section, report):
    assert report['status'] == 'complete' and not report['full_performance_accepted']
    assert report['render']['verified'] and len(report['actual_mesh_audit']) == 4
    assert all(a['status'] == 'complete' and len(a['records']) == 2
               for a in report['actual_mesh_audit'].values())
    baseline, both = report['records']['baseline'], report['records']['both']
    results = []
    for name, record in report['records'].items():
        results.append(f"{name}: {record['onset_passes']}/{record['original_score_onsets']} onsets; "
            f"{100*record['original_score_hold_coverage']:.1f}% of required held-key samples; "
            f"{len(record['changed_note_holds'])} shortened or missing scheduled holds")
    section['items'][0:0] = [
        dict(kind='video', src='media/rig_realism_composure_comparison.mp4', wide=True,
            title='Curl and wrist-control comparison: six seconds',
            note='Fresh grayscale comparison, fixed view, same passage and measured hand size 0.7284. '
                 'Left: current controller. Right: experimental curled free fingers and wrist-carried reach. '
                 'Both are regenerated; the one-minute preview below retains the older size-1 motion. '
                 'The experiment remains off by default: sampled skin checks and musical support still fail. '
                 '<a href="media/rig_realism_composure_results.json">All four runs, original note ledger, mesh checks and render identity</a>.'),
        dict(kind='image', src='media/rig_realism_composure_metrics.png', wide=True,
            title='Composure trial: every outcome retained',
            note='<br>'.join(results) + '.<br>Held-key coverage uses the original score. '
                 'The middle-finger comparison includes identical unused frames across all runs. '
                 'A calmer pose alone does not establish correct contact or collision-free skin. '
                 f"{report['unit_tests']['passed']} tests pass. The native research graph completed five roles; "
                 '<a href="media/rig_realism_composure_review.json">method review</a> is not rig acceptance.')]

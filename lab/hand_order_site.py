"""Publish completed no-crossing diagnostics on the existing band site."""


def apply_hand_order(section,report):
    assert report['status']=='complete' and report['render']['verified']
    assert report['minute']['metrics']['order']['passed']
    old,new=report['old_minute']['metrics'],report['minute']['metrics']
    failed_joins=sum(any(not h['joint_speed_bound_passed'] for h in j['hands'].values()) for j in new['chunk_boundaries'])
    baseline,ordered=(report['matched'][n]['metrics'] for n in ('baseline','ordered'))
    closest=report['actual_mesh_audit']['minute']['records'][0]
    closest_depth=max(h['maximum_penetration_mm'] for h in closest['instrument'].values())
    section['intro']='The latest controller enforces left/right wrist order and preserves the original score ledger. '
    section['intro']+='Fresh performance renders show the current limits: key support, finger skin and continuous motion still fail.'
    section['items'][0:0]=[
        dict(kind='video',src='media/rig_realism_ordered_piano_60s.mp4',wide=True,
            title='New minute: wrist order enforced, original score and audio',
            note=f"Fresh grayscale performance at the original size 1, 768×432, 12 fps. "
                 f"The earlier minute contains {len(old['order']['crossing_frames'])} crossed samples; "
                 f"the new 1,440 solved samples have zero crossings and a minimum wrist gap of {new['order']['minimum_gap_mm']:.1f} mm. "
                 'Full-score ownership respects held notes, the first-note approach is restrained, and roots are bounded before finger solving. '
                 'This correction isolates hand order; the curl/wrist-carry experiment remains separate below. '
                 f"Musical support and rendered-skin failures remain; {failed_joins}/7 chunk joins exceed the joint-speed gate. "
                 f"The closest-hands snapshot still clips the instrument by {closest_depth:.2f} mm. "
                 'Independent chunks preserve every solved pose and are not accepted as a continuous realistic performance. '
                 '<a href="media/rig_realism_hand_order_results.json">All matched outcomes, skin checks and chunk-boundary evidence</a>.'),
        dict(kind='video',src='media/rig_realism_hand_order_comparison.mp4',wide=True,
            title='Eight-second comparison: baseline left, ordered hands right',
            note='Matched fixed view and original size 1, with original performed audio. '
                 'Every frame is freshly solved and rendered. The right side removes wrist inversions; '
                 'Finger folds and missing physical key support are recorded in the independent skin audit.'),
        dict(kind='image',src='media/rig_realism_hand_order_metrics.png',wide=True,
            title='No-crossing regression: old minute and matched stress trial',
            note=f"The matched 287–295 s trial keeps the same score, hand size and complete controller budgets. "
                 f"Baseline: {baseline['onset_passes']}/{baseline['original_score_onsets']} onsets, "
                 f"{100*baseline['hold_coverage']:.1f}% held-key coverage. "
                 f"Ordered: {ordered['onset_passes']}/{ordered['original_score_onsets']} onsets, "
                 f"{100*ordered['hold_coverage']:.1f}% held-key coverage. "
                 f"{report['unit_tests']['passed']} tests pass. Root order is a separate gate from actual skin, body/costume, forces and continuous collision.")]

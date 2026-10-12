"""Publish complete geometry checks without turning static studies into a performance."""


def apply_skin_initialization(section, report):
    assert report['status'] == 'complete' and not report['default_adoption']
    second = report['second_phase']
    rows = [['Right-hand check', 'Previous witnesses', 'Complete cold start', 'Complete staged start']]
    for label, getter in (
        ('Full triangle-choice bucket', lambda r: r['actual']['hands']['R']['maximum_self_pair_bucket']),
        ('Key triangles intersecting', lambda r: r['actual']['hands']['R']['instrument']['triangle_intersections']),
        ('Required thumb supported', lambda r: 'yes' if all(a['supported'] for a in r['actual']['hands']['R']['assigned_support']) else 'no'),
        ('Independent motion bounds', lambda r: 'pass' if r['independent_motion_bounds']['passed'] else 'fail'),
        ('Local right-hand gate', lambda r: 'pass' if r['independently_verified_local_right_hand'] else 'fail')):
        rows.append([label] + [getter(second[mode]) for mode in ('uniform', 'complete', 'complete_seeded')])
    first_rows = [['Earlier initialization trial', 'R triangle-pair bucket', 'L triangle-pair bucket', 'L key penetration']]
    for cell in ('legacy_source', 'legacy_quiet', 'both_source', 'both_quiet'):
        actual = report['first_phase'][cell]['actual']['hands']
        first_rows.append([cell.replace('_', ' '), actual['R']['maximum_self_pair_bucket'],
            actual['L']['maximum_self_pair_bucket'], f"{actual['L']['instrument']['maximum_penetration_mm']:.3f} mm"])
    section['items'][0:0] = [
        dict(kind='image', src='media/rig_realism_skin_diagonal_comparison.png', wide=True,
            title='Latest: a stricter skin check and matched starting-pose repair',
            note='Actual production renders at the same 316.9-second instant. The original left hand remains unchanged in all three new trials and is still rejected. '
                'The comparison preserves key commands, source note duties, anatomy, next-pose bounds and solver budgets. '
                'The staged start adds a full preparation solve and is exploratory; equal per-stage budgets do not mean equal total work. '
                'These are static diagnostics, not a new continuous performance. '
                '<a href="media/rig_realism_skin_initialization_results.json">Every numerical result, full triangle audit, support check and source hash</a>.'),
        dict(kind='table', title='Independent production-mesh checks — matched right-hand repair', rows=rows),
        dict(kind='note', title='The renderer exposed a missed collision case',
            note='A numerical right-hand pose passed the earlier uniform-diagonal check, but Blender found three thumb folds. '
                'The largest crossing was 1.434 mm. CPU reconstruction confirmed all three: different faces can choose different quad diagonals. '
                'The corrected check includes these compatible choices and excludes only conflicting alternatives of the same face. '
                'Folded quads and unexpected shared-vertex crossings remain rejection cases. '
                'Triangle-pair buckets are diagnostic counts, not counts of physical collision events.'),
        dict(kind='image', src='media/rig_realism_skin_initialization_grid.png', wide=True,
            title='All four earlier initialization treatments — retained failures',
            note='Both hands were solved in each trial. Quiet-finger seeds and the earlier expanded witness set did not produce an independently accepted complete starting state. '
                'Every frozen production snapshot has been rechecked with complete diagonal-choice coverage. Historical results and executed sources remain archived.'),
        dict(kind='table', title='Complete recheck of the original four treatments', rows=first_rows),
        dict(kind='note', title='What this establishes and what remains',
            note=f"{report['unit_tests']['passed']} tests pass. {report['independent_right_hand_passes']}/3 new right-hand trials pass their independently checked local gate. "
                'The native research graph reviews the method, not empirical performance acceptance. '
                'Source-held pitches 56 and 64 lack assignments at this instant; the left hand, original next pose and continuous path remain unfinished. '
                'Source offset semantics and key/foot force mechanics are unverified. Natural finger restraint, whole-wrist reach, '
                'positive mesh distance, full body/costume clearance and drummer stress cases remain open. '
                'The complete 6:34 progress performance stays at the top of the page.')]

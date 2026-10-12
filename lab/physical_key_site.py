"""Publish source/physical-action differences alongside the full-suite preview."""


def apply_physical_keys(section,report):
    assert report['status']=='complete' and not report['default_adoption']
    records=report['records']
    rows=[['Measurement','Current','Conservative holds','Conditional releases']]
    for label,getter in (
        ('Original onset checks',lambda r:f"{r['metrics']['onset_passes']}/{r['metrics']['original_score_onsets']}"),
        ('Original gate samples supported',lambda r:f"{100*r['metrics']['hold_coverage']:.1f}%"),
        ('Proposed key holds supported',lambda r:f"{100*r['proposed_physical_hold_support']['coverage']:.1f}%"),
        ('Source voices represented',lambda r:r['source_actions']['scheduled_actions']),
        ('Physical action proposals',lambda r:r['source_actions']['physical_action_proposals']),
        ('Visible overlapping finger requests',lambda r:r['requested_interval_conflicts']['visible'])):
        rows.append([label]+[getter(records[n]) for n in ('legacy','conservative','conditional')])
    section['items'][0:0]=[
        dict(kind='video',src='media/rig_realism_physical_key_comparison.mp4',wide=True,
            title='Latest key-hold experiment: current / conservative / conditional pedal releases',
            note='Four seconds at normal tempo, around 5:19 in the suite, with the unchanged synthesized soundtrack. '
                'All three treatments compile the entire original score, retain a real warmup, and use the same solver budgets. '
                'Shared score unisons retain every source voice in the ledger; one physical strike is an explicit hypothesis '
                'and does not reproduce separate jittered audio attacks. '
                '<a href="media/rig_realism_physical_key_results.json">Complete music, motion, initial skin and contact results</a>.'),
        dict(kind='image',src='media/rig_realism_physical_key_traces.png',wide=True,
            title='Original music versus physical key requests, with independent rendered-skin checks',
            note='Strict scheduling retains requested holds and reports conflicts instead of quietly shortening them. '
                'Conditional cuts require uninterrupted recovered sustain and a nearby estimated pedal-up for every voice. '
                'These remain uncertain model-conditioned releases. Correct key depth and assignment are measured separately '
                'from actual skin contact. Initial/warmup geometry, both triangle diagonals and sampled finite supports remain visible.'),
        dict(kind='table',title='Matched passage — music and physical hold diagnostics',rows=rows),
        dict(kind='note',title='Conditional-release method rejected; no controller adoption',
            note=f"{report['unit_tests']['passed']} tests pass. Original source events and audio remain unchanged. "
                'The native reviewer rejected the conditional-release method: the physical meaning of transcribed offsets is unverified. '
                'The comparison retains those assumptions as diagnostic hypotheses; zero conditional releases are physically validated. '
                'Actual skin and support failures also block default adoption. '
                'Natural finger restraint, whole-wrist reach, full-body/costume clearance, key/foot force mechanics, '
                'continuous motion and drummer are still in progress. The full 6:34 performance stays at the top of this page.')]

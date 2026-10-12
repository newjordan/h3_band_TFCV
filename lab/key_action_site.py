"""Explain recovered pedal evidence and remaining original-music failures."""


def apply_key_actions(section,report):
    assert report['status']=='complete' and not report['physical_key_actions_validated']
    actions=report['source_actions'];pedal=report['estimated_pedal_release_audit'];sampler=report['sampler_damping_comparison']
    section['items'][0:0]=[
        dict(kind='image',src='media/rig_realism_key_action_semantics.png',wide=True,
            title='Recovered pedal estimates: audible sustain and finger holds need separate plans',
            note=f"The cached Chopin transcription contains 282 sustain-pedal events. "
                'The later note-only export discarded them. Recovery reproduces both original MIDI files byte for byte. '
                '280 events map inside the Chopin excerpt; one earlier event supplies its starting state and one later event stays outside the crop. '
                'These are model estimates, with separate note timing variation and no validated foot motion. '
                f"Every original attack now has a source record; {len(actions['missing_source_ids'])} attacks remain unscheduled. "
                'The current suite video and soundtrack are unchanged. '
                '<a href="media/rig_realism_key_action_results.json">Complete release audit, source hashes and uncertainty</a>.'),
        dict(kind='table',title='Whole-suite action audit — failures retained',rows=[
            ['Measurement','Result'],['Original source attacks',actions['source_attacks']],
            ['Scheduled action proposals',actions['scheduled_actions']],
            ['Shortened hold proposals',len(actions['shortened_source_ids'])],
            ['Covered by uninterrupted estimated Chopin pedal',pedal['covered']],
            ['Not covered by that estimate',pedal['failed']],
            ['Outside recovered pedal domain',pedal['unknown']],
            ['Proposed releases that change old sampler damping time',sampler['different']]]),
        dict(kind='note',title='Next: supported physical key releases, restrained fingers and pedal action',
            note='A pedal-down estimate does not identify when a finger actually released a key. '
                'The two omitted attacks at 2:00.56 and 2:08.52 are explicitly reported. '
                f"{report['unit_tests']['passed']} tests pass; prepared motion inputs match the preceding controller in three passages. "
                'The full-suite preview uses the original synthesized Salamander soundtrack; it is not the source Frank Levy recording. '
                'Natural finger form, actual skin clearance, whole-body/costume geometry, forces, piano action and drummer remain unfinished.')]

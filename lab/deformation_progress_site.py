"""Publish the newest full deformation/joint/contact experiments without promotion."""


def apply_deformation(section, report):
    trials = report['union_summary']
    passed = sum(t['accepted'] for t in trials.values())
    verdict = (f'{passed}/4 pass the local static gates only; connected motion and full realism remain unverified.'
               if passed else 'All four trials reject; no waypoint or animation is exported.')
    rows = [['Contact surface / initialization','CPU instrument depth','Instrument triangles','Self pairs','Supported anchors','Local result']]
    for name in ('fixed_source','fixed_lift','union_source','union_lift'):
        t = trials[name]
        rows.append([name.replace('_',' / '),f"{t['instrument_depth_mm']:.5f} mm",t['instrument_triangles'],
                     t['self_pairs'],f"{t['supported']}/{t['requested']}",'LOCAL ONLY' if t['accepted'] else 'REJECT'])
    items = [
        dict(kind='image',src='media/rig_realism_deformation_and_union.png',wide=True,
             title='Latest: identical-bone deformation, isolated joint changes and connected-key contact',
             note='Sixteen actual Blender mesh snapshots hold bone matrices, weights, size and topology fixed '
                  'within linear/quaternion pairs. The bind control has zero crossings at both levels. '
                  'Playing poses still fail with either method: full two-diagonal crossing totals are '
                  'fresh 195→171, contact correction 202→169, and form guidance 163→165. '
                  'These are triangle-pair counts, not numbers of folds. Switching deformation is not adopted. '
                  'A separate 15-point joint sweep and four actual Blender reconstructions isolate the posture tradeoff. '
                  '<a href="media/rig_realism_deformation_and_union_results.json">Every treatment, input identity and conclusion</a>.'),
        dict(kind='image',src='media/rig_realism_ring_uncurled.png',
             title='Same fixed camera: uncurling clears self folds but clips the keys',
             note='Only the right ring PIP changes to the calibrated comfort coordinate. Actual Blender '
                  'self crossings clear in that hand, while instrument penetration becomes 6.327 mm '
                  '(2426 triangle/solid contacts). Wrist, elbow and other joints remain fixed. '
                  'The left hand remains rejected too. This is a diagnostic pose, not an accepted performance.'),
        dict(kind='image',src='media/rig_realism_ring_overcurled.png',
             title='Unchanged overcurled control: instrument-clear right hand still folds through itself',
             note='Right-hand instrument triangles are clear, but its actual surface has up to 127 crossing '
                  'pairs across the two diagonal choices. The original ring PIP measure is about 92°. '
                  'The left thumb CMC grid shows the same tradeoff: selected uncurling clears left self crossings '
                  'but creates 6.139 mm instrument penetration. Clearing one constraint is insufficient.'),
        dict(kind='table',title='Matched connected-key-top × initialization pilot',rows=rows,
             note='The earlier soft anchors were confined to one head or stem component. A new opt-in '
                  'mode measures distance to the connected key’s physical outer boundary, so an internal '
                  'head/stem seam is not a forbidden gap. Notches and outer edges retain the 10 µm margin. '
                  'Finite exposed support, the 10–100 µm top gap, instrument triangles, self skin, finger '
                  'spacing, anatomy and exact arms remain hard. Both source and uncurl+8 mm lift initializations '
                  'use the same original objective and solver budget. Neighbor rates are absent for static planning. '+verdict),
        dict(kind='note', title='Independent Blender verification of the connected-key trials',
             note='Both rejected numerical trials reproduce their original solver results. Reconstructed with '
                  'the exact score-held key commands, the actual right-hand skin has zero instrument triangle '
                  'contacts in both cases, but up to 151 and 97 self crossing pairs. The unchanged left hand '
                  'still has 0.517 mm instrument penetration and up to 36 self crossing pairs. These are '
                  'static diagnostic reconstructions, with no pose or animation acceptance. The first '
                  'verification used mismatched source key depths; that failed protocol attempt is retained '
                  'separately and excluded from the matched results.'),
        dict(kind='note',title='Research result and remaining work',
             note=f"{report['unit_tests']['passed']} tests pass. The native five-node graph completed with one "
                  'synthesis/review retry (seven terminal traces); its PASS applies to the proposed method. '
                  'The reviewed connected planning × anatomically justified weights × deformation comparison '
                  'has not been completed. Current evidence supports jointly planning posture, wrist movement '
                  'and finite contact patches rather than relying on a deformation switch or one-joint fix. '
                  'The unchanged legacy left input’s tiny rounded q-bound excess is retained in the ledger. '
                  'Complete music, body/costume/drummer geometry, positive-distance/containment checks, actual '
                  'force/action dynamics, continuous time and natural visual motion remain open. '
                  '<a href="media/rig_realism_deformation_review.json">Completed research method review</a>.')]
    # Retain the two published motion videos before the new diagnostic stills.
    section['items'] = section['items'][:2] + items + section['items'][2:]
    section['title'] = 'Rig realism: posture, skin deformation and connected key contact'
    section['intro'] = ('Latest experiments expose coupled posture/contact failures. Quaternion deformation '
        'does not clear the playing poses; isolated uncurling trades self folds for key penetration. '
        'Connected key-top contact removes an artificial head/stem boundary. '+verdict+
        ' Videos remain diagnostic baselines; complete realistic performance is unfinished.')

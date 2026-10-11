"""Latest band-site results, explicitly superseding narrower historical gates."""


def apply_latest(section, mesh):
    replacements = {
        'pass actual skin, finite exposed-key contact': 'passed instrument-skin checks, finite exposed-key contact',
        'Contact-only and posture-guided corrections pass their one-pose gates':
            'Contact-only and posture-guided corrections passed the earlier instrument/contact/proxy gates',
        'One accepted right-hand pose at frame 24.':
            'One right-hand pose at frame 24 passed the earlier instrument/contact/proxy checks. The new actual same-hand surface audit rejects it.',
        'The four held contacts and hard local gates remain satisfied':
            'The four held contacts and earlier instrument/bounds gates remain satisfied',
        'The hand still looks overcurled.':
            'The hand still looks overcurled, and the new actual same-hand mesh check rejects it.'}
    for item in section['items']:
        if not item['title'].startswith('Earlier'):
            item['title'] = 'Earlier checkpoint: ' + item['title'].removeprefix('Latest: ')
        for before, after in replacements.items():
            item['note'] = item.get('note', '').replace(before, after)
    section['title'] = 'Rig realism: stricter skin checks, static experiments and video samples'
    section['intro'] = ('Latest results remain rejected. Actual same-hand surface crossings are now a hard pose gate, '
        'and sampled paths must explicitly pass same/interhand checks. This supersedes earlier narrower local '
        'instrument/contact/proxy acceptance. Videos below expose the current motion; neither is a complete realism pass.')
    rows = [['Static planning treatment', 'Instrument depth', 'Instrument triangles', 'Self crossing pairs', 'Supported anchors', 'Result']]
    for name in ('hard_proxy', 'hard_mesh', 'soft_proxy', 'soft_mesh'):
        t = mesh['summaries'][name]
        rows.append([name.replace('_', ' + '), f"{t['instrument_depth_mm']:.5f} mm",
            t['instrument_triangle_intersections'], t['same_hand_triangle_pair_intersections'],
            f"{t['supported_anchors']}/4", 'REJECT'])
    section['items'] = [
        dict(kind='video', src='media/rig_realism_measured_piano_diagnostic.mp4',
             title='Piano video: fresh measured-size failing baseline — fixed camera, both hands',
             note='86 frames at 24 fps, 1280×720, hand scale 0.7284, unchanged original timing. '
                  '63/86 saved frames intersect the instrument (maximum 5.909 mm). '
                  'This shows the baseline, not an improved or accepted performance. '
                  'The new one-pose treatments all fail and produce no candidate animation. Silent diagnostic clip.'),
        dict(kind='video', src='media/rig_realism_drummer_fixed.mp4',
             title='Drummer video: changing stroke and grip activation — fixed camera proxy preview',
             note='10 demo impacts, 60 fps preview, independently checked at 240 Hz. Demo 10/10 and roll 16/16 '
                  'pass rigid-length/contact/proxy gates with 6 mm within-hand spacing. All 12 stress patterns '
                  'remain rejected. Colored hands are finger chains, not production skin. Full hand/body meshes, '
                  'drum shells, feet/pedals, forces and continuous time remain unverified. Silent diagnostic clip.'),
        dict(kind='note', title='New failure found: separated fingers can still have folded skin',
             note='Actual level-2 Blender triangles at frame 24 have 150 right-hand crossing pairs in the earlier '
                  'contact-only correction and 125 with posture guidance. Both retain 25 left-hand pairs in '
                  'the Blender BVH candidate audit. The CPU candidate search on the guided rendered '
                  'mesh finds all 150 plus one additional left shared-vertex crossing: 151 total. Its independently '
                  'verified intersection segment is 0.311 mm long; this is not penetration depth. Shared topology '
                  'is exempt only when contact is confined to its shared feature. Same-finger, palm and forearm '
                  'surfaces receive the checks. Counts are triangle pairs, not independent fold counts. '
                  'Zero interhand intersections at this one instant does not validate the connected path.'),
        dict(kind='image', src='media/rig_realism_mesh_waypoint_experiment.png', wide=True,
             title='Hypothesis → ablation → result: soft depth guidance helps key clearance; all four trials fail',
             note='Same R-frame-24 input, measured size, score/key commands, anatomy, source seed, 120 mm '
                  'static wrist workspace, human weight 4 and solver budget. Hard pooled median ±2 mm versus '
                  'soft pooled depth band, each with active mesh constraints off/on; every treatment retains '
                  'the same final actual mesh rejection gate. Soft guidance reduces instrument penetration '
                  'from about 3.04 mm to 0.0147 mm, but remaining instrument/proxy/self/contact failures reject '
                  'both soft trials. Mesh constraints do not resolve the folds in this budget. Neighbor motion '
                  'bounds are absent only for static waypoint planning; no waypoint or animation is exported. '
                  '<a href="media/rig_realism_mesh_waypoint_results.json">Every outcome, audit, hash and video identity</a>.'),
        dict(kind='table', title='All four matched static planning outcomes — no passing treatment', rows=rows),
        dict(kind='note', title='Control, research review and next decisive test',
             note=f"{mesh['unit_tests']['passed']} tests pass, including actual shared-vertex fold witnesses, "
                  'broad-phase completeness on varied triangle sizes, near-plane geometry and fail-closed pose/path '
                  'acceptance. The native five-node research graph completed; its PASS reviews the connected '
                  'experiment method, not these rigs. That connected comparison is still pending. Hard physical '
                  'limits remain active through preparation, impact, hold and release; activation currently '
                  'describes animation control, not measured muscle force. Next, distinguish pose-induced folds '
                  'from skin binding/deformation behavior on identical frozen poses, examine each joint angle, '
                  'then plan contact/wrist motion with original rate limits and the full musical ledger. '
                  'Positive surface distance, volume containment, full body/costume/drummer geometry, complete '
                  'music, force/action dynamics, continuous collision detection and naturalness remain open. '
                  '<a href="media/rig_realism_form_mesh_review.json">Completed method review</a>.')
    ] + section['items']

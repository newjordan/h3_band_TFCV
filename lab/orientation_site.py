"""Display complete orientation comparison without adopting failed motion."""


def apply_orientation(section,report):
    assert report['status']=='complete' and report['render']['verified']
    rows=[['Treatment','Original onsets','Original held coverage','Maximum wrist rotation speed']]
    for name in ('legacy','so3','forearm','combined'):
        record=report['records'][name]
        m=record['metrics']
        angular=max(h['proper_rotation_maximum_speed_rad_s'] for h in record['orientation']['hands'].values())
        rows.append([name,f"{m['onset_passes']}/{m['original_score_onsets']}",
            f"{100*m['hold_coverage']:.1f}%",f'{angular:.2f} rad/s'])
    worst=max(h['maximum_penetration_mm'] for audit in report['actual_mesh_audit'].values()
        for record in audit['records'] for h in record['instrument'].values())
    section['items'][0:0]=[
        dict(kind='video',src='media/rig_realism_orientation_comparison.mp4',wide=True,
            title='Wrist orientation: four matched controller treatments',
            note='Four seconds of the same score at normal tempo, with original audio. '
                'Top left: legacy; top right: absolute rotation history and smoothing. '
                'Bottom left: forearm angles that distinguish a backward-facing hand; bottom right: both changes. '
                'All use the same warmup duration and solve protocol, anatomy, hand size and original notes. '
                'The legacy trajectory reproduces the earlier source exactly. '
                'The rotation treatment adds an offline fit; total computation differs. '
                f"Twelve actual skin checks, including each initial warmup pose, still expose failures (up to {worst:.2f} mm instrument clipping). "
                f"{report['unit_tests']['passed']} tests pass. These controls remain experimental. "
                '<a href="media/rig_realism_orientation_results.json">Every treatment, source hash, musical error and actual skin check</a>.'),
        dict(kind='table',title='Orientation experiment — original music and measured motion',rows=rows),
        dict(kind='image',src='media/rig_realism_orientation_traces.png',wide=True,
            title='Whole solved history: wrist direction, rotation speed and inactive middle finger',
            note='Shaded opening is the physical warmup. Blank middle-finger segments are active or assigned frames. '
                'Direction and rest-posture measurements describe motion; they do not prove naturalness or force control. '
                'Actual skin, complete music, body/costume and drummer remain unfinished.')]

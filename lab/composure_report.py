"""Compare original note obligations and wrist-local composure, without accepting motion."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HAND_SCALE'] = '.7284'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from blockout.rig.hand import HandRig
from blockout.rig.composure import relaxed_pads
from blockout import piano


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    animation = json.loads(path.read_text())
    notes_file = path.with_suffix('.notes.json')
    notes = json.loads(notes_file.read_text())
    frames, start, fps = animation['frames'], animation['start'], animation['fps']
    times = start + np.arange(len(frames)) / fps
    onsets, holds, schedule_changes = [], [], []
    scheduled = [e for hand in animation['scheduled_note_events'].values() for events in hand.values() for e in events]
    for identity, (on, off, pitch, velocity, track) in enumerate(notes):
        if off <= times[0] or on > times[-1]:
            continue
        coverage = [(i, frame['keys'].get(str(pitch), 0.) >= .5) for i, frame in enumerate(frames)
                    if on + .04 <= times[i] < off - .02]
        if coverage:
            holds.append(dict(note=identity, pitch=pitch, start=on, end=off,
                required_samples=len(coverage), supported_samples=sum(ok for _, ok in coverage),
                first_missing_frame=next((i for i, ok in coverage if not ok), None)))
        if start <= on <= times[-1]:
            ids = np.flatnonzero((times >= on) & (times <= on + .25))
            ok = bool(len(ids) and max(frames[i]['keys'].get(str(pitch), 0.) for i in ids) >= .5)
            onsets.append(dict(note=identity, pitch=pitch, onset=on, passed=ok))
        matches = [e for e in scheduled if abs(e[0] - on) < 1e-6 and e[2] == pitch]
        effective = max((e[1] for e in matches), default=None)
        if effective is None or effective < off - 1e-6:
            schedule_changes.append(dict(note=identity, pitch=pitch, start=on, required_end=off,
                effective_end=effective, shortened_seconds=None if effective is None else off - effective))
    digits, anatomy, velocities, hover, arm_errors, pad_distances = {}, [], [], [], [], []
    for side in ('R', 'L'):
        rig = HandRig(side)
        series, local, wrists, abd = [], [], [], []
        middle = rig.chains[2]
        rest_pip = float(rig.q_rest[rig.sl[2] + middle.i_pip])
        for i, frame in enumerate(frames):
            hand = frame['hands'][side]
            rotation, position = np.array(hand['root']['rot']), np.array(hand['root']['pos'])
            shoulder = np.array(frame['body']['shoulders'][side])
            elbow = np.array(hand['elbow'])
            arm_errors.extend([abs(np.linalg.norm(elbow-shoulder)-piano.performer.UPPER_ARM),
                               abs(np.linalg.norm(position-elbow)-piano.performer.FOREARM)])
            pad_distances.extend(np.linalg.norm(np.diff(np.array(hand['pads']),axis=0),axis=1))
            wrists.append(position)
            q = np.array([hand['angles'][bone][dof] for ch in rig.chains for bone, dof, *_ in ch.dofs])
            if np.any(q < rig.lo - 1e-9) or np.any(q > rig.hi + 1e-9):
                anatomy.append(dict(side=side, frame=i))
            if i:
                velocities.append(float(np.max(np.abs(q - last_q)) * fps))
            last_q = q
            if hand['assigned_keys'][2] is None and hand['activation'][2] < .05:
                series.append((i, float(q[rig.sl[2] + middle.i_pip])))
                local.append((np.array(hand['pads'][2]) - position) @ rotation)
                relaxed = relaxed_pads(rig, rotation, np.array(hand['root']['plan']))[2]
                minimum = piano.surface_under(relaxed[0], relaxed[1], piano.TIP_R[2],
                    {int(k): v for k,v in frame['keys'].items()}) + .003
                hover.append(dict(side=side, frame=i, relaxed_reference_pad=relaxed.tolist(),
                    minimum_hover_z=float(minimum), required_reference_lift_mm=float(max(0.,minimum-relaxed[2])*1000.),
                    achieved_pad=hand['pads'][2],
                    scope='posthoc relaxed-reference and instrument floor, not exact prepared target'))
            for finger, pitch in enumerate(hand['assigned_keys']):
                if pitch is not None and hand['activation'][finger] >= .8:
                    ch = rig.chains[finger]
                    for d, (bone, dof, *_) in enumerate(ch.dofs):
                        if dof == 'abd':
                            abd.append(float(abs(q[rig.sl[finger] + d] - rig.q_rest[rig.sl[finger] + d])))
        wrists = np.asarray(wrists)
        values, local = np.array([v for _, v in series]), np.asarray(local)
        worst = None if not series else series[int(np.argmax(abs(values - rest_pip)))][0]
        digits[side] = dict(inactive_middle_samples=len(series), rest_middle_pip_deg=float(np.degrees(rest_pip)),
            inactive_middle_pip_mean_deg=None if not len(values) else float(np.degrees(values.mean())),
            inactive_middle_rest_deviation_rms_deg=None if not len(values) else float(np.degrees(np.sqrt(np.mean((values - rest_pip) ** 2)))),
            inactive_middle_wrist_local_excursion_mm=None if not len(local) else float(np.linalg.norm(np.ptp(local, axis=0)) * 1000.),
            worst_inactive_middle_frame=worst,
            active_abduction_rest_deviation_rms_deg=None if not abd else float(np.degrees(np.sqrt(np.mean(np.asarray(abd) ** 2)))),
            wrist_lateral_travel_mm=float(np.abs(np.diff(wrists[:, 0])).sum() * 1000.),
            wrist_max_speed_m_s=float(np.linalg.norm(np.diff(wrists, axis=0), axis=1).max() * fps))
    required = sum(h['required_samples'] for h in holds)
    supported = sum(h['supported_samples'] for h in holds)
    return dict(animation=str(path.resolve()), animation_sha256=sha(path), notes_sha256=sha(notes_file),
        provenance=animation['experiment_provenance'], control=animation['control_experiment'],
        fps=fps, frames=len(frames), start=start, duration_seconds=len(frames)/fps, hand_scale=animation['hand_scale'],
        original_score_onsets=len(onsets), onset_passes=sum(n['passed'] for n in onsets),
        original_score_required_hold_samples=required, original_score_supported_hold_samples=supported,
        original_score_hold_coverage= supported / required if required else None,
        changed_note_holds=schedule_changes, note_onsets=onsets, note_holds=holds,
        middle_and_wrist=digits, anatomy_violation_frames=anatomy,
        maximum_joint_speed_deg_s=float(np.degrees(max(velocities, default=0.))),
        joint_speed_bound_rad_s=12., joint_speed_bound_violated=max(velocities, default=0.) > 12.+1e-8,
        maximum_arm_length_error_mm=float(max(arm_errors,default=0.)*1000.),
        minimum_adjacent_pad_control_point_distance_mm=float(min(pad_distances,default=0.)*1000.),
        pad_distance_scope='control-point spacing, not actual positive skin clearance',
        hover_floor_diagnostics=hover,
        legacy_proxy_qa=animation['qa'], accepted=False,
        metric_scope='key-depth samples are solved pad/proxy diagnostics, not actual mesh support or forces')


def main():
    records = {name: summarize(ROOT / f'artifacts/realism/composure_{name}.anim.json')
               for name in ('baseline', 'free', 'wrist', 'both')}
    assert len({r['notes_sha256'] for r in records.values()}) == 1
    hashes = [r['provenance']['input_sha256'] for r in records.values()]
    assert all(h == hashes[0] for h in hashes)
    animations = {name: json.loads(Path(r['animation']).read_text()) for name, r in records.items()}
    common, solo = {}, {}
    for side in ('R', 'L'):
        rig = HandRig(side)
        rest = rig.q_rest[rig.sl[2] + rig.chains[2].i_pip]
        ids = [i for i in range(len(next(iter(animations.values()))['frames']))
               if all(a['frames'][i]['hands'][side]['assigned_keys'][2] is None
                      and a['frames'][i]['hands'][side]['activation'][2] < .05 for a in animations.values())]
        for name, animation in animations.items():
            values, local = [], []
            for i in ids:
                hand = animation['frames'][i]['hands'][side]
                values.append(hand['angles']['finger3-2']['flex'])
                local.append((np.array(hand['pads'][2]) - hand['root']['pos']) @ np.array(hand['root']['rot']))
            common.setdefault(name, {})[side] = dict(frames=ids, samples=len(ids),
                middle_rest_deviation_rms_deg=None if not values else float(np.degrees(np.sqrt(np.mean((np.array(values)-rest)**2)))),
                middle_wrist_local_excursion_mm=None if not local else float(np.linalg.norm(np.ptp(local,axis=0))*1000.))
        isolated = []
        for i in range(len(next(iter(animations.values()))['frames'])):
            hands = [a['frames'][i]['hands'][side] for a in animations.values()]
            assigned = [h['assigned_keys'] for h in hands]
            if all(keys == assigned[0] for keys in assigned) and sum(p is not None for p in assigned[0]) == 1:
                finger = next(f for f,p in enumerate(assigned[0]) if p is not None)
                if all(h['activation'][finger] >= .8 for h in hands):
                    isolated.append((i, finger))
        for name, animation in animations.items():
            offsets = []
            for i,finger in isolated:
                h=animation['frames'][i]['hands'][side]
                restpad=relaxed_pads(rig,np.array(h['root']['rot']),np.array(h['root']['pos']))[finger]
                offsets.append(abs(h['pads'][finger][0]-restpad[0])*1000.)
            solo.setdefault(name,{})[side]=dict(frames=[i for i,_ in isolated], samples=len(isolated),
                horizontal_finger_excursion_from_relaxed_pad_mean_mm=None if not offsets else float(np.mean(offsets)))
    report = dict(schema='piano-composure-factorial-results/v1', status='controller_metrics_complete',
        hypothesis='Wrist-carried curled free digits and isolated wrist translation improve composure while preserving notes.',
        records=records, common_inactive_middle=common, common_isolated_reach=solo,
        original_score_identical=True, frozen_code_identical=True,
        full_performance_accepted=False, actual_mesh_audit_pending=True,
        exclusions=['force dynamics', 'continuous collision', 'positive distance/containment', 'body/costume', 'whole score'])
    destination = ROOT / 'lab/realism_results/composure_20261011.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    for name, r in records.items():
        print(json.dumps(dict(trial=name, onset=f"{r['onset_passes']}/{r['original_score_onsets']}",
            hold=r['original_score_hold_coverage'], shortened=len(r['changed_note_holds']),
            digits=r['middle_and_wrist'])), flush=True)


if __name__ == '__main__':
    main()

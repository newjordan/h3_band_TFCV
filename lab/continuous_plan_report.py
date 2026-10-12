"""Original-score, digit-control and continuous-history diagnostics for every treatment."""
import copy
import json
import os
from pathlib import Path
import sys

os.environ['HAND_SCALE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'
sys.path.insert(0, str(ROOT))
import numpy as np
from blockout.rig.hand import HandRig
from blockout.rig.composure import relaxed_pads
from lab.hand_order_report import metrics, sha

TREATMENTS = ('windowed', 'coherent', 'warm', 'composed')


def digit_metrics(animation, common=None):
    result, distances = {}, []
    for side in ('R', 'L'):
        rig = HandRig(side)
        middle = rig.chains[2]
        rest = float(rig.q_rest[rig.sl[2]+middle.i_pip])
        deviations, curls, indices, abductions, solo_offsets = [], [], [], [], []
        for i, frame in enumerate(animation['frames']):
            hand = frame['hands'][side]
            q = np.array([hand['angles'][bone][dof] for ch in rig.chains for bone, dof, *_ in ch.dofs])
            pads = np.array(hand['pads'])
            distances.extend(np.linalg.norm(np.diff(pads, axis=0), axis=1))
            if hand['assigned_keys'][2] is None and hand['activation'][2] < .05 and (common is None or i in common[side]):
                indices.append(i)
                pip = float(q[rig.sl[2]+middle.i_pip])
                curls.append(pip)
                deviations.append(pip-rest)
            active = [f for f, p in enumerate(hand['assigned_keys']) if p is not None and hand['activation'][f] >= .8]
            for f in active:
                for d, (_, dof, *_) in enumerate(rig.chains[f].dofs):
                    if dof == 'abd':
                        abductions.append(float(q[rig.sl[f]+d]-rig.q_rest[rig.sl[f]+d]))
            # Assignment-isolated proxy only: original held-score conflicts are reported separately.
            if len(active) == 1 and sum(p is not None for p in hand['assigned_keys']) == 1:
                f = active[0]
                relaxed = relaxed_pads(rig, np.array(hand['root']['rot']), np.array(hand['root']['pos']))[f]
                solo_offsets.append(float(abs(pads[f,0]-relaxed[0])*1000.))
        result[side] = dict(inactive_middle_frames=indices, inactive_middle_samples=len(indices),
            rest_middle_pip_deg=float(np.degrees(rest)),
            inactive_middle_pip_mean_deg=float(np.degrees(np.mean(curls))) if curls else None,
            inactive_middle_rest_deviation_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(deviations))))) if deviations else None,
            active_abduction_rest_deviation_rms_deg=float(np.degrees(np.sqrt(np.mean(np.square(abductions))))) if abductions else None,
            assignment_isolated_samples=len(solo_offsets),
            assignment_isolated_horizontal_excursion_mean_mm=float(np.mean(solo_offsets)) if solo_offsets else None)
    return dict(hands=result, minimum_adjacent_pad_control_point_distance_mm=float(min(distances)) * 1000.,
        scope='soft rest-reference/assignment diagnostics; not naturalness, tendon forces or positive skin clearance')


def changed_holds(animation, notes):
    schedules = [p['scheduled_note_events'] for p in animation.get('experiment_parts', [animation])]
    events = [e for schedule in schedules for hand in schedule.values() for ev in hand.values() for e in ev]
    first, last = animation['start'], animation['start'] + animation['frames'][-1]['t']
    result = []
    for identity, (on, off, pitch, _, _) in enumerate(notes):
        if off <= first or on > last:
            continue
        matches = [e for e in events if abs(e[0]-on) < 1e-6 and e[2] == pitch]
        effective = max((e[1] for e in matches), default=None)
        if effective is None or effective < off-1e-6:
            result.append(dict(note=identity, pitch=pitch, onset=on, original_release=off,
                scheduled_release=effective, missing=effective is None))
    return result


def main():
    runs = json.loads((ART/'continuous_plan_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    animations = {name: json.loads((ART/f'continuous_plan_{name}.anim.json').read_text()) for name in TREATMENTS}
    notes = json.loads((ART/'performance_source_v19/notes.json').read_text())
    common = {side: set.intersection(*[set(i for i, f in enumerate(a['frames'])
        if f['hands'][side]['assigned_keys'][2] is None and f['hands'][side]['activation'][2] < .05)
        for a in animations.values()]) for side in ('R', 'L')}
    records = {}
    for name, animation in animations.items():
        path = ART/f'continuous_plan_{name}.anim.json'
        assert json.loads(path.with_suffix('.notes.json').read_text()) == notes
        record = dict(source=str(path), sha256=sha(path), metrics=metrics(animation, notes, [48]),
            digit_control=digit_metrics(animation), common_inactive_middle=digit_metrics(animation, common),
            changed_original_holds=changed_holds(animation, notes),
            provenance=animation['experiment_provenance'], continuity=animation['continuity'])
        warmup = animation.get('warmup', {}).get('frames', [])
        if warmup:
            internal = copy.deepcopy(animation)
            internal['start'] -= 2.
            internal['frames'] = copy.deepcopy(warmup + animation['frames'])
            for f in internal['frames']:
                f['t'] += 2.
            record['internal_history_metrics'] = metrics(internal, notes, [48, 96])
            record['warmup_geometry_scope'] = 'All warmup poses retained; actual warmup skin not audited, no feasibility claim'
        records[name] = record
    result = dict(schema='piano-continuous-plan-results/v1', status='controller_complete', runs=runs,
        records=records, common_inactive_middle_indices={s: sorted(v) for s,v in common.items()},
        per_frame_budget_unchanged=True, equal_total_work=False, actual_mesh_audit_pending=True,
        full_performance_accepted=False, candidate_exported=False,
        limitations=['joint-speed at sampled frames is not nonlinear CCD', 'initial skin infeasibility',
            'actual warmup geometry unverified', 'body/costume/pedals', 'physical force/action',
            'original-score scheduling conflicts', 'full performance and drummer remain unaccepted'])
    path = ROOT/'lab/realism_results/continuous_plan_20261011.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({n: {k: r['metrics'][k] for k in ('onset_passes','original_score_onsets','hold_coverage',
        'maximum_joint_speed_rad_s','maximum_wrist_speed_m_s','maximum_wrist_rotation_speed_rad_s')}
        for n,r in records.items()}, indent=2))


if __name__ == '__main__':
    main()

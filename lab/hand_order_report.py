"""Stitch unaltered solves and report order, music, bounds and chunk joins."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HAND_SCALE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/realism'
sys.path.insert(0, str(ROOT))
import numpy as np
from blockout import performer
from blockout.rig.hand import HandRig
from blockout.rig.hand_order import wrist_order_audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(animation, notes, boundaries=()):
    frames, fps = animation['frames'], animation['fps']
    times = animation['start'] + np.array([f['t'] for f in frames])
    mcp_gaps = np.array([np.asarray(f['hands']['R']['fingers'])[1:,0,0].mean()
                        - np.asarray(f['hands']['L']['fingers'])[1:,0,0].mean() for f in frames])
    onsets, holds = [], []
    for identity, (on, off, pitch, velocity, track) in enumerate(notes):
        ids = np.flatnonzero((times >= on + .04) & (times < off - .02))
        if len(ids):
            supported = [frames[i]['keys'].get(str(pitch), 0.) >= .5 for i in ids]
            holds.append(dict(note=identity, pitch=pitch, start=on, end=off, samples=len(ids),
                supported=sum(supported), first_missing=next((int(i) for i, ok in zip(ids, supported) if not ok), None)))
        if times[0] <= on <= times[-1]:
            ids = np.flatnonzero((times >= on) & (times <= on + .25))
            supported = bool(len(ids) and max(frames[i]['keys'].get(str(pitch),0.) for i in ids) >= .5)
            onsets.append(dict(note=identity, pitch=pitch, onset=on, passed=supported,
                               complete_observation_window=bool(times[-1] >= on+.25)))
    anatomy, joint_speed, root_speed, root_rotation_speed, arm_errors = [], [], [], [], []
    full_precision = animation.get('pose_precision') == 'full solver precision'
    position_tolerance = 1e-9 if full_precision else 1e-5
    joint_tolerance = 1e-8 if full_precision else 1e-5
    joins = {int(i): dict(frame=int(i), absolute_time=float(times[i]), hands={}) for i in boundaries}
    for side in ('R','L'):
        rig = HandRig(side)
        previous = None
        for i, frame in enumerate(frames):
            hand = frame['hands'][side]
            q = np.array([hand['angles'][bone][dof] for ch in rig.chains for bone,dof,*_ in ch.dofs])
            position, rotation = np.array(hand['root']['pos']), np.array(hand['root']['rot'])
            if np.any(q < rig.lo - joint_tolerance) or np.any(q > rig.hi + joint_tolerance):
                anatomy.append(dict(frame=i, hand=side))
            np.testing.assert_allclose(position, hand['wrist'], atol=position_tolerance, rtol=0.)
            shoulder, elbow = np.array(frame['body']['shoulders'][side]), np.array(hand['elbow'])
            arm_errors += [abs(np.linalg.norm(elbow-shoulder)-performer.UPPER_ARM),
                           abs(np.linalg.norm(position-elbow)-performer.FOREARM)]
            if previous is not None:
                dt = times[i]-times[i-1]
                q_speed = float(np.max(abs(q-previous[0]))/dt)
                wrist_speed = float(np.linalg.norm(position-previous[1])/dt)
                angular = float(np.arccos(np.clip((np.trace(rotation @ previous[2].T)-1)/2,-1.,1.))/dt)
                joint_speed.append(q_speed); root_speed.append(wrist_speed); root_rotation_speed.append(angular)
                if i in joins:
                    joins[i]['hands'][side] = dict(maximum_joint_speed_rad_s=q_speed,
                        joint_speed_bound_passed=q_speed <= 12.+1e-8, wrist_speed_m_s=wrist_speed,
                        wrist_displacement_mm=float(np.linalg.norm(position-previous[1])*1000.),
                        wrist_rotation_speed_rad_s=angular)
            previous = (q, position, rotation)
    required, supported = sum(h['samples'] for h in holds), sum(h['supported'] for h in holds)
    return dict(order=wrist_order_audit(frames,fps), frames=len(frames), fps=fps,
        mcp_centre_minimum_gap_mm=float(mcp_gaps.min()*1000.),
        mcp_centre_crossing_frames=np.flatnonzero(mcp_gaps<0.).tolist(),
        mcp_scope='Mean of four long-finger MCP heads; skeleton landmarks, not physical skin clearance',
        start=animation['start'], duration_seconds=len(frames)/fps,
        original_score_onsets=len(onsets), onset_passes=sum(n['passed'] for n in onsets),
        censored_onset_windows=sum(not n['complete_observation_window'] for n in onsets),
        original_score_hold_samples=required, supported_hold_samples=supported,
        hold_coverage=supported/required if required else None, onsets=onsets, holds=holds,
        anatomy_violations=anatomy, maximum_joint_speed_rad_s=max(joint_speed,default=0.),
        joint_speed_bound_rad_s=12., joint_speed_bound_passed=max(joint_speed,default=0.)<=12.+1e-8,
        maximum_wrist_speed_m_s=max(root_speed,default=0.),
        maximum_wrist_rotation_speed_rad_s=max(root_rotation_speed,default=0.),
        maximum_arm_length_error_mm=float(max(arm_errors,default=0.)*1000.),
        recorded_pose_precision=animation.get('pose_precision','legacy rounded pose'),
        wrist_identity_tolerance_m=position_tolerance, anatomy_tolerance_rad=joint_tolerance,
        chunk_boundaries=list(joins.values()),
        metric_scope='Original-score solved key-depth diagnostics and sampled skeleton bounds; independent actual skin required')


def main():
    runs = json.loads((ART/'hand_order_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    parts = [json.loads((ART/f'hand_order_minute_{i}.anim.json').read_text()) for i in range(8)]
    assert all(p['hand_scale']==1. and p['fps']==24 and len(p['frames'])==180 for p in parts)
    assert all(p['hand_order']['ownership']==parts[0]['hand_order']['ownership'] for p in parts)
    assert all(p['experiment_provenance']['input_sha256']==parts[0]['experiment_provenance']['input_sha256'] for p in parts)
    full = copy.deepcopy(parts[0])
    full['start'] = 239.9
    full['frames'] = []
    for part in parts:
        for frame in part['frames']:
            changed = copy.deepcopy(frame)
            changed['t'] = part['start']-full['start']+frame['t']
            assert {k:v for k,v in changed.items() if k!='t'} == {k:v for k,v in frame.items() if k!='t'}
            full['frames'].append(changed)
    assert len(full['frames'])==1440
    np.testing.assert_allclose([f['t'] for f in full['frames']], np.arange(1440)/24, atol=1e-10,rtol=0.)
    full['qa'] = dict(scope='Independent chunks; complete stitched audits recorded separately',
                      chunks=[p['qa'] for p in parts])
    full.pop('scheduled_note_events',None)
    full.pop('fingering',None)
    full['chunk_provenance'] = [dict(source=str(ART/f'hand_order_minute_{i}.anim.json'),
        sha256=sha(ART/f'hand_order_minute_{i}.anim.json'), start=p['start'], frames=len(p['frames'])) for i,p in enumerate(parts)]
    full['hand_order']['audit'] = wrist_order_audit(full['frames'],24)
    full['experiment_provenance']['scope'] = 'Unaltered independently solved chunks; boundaries audited, no trajectory acceptance'
    full_path=ART/'hand_order_minute.anim.json'
    full_path.write_text(json.dumps(full)+'\n')
    notes=json.loads((ART/'performance_source_v19/notes.json').read_text())
    matched={}
    for trial in ('baseline','ordered'):
        path=ART/f'hand_order_{trial}.anim.json'
        a=json.loads(path.read_text())
        assert json.loads(path.with_suffix('.notes.json').read_text())==notes
        assert a['experiment_provenance']['input_sha256']==parts[0]['experiment_provenance']['input_sha256']
        matched[trial]=dict(animation=str(path),sha256=sha(path), metrics=metrics(a,notes),
            original_note_events=a['hand_order']['ownership'],scheduled_note_events=a['scheduled_note_events'],
            proxy_qa=a['qa'])
    old=json.loads((ART/'performance_60s.anim.json').read_text())
    report=dict(schema='piano-hand-order-results/v1',status='controller_metrics_complete',
        requirement='No hand crossing in this performance',hypothesis='Persistent ordered ownership and bounded wrist planning prevent inversions.',
        runs=runs,matched=matched,old_minute=dict(animation=str(ART/'performance_60s.anim.json'),
            sha256=sha(ART/'performance_60s.anim.json'),metrics=metrics(old,notes)),
        minute=dict(animation=str(full_path),sha256=sha(full_path), metrics=metrics(full,notes,range(180,1440,180)),
            sources=full['chunk_provenance'], original_pose_key_body_data_preserved=True,
            ownership=parts[0]['hand_order']['ownership']),
        original_score_identical=True, hand_scale=1., free_control=False, wrist_carry=False,
        ordering_default_enabled=True, full_performance_accepted=False, actual_mesh_audit_pending=True,
        exclusions=['positive skin distance','containment','full body/costume','forces/action','nonlinear CCD','all music'])
    path=ROOT/'lab/realism_results/hand_order_20261011.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for label,item in [('old',report['old_minute']),*matched.items(),('new',report['minute'])]:
        m=item['metrics']
        print(json.dumps(dict(trial=label,order=m['order'],onsets=f"{m['onset_passes']}/{m['original_score_onsets']}",
            hold=m['hold_coverage'],maximum_joint_speed=m['maximum_joint_speed_rad_s'],
            joins=m['chunk_boundaries'])),flush=True)


if __name__=='__main__':
    main()

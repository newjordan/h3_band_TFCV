"""All-sample physical direction/musical diagnostics; actual skin is separate."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/realism'
sys.path.insert(0,str(ROOT))
from lab.orientation_trial import TREATMENTS,sha
from lab.continuous_plan_report import digit_metrics,changed_holds
from lab.hand_order_report import metrics
from blockout.rig.hand import HandRig
from blockout.rig.orientation import split_reference_rotations,forearm_angles
from scipy.spatial.transform import Rotation
import numpy as np


def direction_metrics(animation):
    hands={}
    for side in ('R','L'):
        rig=HandRig(side)
        alignment,flexion,deviation,orthogonality,increments=[],[],[],[],[]
        previous=None
        for frame in animation['frames']:
            hand=frame['hands'][side]
            rotation=np.asarray(hand['root']['rot'])
            proper,_=split_reference_rotations(rotation)
            rh=rotation @ rig.sk.R0['wrist'].T
            across,forward,up=(rh @ v for v in rig.M_rest)
            forward/=np.linalg.norm(forward)
            arm=np.asarray(hand['root']['pos'])-hand['elbow']
            arm/=np.linalg.norm(arm)
            alignment.append(float(np.arccos(np.clip(arm @ forward,-1,1))))
            flex,dev=forearm_angles(arm,across,forward,up)
            flexion.append(flex);deviation.append(dev)
            orthogonality.append(float(np.max(np.abs(rotation @ rotation.T-np.eye(3)))))
            if previous is not None:
                delta=Rotation.from_matrix(proper @ previous[0].T).as_rotvec()/(frame['t']-previous[1])
                increments.append(delta)
            previous=(proper,frame['t'])
        acceleration=np.diff(increments,axis=0)*animation['fps']
        hands[side]=dict(samples=len(alignment),
            proper_rotation_maximum_speed_rad_s=float(max(np.linalg.norm(v) for v in increments)),
            forward_forearm_angle_mean_deg=float(np.degrees(np.mean(alignment))),
            forward_forearm_angle_p95_deg=float(np.degrees(np.percentile(alignment,95))),
            forward_forearm_angle_maximum_deg=float(np.degrees(max(alignment))),
            backwards_samples=sum(a>np.pi/2 for a in alignment),
            absolute_direction_flexion_maximum_deg=float(np.degrees(max(np.abs(flexion)))),
            absolute_direction_deviation_maximum_deg=float(np.degrees(max(np.abs(deviation)))),
            rotation_orthogonality_maximum_error=max(orthogonality),
            spatial_angular_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.sum(acceleration**2,axis=1)))),
            spatial_angular_acceleration_maximum_rad_s2=float(np.max(np.linalg.norm(acceleration,axis=1))))
    return dict(hands=hands,scope='unaliased hand-forward/forearm direction and sampled spatial increments; not clinical wrist angles, naturalness or hard motion limits')


def main():
    runs=json.loads((ART/'orientation_runs.json').read_text())
    assert runs['status']=='complete' and runs['code_unchanged_during_run']
    animations={n:json.loads((ART/f'orientation_{n}.anim.json').read_text()) for n in TREATMENTS}
    notes=json.loads((ART/'performance_source_v19/notes.json').read_text())
    common={s:set.intersection(*[set(i for i,f in enumerate(a['frames'])
        if f['hands'][s]['assigned_keys'][2] is None and f['hands'][s]['activation'][2]<.05)
        for a in animations.values()]) for s in ('R','L')}
    records={}
    for name,animation in animations.items():
        path=ART/f'orientation_{name}.anim.json'
        assert json.loads(path.with_suffix('.notes.json').read_text())==notes
        internal=copy.deepcopy(animation)
        internal['start']-=2.
        internal['frames']=copy.deepcopy(animation['warmup']['frames']+animation['frames'])
        for frame in internal['frames']:
            frame['t']+=2.
        records[name]=dict(source=str(path),sha256=sha(path),metrics=metrics(animation,notes,[48]),
            orientation=direction_metrics(animation),internal_metrics=metrics(internal,notes,[48,96]),
            internal_orientation=direction_metrics(internal),
            digit_control=digit_metrics(animation),common_inactive_middle=digit_metrics(animation,common),
            changed_original_holds=changed_holds(animation,notes),
            provenance=animation['experiment_provenance'],controller=animation['orientation_experiment'])
    report=dict(schema='piano-wrist-orientation-results/v1',status='controller_complete',runs=runs,records=records,
        common_inactive_middle_indices={s:sorted(v) for s,v in common.items()},
        actual_mesh_audit_pending=True,candidate_exported=False,full_performance_accepted=False,
        limitations=['SO3 bundles reference transport and rotation filtering','extra offline SO3 fit work',
            'sampled orientation changes are not hard bounds or continuous collision proof',
            'initial/warmup skin feasibility unverified','full music/body/costume/forces/action and drummer incomplete'])
    (ROOT/'lab/realism_results/orientation_20261011.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:dict(onsets=r['metrics']['onset_passes'],held=r['metrics']['hold_coverage'],
        angular_speed=r['metrics']['maximum_wrist_rotation_speed_rad_s'],direction=r['orientation']['hands'])
        for n,r in records.items()},indent=2))


if __name__=='__main__':
    main()

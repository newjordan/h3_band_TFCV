"""Frozen one-joint/CMC-grid diagnosis; no waypoint, path or candidate export."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'artifacts/realism/human_pilot_form.render.anim.json'
OUTPUT = ROOT / 'artifacts/realism/pose_fold_sweep_20261011.json'
animation = json.loads(SOURCE.read_text())
os.environ['HAND_SCALE'] = str(animation['hand_scale'])
import numpy as np
from blockout.rig.hand import HandRig
from blockout.rig.skin_contact import pose_vector
from blockout.rig.human_form import PianistFormPrior
from blockout.skin import Skin
from blockout.subdivision import catmull_clark
from blockout.mesh_self import quad_self_intersections
from blockout.piano_geometry import scene_from_animation
from blockout.skin_contacts import key_support
from lab.skin_audit import hand_audit, quad_triangles

files = ('lab/pose_fold_sweep.py', 'blockout/rig/hand.py', 'blockout/rig/skeleton.py',
    'blockout/rig/human_form.py', 'blockout/rig/pianist_reference.json', 'blockout/human_score.py',
    'blockout/skin.py', 'blockout/subdivision.py', 'blockout/mesh_self.py', 'blockout/triangle_contact.py',
    'blockout/skin_contacts.py', 'blockout/piano_geometry.py', 'blockout/key_shapes.py',
    'lab/skin_audit.py', 'blockout/hand_model/skin_bind.npz', 'blockout/hand_model/skeleton.json',
    'blockout/rig/skin_contact.py', 'blockout/rig/comfort.json', 'blockout/rig/furelise_cal.json',
    'blockout/human_ref.json')
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
report = dict(schema='pose-fold-sweep/v1', source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    code_sha256=hashes, hand_scale=animation['hand_scale'], frame=24, records=[],
    scope=__doc__, candidate_exported=False, full_performance_accepted=False,
    hypothesis='Removing individual overcurl may clear self folds but lose physical key contact; root/other joints must then be planned together.',
    exclusions=['actual Blender parity at each changed pose pending', 'neighbor speed bounds', 'interhand/body/costume',
                'full music', 'forces/action', 'continuous time', 'naturalness acceptance'])
prior = PianistFormPrior()
begun = time.monotonic()
for side in ('R', 'L'):
    rig, skin = HandRig(side), Skin(side, scale=animation['hand_scale'])
    operator, faces = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
    triangles = quad_triangles(faces)
    frame = animation['frames'][24]
    hand = frame['hands'][side]
    source_q = pose_vector(rig, hand)[6:]
    distal = {f: np.flatnonzero(np.asarray(operator @ skin.weights[:, skin.bone_index[f'finger{f}-3']]) > .2)
              for f in range(1, 6)}
    scene = scene_from_animation(animation, frame, music_desk=True)
    dofs = [(b, d) for ch in rig.chains for b, d, *_ in ch.dofs]
    if side == 'R':
        index = dofs.index(('finger4-2', 'flex'))
        samples = [{index: value} for value in (rig.q_rest[index], .6, .8, 1., 1.2, source_q[index])]
    else:
        abd, flex = (dofs.index(('finger1-1', d)) for d in ('abd', 'flex'))
        samples = [{abd: a, flex: f} for a in (0., source_q[abd] / 2., source_q[abd])
                   for f in (0., source_q[flex] / 2., source_q[flex])]
    for n, changes in enumerate(samples):
        q = source_q.copy()
        for i, value in changes.items():
            q[i] = value
        # Frozen legacy input includes a 2.015e-6 rad rounding excess in one
        # unchanged left pinky DOF. Retain and report it; do not silently alter
        # other joints to make a diagnostic seem anatomically accepted.
        assert all(rig.lo[i] <= q[i] <= rig.hi[i] for i in changes)
        bounds_excess = float(max(0., (rig.lo - q).max(), (q - rig.hi).max()))
        vector = np.r_[hand['root']['pos'], np.zeros(3), q]
        rotation, root, points = rig.points(vector, np.asarray(hand['root']['rot']))
        vertices = operator @ skin.pose(rotation, root, rig.angles(vector), hand['elbow'])
        mesh = quad_self_intersections(vertices, faces)
        instrument = hand_audit(vertices, triangles, scene)
        support = [dict(finger=f, pitch=p, **key_support(vertices, distal[f], p, scene))
                   for f, p in enumerate(hand['assigned_keys'], 1) if p is not None]
        form = prior.evaluate(np.asarray([root] + [p for chain in points for p in chain]), side, hand['activation'])
        result = dict(side=side, sample=n, changes={b+'.'+d:float(q[i]) for i,(b,d) in enumerate(dofs) if i in changes},
            q=q.tolist(), self_clear=mesh['surface_intersection_clear'],
            self_crossing_pairs=mesh['unexpected_surface_intersections'], degenerate_triangles=mesh['degenerate_triangles'],
            source_anatomy_excess_rad=float(max(0., (rig.lo-source_q).max(), (source_q-rig.hi).max())),
            anatomy_excess_rad=bounds_excess, changed_joints_in_bounds=True,
            mesh=mesh, instrument=instrument, support=support, form=form,
            wrist_elbow_and_other_q_unchanged=True, candidate_exported=False,
            seconds=time.monotonic()-begun)
        report['records'].append(result)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k:result[k] for k in ('side','sample','changes','self_crossing_pairs','seconds')} |
                         dict(instrument_depth_mm=instrument['maximum_penetration_mm'],
                              supported=sum(s['supported'] for s in support))), flush=True)
report['status'] = 'complete'
report['source_unchanged_during_run'] = hashes == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
assert report['source_unchanged_during_run']
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
print('Complete diagnostic sweep; no pose or motion candidate')

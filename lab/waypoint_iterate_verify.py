"""Reconstruct rejected numerical checkpoints in Blender for independent QA.

No rejected hand is exported as a motion or an accepted pose. The full mesh
snapshots and diagnostic stills preserve its failure for independent analysis.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.rig.hand import HandRig
from blockout.skin import Skin
from lab.skin_deformation_snapshot import extract


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elbow_at(shoulder, wrist, reference):
    # Independent geometry reconstruction: no SciPy/optimizer inside Blender.
    from blockout.performer import UPPER_ARM, FOREARM
    shoulder, wrist, reference = map(lambda v: np.asarray(v, float), (shoulder, wrist, reference))
    delta = wrist - shoulder
    distance = np.linalg.norm(delta)
    assert abs(UPPER_ARM - FOREARM) < distance < UPPER_ARM + FOREARM
    u = delta / distance
    along = (UPPER_ARM ** 2 - FOREARM ** 2 + distance ** 2) / (2 * distance)
    center = shoulder + along * u
    radial = reference - center
    radial -= u * (radial @ u)
    assert np.linalg.norm(radial) > 1e-10
    return center + radial / np.linalg.norm(radial) * np.sqrt(UPPER_ARM ** 2 - along ** 2)


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {name: ROOT / f'artifacts/realism/waypoint_union_replay_{name}.json'
             for name in ('source', 'lift')}
    trials = {name: json.loads(p.read_text()) for name, p in files.items()}
    source = Path(trials['source']['source'])
    animation = json.loads(source.read_text())
    assert all(sha(source) == t['source_sha256'] and not t['accepted']
               and t['source_unchanged_during_run'] and t['frozen_input_unchanged_during_run']
               and not t['animation_candidate_exported'] for t in trials.values())
    scale = trials['source']['hand_scale']
    os.environ['HAND_SCALE'] = str(scale)
    frame_id = trials['source']['frame']
    sys.argv = ['blender_piano.py', '--', str(source), str(output / 'scene'),
        '--frames', str(frame_id), '--meshqa', str(output / 'startup.json'), '--view', 'three4',
        '--eye', '0.30,-0.25,0.19', '--look', '0.16,0.04,0.025', '--lens', '50',
        '--res', '1024x768', '--hide', 'body']
    renderer = runpy.run_path(str(ROOT / 'blockout/blender_piano.py'))
    mesh, arm, rest = renderer['MPFB_MESH'], renderer['MPFB_ARM'], renderer['MPFB_REST']
    mesh.modifiers['smooth'].show_viewport = True
    mesh.modifiers['smooth'].levels = 2
    inputs = [source, *files.values(), Path(__file__), ROOT / 'blockout/blender_piano.py',
              ROOT / 'blockout/skin.py', ROOT / 'blockout/rig/hand.py']
    fingerprints = {str(p): sha(p) for p in inputs}
    report = dict(schema='rejected-waypoint-blender-snapshots/v1', hand_scale=scale,
        frozen_input_sha256=fingerprints, candidate_exported=False, full_performance_accepted=False,
        scope='Independent actual-mesh reconstruction of two rejected static numerical iterates; no performance candidate',
        records=[])
    for name, trial in trials.items():
        frame = copy.deepcopy(animation['frames'][frame_id])
        frame['keys'] = copy.deepcopy(trial['key_commands'])
        side = trial['side']
        hand = frame['hands'][side]
        rig = HandRig(side)
        z = np.asarray(trial['numerical_iterate']['solver_coordinates'], float)
        assert z.shape == (len(rig.lo) + 6,) and np.isfinite(z).all()
        x = z.copy()
        x[:3] = np.asarray(hand['root']['pos']) + z[:3] / 1000.
        rotation, root, _ = rig.points(x, np.asarray(hand['root']['rot']))
        hand['angles'] = rig.angles(x)
        hand['root']['pos'] = root.tolist()
        hand['root']['rot'] = rotation.tolist()
        hand['wrist'] = root.tolist()
        hand['elbow'] = elbow_at(frame['body']['shoulders'][side], root, hand['elbow']).tolist()
        for s, value in frame['hands'].items():
            renderer['pose_rig'](arm, rest, s, value)
        for pitch, key in renderer['keys'].items():
            key.rotation_euler = (float(frame['keys'].get(pitch, 0.)) * .010 / key['length'], 0., 0.)
        bpy.context.view_layer.update()
        vertices, faces, triangles = extract(mesh)
        path = output / f'union_{name}.npz'
        np.savez_compressed(path, vertices=vertices, faces=faces, triangles=triangles,
            **{s: Skin(s, scale=scale).from_frame(h) for s,h in frame['hands'].items()})
        item = dict(context=f'union_{name}', frame=frame_id, source=str(source),
            source_sha256=sha(source), mode='linear', subdivision_levels=2,
            mesh=str(path), mesh_sha256=sha(path), changed_side=side,
            pose_changes=dict(rejected_checkpoint_sha256=sha(files[name]), coordinates=z.tolist()),
            reconstructed_hand=hand, candidate_exported=False, no_pose_or_animation_candidate=True)
        item['key_commands'] = frame['keys']
        report['records'].append(item)
        # Refresh visible arm primitives as well as the true skinned hand mesh.
        el, wr = Vector(hand['elbow']), Vector(hand['wrist'])
        renderer['rig'][side]['forearm'].set(el, el + (wr - el).normalized() * .06)
        renderer['rig'][side]['upper'].set(frame['body']['shoulders'][side], hand['elbow'])
        bpy.context.scene.render.filepath = str(output / f'union_{name}.png')
        bpy.ops.render.render(write_still=True)
    report.update(status='complete', source_unchanged_during_run=all(sha(Path(p)) == v for p,v in fingerprints.items()))
    assert report['source_unchanged_during_run']
    (output / 'snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Two rejected static iterates independently reconstructed; no motion candidate')


if __name__ == '__main__':
    main()

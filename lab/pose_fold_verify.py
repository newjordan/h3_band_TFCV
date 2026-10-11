"""Reconstruct selected diagnostic joint-sweep poses in Blender; no animation."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.rig.hand import HandRig
from blockout.skin import Skin
from lab.skin_deformation_snapshot import extract


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sweep_file = ROOT / 'artifacts/realism/pose_fold_sweep_20261011.json'
    sweep = json.loads(sweep_file.read_text())
    assert sweep['status'] == 'complete' and sweep['source_unchanged_during_run']
    source = Path(sweep['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sweep['source_sha256']
    animation = json.loads(source.read_text())
    sys.argv = ['blender_piano.py', '--', str(source), str(output / 'scene'),
        '--frames', '24', '--meshqa', str(output / 'startup.json'), '--view', 'three4',
        '--eye', '0.30,-0.25,0.19', '--look', '0.16,0.04,0.025', '--lens', '50',
        '--res', '1024x768', '--hide', 'body']
    renderer = runpy.run_path(str(ROOT / 'blockout/blender_piano.py'))
    mesh, arm, rest = renderer['MPFB_MESH'], renderer['MPFB_ARM'], renderer['MPFB_REST']
    mesh.modifiers['smooth'].show_viewport = True
    mesh.modifiers['smooth'].levels = 2
    selected = [next(r for r in sweep['records'] if r['side'] == 'R' and r['sample'] == n) for n in (0, 5)]
    left = [r for r in sweep['records'] if r['side'] == 'L']
    selected += [min(left, key=lambda r: (r['self_crossing_pairs'], r['instrument']['maximum_penetration_mm'])), left[-1]]
    report = dict(schema='selected-joint-sweep-blender-snapshot/v1', hand_scale=sweep['hand_scale'],
        source_sha256=sweep['source_sha256'], sweep_sha256=hashlib.sha256(sweep_file.read_bytes()).hexdigest(),
        selection='R comfort/original PIP; L fewest self pairs then lowest instrument depth, plus unchanged CMC control',
        candidate_exported=False, full_performance_accepted=False, records=[])
    for trial in selected:
        frame = copy.deepcopy(animation['frames'][24])
        rig = HandRig(trial['side'])
        hand = frame['hands'][trial['side']]
        hand['angles'] = rig.angles(np.r_[hand['root']['pos'], np.zeros(3), trial['q']])
        for side, value in frame['hands'].items():
            renderer['pose_rig'](arm, rest, side, value)
        bpy.context.view_layer.update()
        vertices, faces, triangles = extract(mesh)
        name = f"{trial['side']}_sample{trial['sample']}"
        path = output / (name + '.npz')
        np.savez_compressed(path, vertices=vertices, faces=faces, triangles=triangles,
            **{s: Skin(s, scale=sweep['hand_scale']).from_frame(h) for s,h in frame['hands'].items()})
        item = dict(context=name, frame=24, source=str(source), source_sha256=sweep['source_sha256'],
            mode='linear', subdivision_levels=2, mesh=str(path), mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pose_changes=trial['changes'], changed_side=trial['side'], source_anatomy_excess_rad=trial['source_anatomy_excess_rad'],
            no_pose_or_animation_candidate=True, wrist_elbow_activation_commands_unchanged=True)
        report['records'].append(item)
        bpy.context.scene.render.filepath = str(output / (name + '.png'))
        bpy.ops.render.render(write_still=True)
    report.update(status='complete', source_unchanged_during_run=hashlib.sha256(source.read_bytes()).hexdigest() == sweep['source_sha256'])
    (output / 'snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Four selected poses reconstructed and rendered; no animation candidate')


if __name__ == '__main__':
    main()

"""Extract actual production Blender meshes at matched composure trial times."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.skin import Skin


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    trial = sys.argv[sys.argv.index('--') + 1]
    source = ROOT / f'artifacts/realism/composure_{trial}.anim.json'
    animation = json.loads(source.read_text())
    directory = ROOT / f'artifacts/realism/composure_actual_{trial}'
    directory.mkdir(exist_ok=True)
    fingerprints = {str(path): sha(path) for path in (source, ROOT / 'blockout/blender_piano.py',
        ROOT / 'blockout/skin.py', ROOT / 'lab/composure_snapshot.py')}
    sys.argv = ['blender', '--', str(source), str(directory), '--frames', '0:0',
                '--res', '768x432', '--quality', 'draft', '--hide', 'body']
    renderer = runpy.run_path(str(ROOT / 'blockout/blender_piano.py'), run_name='composure_renderer')
    arm, rest = renderer['MPFB_ARM'], renderer['MPFB_REST']
    mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH'
                and any(m.type == 'ARMATURE' for m in o.modifiers))
    assert next(m for m in mesh.modifiers if m.type == 'ARMATURE').use_deform_preserve_volume is False
    smooth = next(m for m in mesh.modifiers if m.type == 'SUBSURF')
    # Evaluated viewport extraction must use the production render level.
    smooth.levels = smooth.render_levels
    assert smooth.levels == 2 and smooth.use_limit_surface
    report = dict(status='running', hand_scale=animation['hand_scale'],
        candidate_exported=False, records=[], input_sha256=fingerprints,
        scope='Actual unchanged production skin, same two matched times in each trial; no trajectory acceptance')
    for index in (48, 108):
        frame = animation['frames'][index]
        for side in ('L', 'R'):
            renderer['pose_rig'](arm, rest, side, frame['hands'][side])
        for pitch, key in renderer['keys'].items():
            key.rotation_euler = (float(frame['keys'].get(pitch, 0.)) * .010 / key['length'], 0., 0.)
        bpy.context.view_layer.update()
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        value = evaluated.to_mesh()
        value.calc_loop_triangles()
        vertices = np.array([evaluated.matrix_world @ vertex.co for vertex in value.vertices])
        faces = np.array([polygon.vertices[:] for polygon in value.polygons], np.int32)
        triangles = np.array([triangle.vertices[:] for triangle in value.loop_triangles], np.int32)
        evaluated.to_mesh_clear()
        target = directory / f'frame{index}.npz'
        np.savez_compressed(target, vertices=vertices, faces=faces, triangles=triangles,
            **{side: Skin(side, scale=animation['hand_scale']).from_frame(frame['hands'][side]) for side in ('L', 'R')})
        report['records'].append(dict(context=f'{trial}_frame{index}', frame=index,
            source=str(source), source_sha256=sha(source), mode='linear', subdivision_levels=2,
            mesh=str(target), mesh_sha256=sha(target), trial=trial,
            vertices=len(vertices), quads=len(faces), triangles=len(triangles)))
    report['source_unchanged_during_run'] = all(sha(Path(path)) == value for path, value in fingerprints.items())
    assert report['source_unchanged_during_run']
    report['status'] = 'complete'
    (directory / 'snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

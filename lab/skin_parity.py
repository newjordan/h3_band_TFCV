"""Compare NumPy skin coordinates with Blender at specified saved poses.

HAND_SCALE=1 blender -b --factory-startup -P lab/skin_parity.py -- ANIM.json OUT.json --frames 24,50,68
Uses the production renderer's actual armature, then reads evaluated vertices.
"""
import json
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    source, output = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    options = dict(zip(argv[2::2], argv[3::2]))
    indices = [int(i) for i in options.get('--frames', '24,50,68').split(',')]
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from blockout.skin import Skin
    from blockout.rig.skeleton import hand_scale
    from blockout.piano_geometry import PianoScene
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ['blender_piano.py', '--', str(source), str(output.parent / 'parity_scene'),
                '--frames', str(indices[0]), '--meshqa', str(output.with_suffix('.mesh.json'))]
    renderer = runpy.run_path(str(root / 'blockout' / 'blender_piano.py'))
    mesh, arm, rest = renderer['MPFB_MESH'], renderer['MPFB_ARM'], renderer['MPFB_REST']
    skins = {side: Skin(side, scale=hand_scale()) for side in ('L', 'R')}
    levels = int(options.get('--levels', 0))
    mesh.modifiers['smooth'].show_viewport = levels > 0
    mesh.modifiers['smooth'].levels = max(1, levels)
    anim = json.loads(source.read_text())
    records = []
    for index in indices:
        frame = anim['frames'][index]
        for pitch, key in renderer['keys'].items():
            key.rotation_euler = (float(frame['keys'].get(pitch, 0.)) * .010 / key['length'], 0., 0.)
        for side, hand in frame['hands'].items():
            renderer['pose_rig'](arm, rest, side, hand)
        bpy.context.view_layer.update()
        evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        deformed = evaluated.to_mesh()
        vertices = np.array([v.co for v in deformed.vertices])
        deformed.calc_loop_triangles()
        triangles = np.array([tri.vertices for tri in deformed.loop_triangles], dtype=np.int32)
        polygons = np.array([poly.vertices[:] for poly in deformed.polygons], dtype=np.int32)
        evaluated.to_mesh_clear()
        scene = PianoScene(anim['keyboard'], frame['keys'], music_desk=True)
        box_error = 0.
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for bi, spec in enumerate(scene.specs):
            ob = bpy.data.objects[spec.name]
            actual = np.array([ob.matrix_world @ vertex.co for vertex in ob.data.vertices])
            local = scene.centers[bi] + signs * scene.half[bi]
            expected = local @ scene.rotations[bi].T + scene.origins[bi]
            error = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
            box_error = max(box_error, float(max(error.min(0).max(), error.min(1).max())) * 1000.)
        if levels:
            # Blender ships NumPy but not SciPy. Export both coordinate sets for
            # the separate scientific-environment checker; never mark this pass
            # successful merely because Blender wrote the data.
            data_file = output.with_name(f'{output.stem}_frame{index}.npz')
            np.savez_compressed(data_file, blender=vertices, blender_triangles=triangles,
                                blender_faces=polygons,
                                **{side: skin.from_frame(frame['hands'][side])
                                   for side, skin in skins.items()})
            records.append({'frame': index, 'side': 'both', 'vertices': len(vertices),
                            'data_file': str(data_file), 'scene_box_maximum_error_mm': box_error})
        else:
            for side, skin in skins.items():
                expected = vertices[skin.vertex_ids]
                actual = skin.from_frame(frame['hands'][side])
                error = np.linalg.norm(actual - expected, axis=1) * 1000.
                records.append({'frame': index, 'side': side, 'vertices': len(error),
                                'maximum_error_mm': float(error.max()),
                                'rms_error_mm': float(np.sqrt(np.mean(error ** 2))),
                                'worst_vertex': int(skin.vertex_ids[error.argmax()]),
                                'scene_box_maximum_error_mm': box_error})
    report = {'source': str(source), 'hand_scale': hand_scale(),
              'surface': 'normalized_linear_blend_then_catmull_clark', 'subdivision_levels': levels,
              'limit_surface': bool(mesh.modifiers['smooth'].use_limit_surface),
              'boundary_smooth': mesh.modifiers['smooth'].boundary_smooth,
              'tolerance_mm': .01, 'records': records,
              'passed': None if levels else all(r['maximum_error_mm'] < .01 and
                                                r['scene_box_maximum_error_mm'] < .01 for r in records),
              'validation': 'pending_external_checker' if levels else 'complete'}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['passed'] is False:
        raise RuntimeError('CPU skin does not agree with the production renderer')


if __name__ == '__main__':
    main()

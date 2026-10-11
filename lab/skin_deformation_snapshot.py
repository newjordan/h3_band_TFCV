"""Blender-only frozen-pose LBS/preserve-volume ablation, never a candidate.

blender -b --factory-startup -P lab/skin_deformation_snapshot.py -- OUT_DIR
Uses production bones, weights and topology. Only the Armature modifier's
preserve-volume setting changes inside each matched pair; no solver runs.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import bpy
from mathutils import Matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.skin import Skin
from blockout.rig.pose_io import animation_hand_scale


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_state(arm):
    return np.array([np.array(b.matrix) for b in arm.pose.bones])


def extract(mesh):
    evaluated = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    value = evaluated.to_mesh()
    value.calc_loop_triangles()
    vertices = np.array([evaluated.matrix_world @ v.co for v in value.vertices])
    faces = np.array([p.vertices[:] for p in value.polygons], np.int32)
    triangles = np.array([t.vertices[:] for t in value.loop_triangles], np.int32)
    evaluated.to_mesh_clear()
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError('The independent two-diagonal gate requires complete quad topology')
    return vertices, faces, triangles


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {name: ROOT / f'artifacts/realism/{file}' for name, file in (
        ('fresh', 'measured_notched_overlap.anim.json'),
        ('contact', 'human_pilot_source.render.anim.json'),
        ('form', 'human_pilot_form.render.anim.json'))}
    animations = {n: json.loads(p.read_text()) for n, p in sources.items()}
    scale = animation_hand_scale(animations['fresh'])
    assert all(animation_hand_scale(a) == scale for a in animations.values())
    assert all(a['geometry'] == animations['fresh']['geometry'] for a in animations.values())
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in list(sources.values()) +
        [ROOT / p for p in ('lab/skin_deformation_snapshot.py', 'blockout/blender_piano.py',
         'blockout/skin.py', 'blockout/rig/skeleton.py', 'blockout/rig/furelise_cal.json',
         'blockout/hand_model/mpfb_hands.blend', 'blockout/hand_model/skin_bind.npz')]}
    sys.argv = ['blender_piano.py', '--', str(sources['fresh']), str(output / 'scene'),
                '--frames', '24', '--meshqa', str(output / 'startup_mesh.json'),
                '--view', 'three4', '--eye', '0.35,-0.52,0.38', '--look', '0,0.015,0.02',
                '--lens', '42', '--res', '1280x720', '--hide', 'body']
    renderer = runpy.run_path(str(ROOT / 'blockout/blender_piano.py'))
    mesh, arm, rest = renderer['MPFB_MESH'], renderer['MPFB_ARM'], renderer['MPFB_REST']
    modifiers = [m for m in mesh.modifiers if m.type == 'ARMATURE']
    assert len(modifiers) == 1
    deform = modifiers[0]
    assert deform.use_vertex_groups and not deform.use_bone_envelopes
    original_volume = bool(deform.use_deform_preserve_volume)
    assert not original_volume, 'Frozen reference must use the established LBS renderer'
    smooth = mesh.modifiers['smooth']
    report = dict(schema='skin-deformation-snapshot/v1', blender_version=bpy.app.version_string,
        production_default_preserve_volume=original_volume, hand_scale=scale,
        source_sha256={n: digest(p) for n, p in sources.items()}, frozen_code_sha256=fingerprints,
        candidate_exported=False, full_performance_accepted=False,
        hypothesis='Quaternion deformation may remove some folds caused by linear blending at identical bone poses.',
        ablation='Three frozen frame-24 contexts plus undeformed bind control; preserve volume off/on; '
                 'base cage and actual renderer level-2 limit surface. No pose/bone/weight/scale edits within pairs.',
        records=[])
    for name in ('fresh', 'contact', 'form', 'bind_control'):
        if name == 'bind_control':
            for bone in arm.pose.bones:
                bone.matrix_basis = Matrix.Identity(4)
            frame = None
            expected = {}
        else:
            frame = animations[name]['frames'][24]
            for side, hand in frame['hands'].items():
                renderer['pose_rig'](arm, rest, side, hand)
            for pitch, key in renderer['keys'].items():
                key.rotation_euler = (float(frame['keys'].get(pitch, 0.)) * .010 / key['length'], 0., 0.)
            expected = {side: Skin(side, scale=scale).from_frame(frame['hands'][side]) for side in ('L', 'R')}
        bpy.context.view_layer.update()
        frozen_bones = pose_state(arm)
        for level in (0, 2):
            smooth.show_viewport = level > 0
            smooth.levels = max(1, level)
            pair_faces = None
            for volume in (False, True):
                deform.use_deform_preserve_volume = volume
                bpy.context.view_layer.update()
                bone_error = float(np.abs(pose_state(arm) - frozen_bones).max())
                assert bone_error == 0., 'Changing deformation must not change any bone transform'
                vertices, faces, triangles = extract(mesh)
                if pair_faces is None:
                    pair_faces = faces
                assert np.array_equal(pair_faces, faces)
                mode = 'quaternion' if volume else 'linear'
                target = output / f'{name}_{mode}_level{level}.npz'
                np.savez_compressed(target, vertices=vertices, faces=faces, triangles=triangles,
                    bone_matrices=frozen_bones, **expected)
                report['records'].append(dict(context=name, frame=None if frame is None else 24,
                    source=None if frame is None else str(sources[name]), source_sha256=None if frame is None else digest(sources[name]),
                    mode=mode, subdivision_levels=level, limit_surface=bool(smooth.use_limit_surface),
                    boundary_smooth=smooth.boundary_smooth, mesh=str(target), mesh_sha256=digest(target),
                    vertices=len(vertices), quads=len(faces), triangles=len(triangles), bone_matrix_error=bone_error,
                    bind_control_scale=1. if frame is None else None,
                    fingers_wrist_elbows_activation_key_commands_changed=False))
                print(json.dumps(report['records'][-1]), flush=True)
                (output / 'snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
    report['source_unchanged_during_run'] = all(digest(ROOT / p) == value for p, value in fingerprints.items())
    assert report['source_unchanged_during_run']
    report['status'] = 'complete'
    (output / 'snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
    print('complete: 16 actual mesh snapshots; no animation candidate')


if __name__ == '__main__':
    main()

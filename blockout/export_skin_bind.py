"""Export the renderer's CC0 skinned hand into a NumPy asset (run with Blender).

blender -b --factory-startup -P blockout/export_skin_bind.py -- OUT.npz
This reads the existing .blend and never modifies it. Pose-space coordinates,
normalized vertex weights and inverse bind matrices are exported together.
"""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    output = Path(argv[0])
    source = Path(__file__).resolve().parent / 'hand_model' / 'mpfb_hands.blend'
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(source)) as (_src, dst):
        dst.objects = ['hands', 'Human.rig']
    mesh, arm = dst.objects
    for obj in (mesh, arm):
        bpy.context.scene.collection.objects.link(obj)
    modifier = next(m for m in mesh.modifiers if m.type == 'ARMATURE')
    if modifier.use_deform_preserve_volume:
        raise ValueError('Dual-quaternion skinning requires a different evaluator')
    if not modifier.use_vertex_groups or modifier.use_bone_envelopes:
        raise ValueError('Expected vertex-weight-only linear blend skinning')
    bones = list(arm.data.bones)
    bone_ids = {b.name: i for i, b in enumerate(bones)}
    names = [b.name for b in bones]
    bind = np.array([b.matrix_local for b in bones], dtype=np.float64)
    mesh_to_arm = np.array(arm.matrix_world.inverted() @ mesh.matrix_world)
    local = np.array([v.co for v in mesh.data.vertices], dtype=np.float64)
    vertices = local @ mesh_to_arm[:3, :3].T + mesh_to_arm[:3, 3]
    weights = np.zeros((len(vertices), len(bones)))
    groups = {g.index: g.name for g in mesh.vertex_groups}
    for vertex in mesh.data.vertices:
        for group in vertex.groups:
            name = groups[group.group]
            if name in bone_ids and bones[bone_ids[name]].use_deform:
                weights[vertex.index, bone_ids[name]] += group.weight
    totals = weights.sum(1)
    if np.any(totals <= 0):
        raise ValueError('Unweighted surface vertices cannot share this deformation model')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from blockout.skin import clean_hand_weights
    weights, repairs = clean_hand_weights(weights, names)
    sides = np.array([names[int(i)].rsplit('.', 1)[1] for i in weights.argmax(1)])
    if set(sides) != {'L', 'R'}:
        raise ValueError('Expected exactly two independent hand surfaces')
    # Keep original polygon boundaries: subdivision must precede triangulation.
    faces = [list(p.vertices) for p in mesh.data.polygons]
    lengths = np.array([len(f) for f in faces], dtype=np.int32)
    mesh.data.calc_loop_triangles()
    triangles = np.array([t.vertices for t in mesh.data.loop_triangles], dtype=np.int32)
    metadata = {
        'schema': 1, 'source': 'MPFB2 CC0 mesh used by blender_piano.py',
        'blend_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'blender_version': bpy.app.version_string, 'skinning': 'normalized_linear_blend',
        'vertices': len(vertices), 'bones': len(bones), 'faces': len(faces),
        'mesh_world': np.array(mesh.matrix_world).tolist(),
        'arm_world': np.array(arm.matrix_world).tolist(),
        'weights_sum_before_normalization': [float(totals.min()), float(totals.max())],
        'cross_hand_weight_repairs': repairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices, weights=weights,
                        inverse_bind=np.linalg.inv(bind), bind=bind,
                        bone_names=np.array(names), sides=sides,
                        face_sizes=lengths, face_vertices=np.array(sum(faces, []), dtype=np.int32),
                        triangles=triangles, metadata=np.array(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

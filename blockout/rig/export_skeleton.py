"""Export the MPFB hand skeleton for the rig (Blender 4.0+, no add-on needed):

    blender -b --factory-startup blockout/hand_model/mpfb_hands.blend -P blockout/rig/export_skeleton.py -- OUT.json

Per hand (L, R) and bone: head, tail, length, parent and the full rest frame (matrix_local, armature = world space),
plus collision radii measured off the skinned mesh: the mean distance of the bone's dominant vertices from the bone
axis at its head, middle and tail (a tapered capsule per bone).
"""
import json, sys

import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index("--") + 1]
arm = bpy.data.objects["Human.rig"]
mesh = bpy.data.objects["hands"]
gidx = {g.name: g.index for g in mesh.vertex_groups}
dom = {}
for v in mesh.data.vertices:
    g = max(v.groups, key=lambda e: e.weight, default=None)
    if g and g.weight > 0.5:
        dom.setdefault(g.group, []).append(mesh.matrix_world @ v.co)
data = {"source": "MPFB2 2.0.17 default rig (CC0 mesh), blockout/hand_model/mpfb_hands.blend", "hands": {}}
for side in "LR":
    bones = {}
    for b in arm.data.bones:
        if not b.name.endswith("." + side):
            continue
        name = b.name[:-2]
        h, t = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
        ax = (t - h); L = ax.length; ax.normalize()
        radii = []
        pts = dom.get(gidx.get(b.name), [])
        for u0 in (0.1, 0.5, 0.9):
            ds = [((p - h) - ax * (p - h).dot(ax)).length for p in pts if abs((p - h).dot(ax) / L - u0) < 0.2]
            radii.append(round(sum(ds) / len(ds), 5) if ds else None)
        m = (arm.matrix_world @ b.matrix_local).to_3x3()
        # cross-section at head/mid/tail: half-width along the bone's x, extents toward its +z and -z
        bx, bz = m.col[0].normalized(), m.col[2].normalized()
        sect = []
        for u0 in (0.1, 0.5, 0.9):
            sl = [p - h for p in pts if abs((p - h).dot(ax) / L - u0) < 0.15]
            if not sl:
                sect.append(None); continue
            xs = sorted(abs(v.dot(bx)) for v in sl)
            zp = sorted(v.dot(bz) for v in sl)
            q = lambda arr, f: arr[min(len(arr) - 1, int(f * len(arr)))]
            sect.append([round(q(xs, 0.9), 5), round(q(zp, 0.95), 5), round(-q(zp, 0.05), 5)])
        bones[name] = {"head": list(h), "tail": list(t), "length": L, "parent": b.parent.name[:-2] if b.parent else None,
                       "rest": [list(r) for r in m], "radii": radii, "section": sect}
    data["hands"][side] = bones
json.dump(data, open(out, "w"), indent=1)
print("wrote", out)

"""Export the MPFB hands' skin for the working piano's contact model (Blender 4.0+):

    blender -b --factory-startup blockout/hand_model/mpfb_hands.blend -P blockout/rig/export_mesh.py -- OUT.json

Per hand (L, R): rest vertices (world space, the armature's rest pose, as the skeleton export), triangles, and each
vertex's deform weights by bone (normalised over the hand's bones; vertices mostly on the forearm bones are dropped).
The mesh has only an Armature modifier (linear blend skinning), so posing these with the rig's FK reproduces the
rendered skin."""
import json, sys
import bpy

out = sys.argv[sys.argv.index("--") + 1]
mesh = bpy.data.objects["hands"]
names = {g.index: g.name for g in mesh.vertex_groups}
mesh.data.calc_loop_triangles()
data = {"source": "MPFB2 hands mesh, blockout/hand_model/mpfb_hands.blend", "hands": {}}
for side in "LR":
    keep, verts, wts = {}, [], []
    for v in mesh.data.vertices:
        w = {names[g.group][:-2]: g.weight for g in v.groups if names[g.group].endswith("." + side) and g.weight > 1e-4}
        tot = sum(w.values())
        if tot < 0.5:
            continue
        arm = sum(x for b, x in w.items() if b.startswith("lowerarm"))
        if arm > 0.5 * tot:
            continue
        w = {b: x for b, x in w.items() if not b.startswith("lowerarm")}
        s = sum(w.values())
        keep[v.index] = len(verts)
        co = mesh.matrix_world @ v.co
        verts.append([round(c, 6) for c in co])
        wts.append({b: round(x / s, 5) for b, x in w.items()})
    tris = [[keep[i] for i in t.vertices] for t in mesh.data.loop_triangles if all(i in keep for i in t.vertices)]
    data["hands"][side] = {"verts": verts, "weights": wts, "tris": tris}
    print(side, len(verts), "verts", len(tris), "tris")
json.dump(data, open(out, "w"))
print("wrote", out)

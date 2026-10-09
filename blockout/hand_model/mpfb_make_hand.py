"""Generate the CC0 MPFB2 hand (Blender >= 4.2 with the MPFB2 extension; run on an x86-64 box).

    blender -b --factory-startup -P blockout/hand_model/mpfb_make_hand.py -- OUT_DIR

Makes an adult male MakeHuman body, adds MPFB's default rig with its skin weights, keeps only the two hands
(plus a short forearm stub) and the hand/forearm bones, and writes:
  OUT_DIR/mpfb_hands.glb   skinned mesh + armature (opens in Blender 4.0)
  OUT_DIR/mpfb_hands.json  rest joint positions per hand (metres, Blender world) and measured finger radii
MPFB2 code is GPL-3.0; the meshes it generates are CC0.
"""
import json, os, sys

import addon_utils
import bpy
import bmesh
from mathutils import Vector

out_dir = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out_dir, exist_ok=True)
addon_utils.enable("bl_ext.user_default.mpfb", default_set=True)
from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
macro = TargetService.get_default_macro_info_dict()
macro.update(gender=1.0, age=0.5, muscle=0.7, weight=0.6, height=0.75, proportions=0.6)
body = HumanService.create_human(mask_helpers=False, detailed_helpers=True, extra_vertex_groups=True,
                                 feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
HumanService.add_builtin_rig(body, "default", import_weights=True)
arm = body.parent if body.parent and body.parent.type == "ARMATURE" else next(
    o for o in bpy.data.objects if o.type == "ARMATURE")

# bake the shape keys (body shape) into the mesh so the export carries the knight's proportions
bpy.context.view_layer.objects.active = body
for o in bpy.context.selected_objects:
    o.select_set(False)
body.select_set(True)
if body.data.shape_keys:
    body.shape_key_add(name="mix", from_mix=True)
    keep = body.data.shape_keys.key_blocks["mix"]
    for kb in list(body.data.shape_keys.key_blocks):
        if kb.name != "mix":
            body.shape_key_remove(kb)
    body.shape_key_remove(keep)        # removing the last key applies it to the base mesh

HAND = ["wrist", "metacarpal1", "metacarpal2", "metacarpal3", "metacarpal4"] + [
    f"finger{f}-{s}" for f in range(1, 6) for s in range(1, 4)]
KEEP_BONES = {f"{b}.{s}" for b in HAND + ["lowerarm01", "lowerarm02"] for s in "LR"}

# keep body-surface verts that belong to a hand (or the last few cm of forearm)
gi = {g.name: g.index for g in body.vertex_groups}
hand_groups = {gi[f"{b}.{s}"] for b in HAND for s in "LR" if f"{b}.{s}" in gi}
fore_groups = {gi[f"lowerarm02.{s}"] for s in "LR" if f"lowerarm02.{s}" in gi}
body_g = gi.get("body")
wrist_head = {s: arm.matrix_world @ arm.data.bones[f"wrist.{s}"].head_local for s in "LR"}
el_head = {s: arm.matrix_world @ arm.data.bones[f"lowerarm01.{s}"].head_local for s in "LR"}
STUB = 0.08                                   # metres of forearm kept behind the wrist joint
mw = body.matrix_world
keep_v = set()
for v in body.data.vertices:
    w = {g.group: g.weight for g in v.groups}
    if body_g is not None and w.get(body_g, 0) <= 0:
        continue
    hw = sum(w.get(g, 0) for g in hand_groups)
    fw = sum(w.get(g, 0) for g in fore_groups)
    if hw > 0.05:
        keep_v.add(v.index); continue
    if fw > 0.05:
        p = mw @ v.co
        for s in "LR":
            ax = (wrist_head[s] - el_head[s]).normalized()
            if -STUB < (p - wrist_head[s]).dot(ax) <= 0.01 and (p - wrist_head[s]).length < 0.14:
                keep_v.add(v.index)
bm = bmesh.new(); bm.from_mesh(body.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep_v], context="VERTS")
bm.to_mesh(body.data); bm.free()
for g in list(body.vertex_groups):
    if g.name not in KEEP_BONES:
        body.vertex_groups.remove(g)
for m in list(body.modifiers):
    if m.type != "ARMATURE":
        body.modifiers.remove(m)

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.data.edit_bones
for b in list(eb):
    if b.name not in KEEP_BONES:
        eb.remove(b)
bpy.ops.object.mode_set(mode="OBJECT")

# split into one mesh per hand (by x side)
body.name = "hands"
info = {"source": "MPFB2 2.0.17 default rig, CC0 mesh", "macro": {k: v for k, v in macro.items() if k != "race"},
        "hands": {}}
for s in "LR":
    bones = {}
    for b in arm.data.bones:
        if b.name.endswith("." + s):
            bones[b.name[:-2]] = {"head": list(arm.matrix_world @ b.head_local),
                                  "tail": list(arm.matrix_world @ b.tail_local),
                                  "parent": b.parent.name[:-2] if b.parent else None}
    # finger radii: mean distance of a bone's dominant verts from its axis
    radii = {}
    for name, bd in bones.items():
        g = body.vertex_groups.get(f"{name}.{s}")
        if g is None:
            continue
        h, t = Vector(bd["head"]), Vector(bd["tail"])
        ax = (t - h); L = ax.length; ax.normalize()
        ds = []
        for v in body.data.vertices:
            for e in v.groups:
                if e.group == g.index and e.weight > 0.6:
                    p = mw @ v.co
                    u = (p - h).dot(ax)
                    if 0.2 * L < u < 0.8 * L:
                        ds.append(((p - h) - ax * u).length)
        radii[name] = sum(ds) / len(ds) if ds else None
    info["hands"][s] = {"bones": bones, "radius": radii}
json.dump(info, open(os.path.join(out_dir, "mpfb_hands.json"), "w"), indent=1)

for o in bpy.context.selected_objects:
    o.select_set(False)
arm.select_set(True); body.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(out_dir, "mpfb_hands.glb"), export_format="GLB",
                          use_selection=True, export_skins=True, export_animations=False,
                          export_morph=False, export_materials="NONE", export_yup=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "mpfb_hands_42.blend"))
print("verts", len(body.data.vertices), "bones", len(arm.data.bones))

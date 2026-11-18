"""Generate the CC0 MPFB2 body the drum renderer poses (Blender >= 4.2 with the MPFB2 extension).

    blender -b --factory-startup -P blockout/hand_model/mpfb_make_body.py -- OUT_DIR [ASSET_PACK_DIR]

ASSET_PACK_DIR is the unzipped CC0 MakeHuman system asset pack (makehuman_system_assets_cc0.zip from
static.makehumancommunity.org/assets/assetpacks); with it he is dressed (T-shirt and jeans, sneakers, short hair,
eyebrows), the skin under the clothes is cut away, and the asset materials are dropped (the renderer colours each
part flat), so the .blend needs nothing from the pack.

The same adult male as mpfb_make_hand.py (same macros, so the hands are the same hands), with local targets that
bring the rest of him toward the performer the drum solver moves (blockout/performer.py, drums.py): a shorter torso
and neck, narrower shoulders and ribcage (the arms are kept clear of a 0.13 x 0.10 m half-width ribcage), slimmer
upper arms, shorter and slimmer legs, smaller feet. Whatever length is left over the renderer takes up by
stretching each bone to the solved joints. The whole body surface and the eyes are kept, with the full default rig;
every bone the renderer places is unparented so each one can take its own world pose and length. Writes:
  OUT_DIR/mpfb_body.blend  "body" (skinned mesh) + "Human.rig" (armature)
  OUT_DIR/mpfb_body.json   rest bone heads/tails (metres, Blender world) and the targets used
MPFB2 code is GPL-3.0; the meshes it generates are CC0.
"""
import importlib, json, os, sys

import addon_utils
import bpy
import bmesh

TARGETS = {"torso-scale-vert-decr": 1.0, "torso-scale-horiz-decr": 1.0, "torso-scale-depth-decr": 1.0,
           "measure-neck-height-decr": 1.0,
           "lr-upperarm-scale-horiz-decr": 1.0, "lr-upperarm-scale-depth-decr": 1.0, "lr-upperarm-muscle-decr": 0.5,
           "lr-upperleg-scale-vert-decr": 1.0, "lr-lowerleg-scale-vert-decr": 1.0,
           "lr-upperleg-scale-horiz-decr": 1.0, "lr-upperleg-scale-depth-decr": 1.0, "lr-foot-scale-decr": 1.0,
           "stomach-pregnant-decr": 1.0, "measure-waist-circ-decr": 1.0}
# placed by the renderer (blender_drums.pose_body); everything else rides on its parent
PLACED = ["root", "spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03", "head"] + [
    f"{b}.{s}" for b in ("clavicle", "shoulder01", "upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist",
                         "pelvis", "upperleg01", "upperleg02", "lowerleg01", "lowerleg02", "foot") for s in "LR"]

ASSETS = (("clothes/male_casualsuit06/male_casualsuit06.mhclo", "Clothes", "suit"),
          ("clothes/shoes05/shoes05.mhclo", "Clothes", "shoes"),
          ("hair/short02/short02.mhclo", "Hair", "hair"),
          ("eyebrows/eyebrow001/eyebrow001.mhclo", "Eyebrows", "brows"))
LEG_BONES = ("root", "pelvis", "upperleg", "lowerleg")

args = sys.argv[sys.argv.index("--") + 1:]
out_dir = args[0]
pack = args[1] if len(args) > 1 else None
os.makedirs(out_dir, exist_ok=True)
for repo in ("user_default", "blender_org"):
    mod = f"bl_ext.{repo}.mpfb"
    try:
        addon_utils.enable(mod, default_set=True)
        HumanService = importlib.import_module(mod + ".services.humanservice").HumanService
        TargetService = importlib.import_module(mod + ".services.targetservice").TargetService
        break
    except ImportError:
        continue
else:
    raise SystemExit("MPFB2 extension not found")

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
macro = TargetService.get_default_macro_info_dict()
macro.update(gender=1.0, age=0.5, muscle=float(os.environ.get("MPFB_MUSCLE", 0.7)),
             weight=float(os.environ.get("MPFB_WEIGHT", 0.6)), height=0.75, proportions=0.6)
body = HumanService.create_human(mask_helpers=False, detailed_helpers=True, extra_vertex_groups=True,
                                 feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
stack = [{"target": t.replace("lr-", s, 1) if t.startswith("lr-") else t, "value": v}
         for t, v in TARGETS.items() for s in (("l-", "r-") if t.startswith("lr-") else ("",))]
TargetService.bulk_load_targets(body, stack)
HumanService.add_builtin_rig(body, "default", import_weights=True)     # fitted to the shaped mesh
arm = body.parent if body.parent and body.parent.type == "ARMATURE" else next(
    o for o in bpy.data.objects if o.type == "ARMATURE")
worn = {}
for rel, kind, nm in ASSETS if pack else ():
    bpy.context.view_layer.objects.active = body
    ob = HumanService.add_mhclo_asset(os.path.join(pack, rel), body, asset_type=kind, subdiv_levels=0)
    ob.name = ob.data.name = nm
    worn[nm] = ob

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

# the body surface and the eyeballs; the other helpers (tights, skirt, hair, teeth, tongue, joint cubes) go
gi = {g.name: g.index for g in body.vertex_groups}
keep_g = {gi[n] for n in ("body", "helper-l-eye", "helper-r-eye") if n in gi}
hidden_g = {i for n, i in gi.items() if n.startswith("Delete.")}       # skin the clothes cover
bm = bmesh.new(); bm.from_mesh(body.data)
dl = bm.verts.layers.deform.verify()
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not any(v[dl].get(g, 0) > 0 for g in keep_g)
                           or any(v[dl].get(g, 0) > 0 for g in hidden_g)], context="VERTS")
bm.to_mesh(body.data); bm.free()
eye_g = {gi[n] for n in ("helper-l-eye", "helper-r-eye") if n in gi}
eyes = [v.index for v in body.data.vertices if any(e.group in eye_g and e.weight > 0 for e in v.groups)]
bones = {b.name for b in arm.data.bones}
for g in list(body.vertex_groups):
    if g.name not in bones:
        body.vertex_groups.remove(g)
eg = body.vertex_groups.new(name="eyes")              # material tag for the renderer (no bone of that name)
eg.add(eyes, 1.0, "REPLACE")
for m in list(body.modifiers):
    if m.type != "ARMATURE":
        body.modifiers.remove(m)
body.name, arm.name = "body", "Human.rig"

# the worn assets: bone weights and the armature modifier stay; materials become named slots the renderer colours
# (the suit is one mesh: each of its pieces is shirt or jeans by which bones carry it)
for nm, ob in worn.items():
    me = ob.data
    for m in list(ob.modifiers):
        if m.type != "ARMATURE":
            ob.modifiers.remove(m)
    for g in list(ob.vertex_groups):
        if g.name not in bones:
            ob.vertex_groups.remove(g)
    me.materials.clear()
    if nm != "suit":
        me.materials.append(bpy.data.materials.new(nm))
        continue
    for k in ("shirt", "pants"):
        me.materials.append(bpy.data.materials.new(k))
    names = {g.index: g.name for g in ob.vertex_groups}
    legs = [sum(e.weight for e in v.groups if names[e.group].startswith(LEG_BONES)) /
            max(sum(e.weight for e in v.groups), 1e-9) for v in me.vertices]
    bm = bmesh.new(); bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    seen, shirt_v = set(), set()
    for f in bm.faces:
        if f.index in seen:
            continue
        island, todo = [], [f]
        seen.add(f.index)
        while todo:
            g = todo.pop()
            island.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index); todo.append(h)
        vs = {v.index for g in island for v in g.verts}
        idx = 1 if sum(legs[i] for i in vs) / len(vs) > 0.5 else 0
        for g in island:
            g.material_index = idx
        if idx == 0:
            shirt_v.update(vs)
    bm.to_mesh(me); bm.free()
    # a hem that rides the thighs balloons over a seated lap: the shirt hangs from the pelvis instead
    gid = {g.name: g for g in ob.vertex_groups}
    root = gid.get("root") or ob.vertex_groups.new(name="root")
    for i in shirt_v:
        moved = 0.0
        for gi_, w in [(e.group, e.weight) for e in me.vertices[i].groups]:
            if names[gi_].startswith(("upperleg", "lowerleg")):
                moved += w
                ob.vertex_groups[gi_].remove([i])
        if moved:
            root.add([i], moved, "ADD")
for im in list(bpy.data.images):
    bpy.data.images.remove(im)

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for n in PLACED:
    eb = arm.data.edit_bones[n]
    eb.use_connect = False
    eb.parent = None
bpy.ops.object.mode_set(mode="OBJECT")

info = {"source": "MPFB2 default rig, CC0 mesh", "macro": {k: v for k, v in macro.items() if k != "race"},
        "targets": TARGETS, "placed": PLACED, "worn": {nm: rel for rel, _, nm in ASSETS if nm in worn},
        "bones": {b.name: {"head": list(arm.matrix_world @ b.head_local), "tail": list(arm.matrix_world @ b.tail_local),
                           "parent": b.parent.name if b.parent else None} for b in arm.data.bones}}
json.dump(info, open(os.path.join(out_dir, "mpfb_body.json"), "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "mpfb_body.blend"))
print("verts", len(body.data.vertices), "eyes", len(eyes), "bones", len(arm.data.bones),
      {nm: (len(ob.data.vertices), [sum(p.material_index == i for p in ob.data.polygons)
                                    for i in range(len(ob.data.materials))]) for nm, ob in worn.items()})

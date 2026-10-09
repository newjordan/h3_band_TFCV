"""Blender (bpy) renderer for a piano blockout JSON from blockout/piano.py. Grey matte primitives, Workbench.

    blender -b --factory-startup -P blockout/blender_piano.py -- ANIM.json OUT_DIR [--res 1280x704]
           [--frames a:b] [--engine WORKBENCH|CYCLES] [--view pov|three4|side] [--lens mm]

pov rides the performer's eyes (head hidden); three4 is a concert three-quarter from the treble side;
side is a debug profile view for checking strikes and head nods.

Writes OUT_DIR/f_00000.png ... (one per frame). Everything the scene needs is in the JSON; this file only
builds geometry and poses it.
"""
import json, math, os, sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
anim_path, out_dir = argv[0], argv[1]
opts = dict(zip(argv[2::2], argv[3::2]))
W, H = map(int, opts.get("--res", "1280x704").split("x"))
ENGINE = opts.get("--engine", "WORKBENCH")
anim = json.load(open(anim_path))
frames = anim["frames"]
fa, fb = map(int, opts.get("--frames", f"0:{len(frames)}").split(":"))
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
COL = scn.collection


def mat(name, g):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (g, g, g, 1.0)
    m.roughness = 0.6
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (g, g, g, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return m


M_WHITE, M_BLACK, M_CASE, M_SKIN, M_FLOOR = (mat("white", 0.80), mat("black", 0.035), mat("case", 0.14),
                                             mat("skin", 0.42), mat("floor", 0.06))


def box(name, x0, x1, y0, y1, z0, z1, m, origin=None):
    """Axis-aligned box. origin: point the object pivots on (defaults to the box centre)."""
    o = Vector(origin) if origin else Vector(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((x0 if v.co.x < 0 else x1, y0 if v.co.y < 0 else y1, z0 if v.co.z < 0 else z1)) - o
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = o
    ob.data.materials.append(m)
    COL.objects.link(ob)
    return ob


# ------------------------------------------------------------------ piano
keys = {}
xs = [k["x"] for k in anim["keyboard"]]
X0, X1 = min(xs) - 0.02, max(xs) + 0.02
for k in anim["keyboard"]:
    x, w = k["x"], k["w"]
    gap = 0.0006
    if k["black"]:
        ob = box(f"key{k['pitch']}", x - w / 2, x + w / 2, k["y0"], 0.150, 0.0, k["top"], M_BLACK,
                 origin=(x, 0.150, 0.0))
    else:
        ob = box(f"key{k['pitch']}", x - w / 2 + gap, x + w / 2 - gap, 0.0, 0.150, -0.018, 0.0, M_WHITE,
                 origin=(x, 0.150, 0.0))
    keys[str(k["pitch"])] = ob
    ob["length"] = 0.150 - k["y0"]

box("keybed", X0 - 0.04, X1 + 0.04, -0.02, 0.30, -0.10, -0.019, M_CASE)
box("front_rail", X0 - 0.04, X1 + 0.04, -0.03, -0.004, -0.10, -0.012, M_CASE)
box("cheek_l", X0 - 0.06, X0 - 0.005, -0.03, 0.30, -0.10, 0.045, M_CASE)
box("cheek_r", X1 + 0.005, X1 + 0.06, -0.03, 0.30, -0.10, 0.045, M_CASE)
box("fallboard", X0 - 0.005, X1 + 0.005, 0.153, 0.20, -0.02, 0.055, M_CASE)
box("lid_body", X0 - 0.06, X1 + 0.06, 0.20, 0.32, -0.10, 0.07, M_CASE)
VIEW = opts.get("--view", "pov")
if VIEW == "pov":
    box("music_desk", X0 + 0.45, X1 - 0.45, 0.22, 0.25, 0.07, 0.30, M_CASE)
box("floor", X0 - 3, X1 + 3, -3, 4, -0.80, -0.79, M_FLOOR)

# ------------------------------------------------------------------ hands
def capsule_mesh(name, r0, r1):
    """Unit-length tapered capsule along +Z from 0 to 1 (caps are spheres of r0/r1 at the ends)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=r0, radius2=r1, depth=1.0,
                          matrix=Matrix.Translation((0, 0, 0.5)))
    bm.to_mesh(me); bm.free()
    return me


def sphere(name, r, m):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=r)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(m)
    COL.objects.link(ob)
    return ob


class Seg:
    """A tapered limb segment between two points, with sphere joints at both ends."""

    def __init__(self, name, r0, r1):
        me = capsule_mesh(name, r0, r1)
        for p in me.polygons:
            p.use_smooth = True
        self.ob = bpy.data.objects.new(name, me)
        self.ob.data.materials.append(M_SKIN)
        COL.objects.link(self.ob)
        self.j0, self.j1 = sphere(name + "_j0", r0, M_SKIN), sphere(name + "_j1", r1, M_SKIN)

    def set(self, a, b):
        a, b = Vector(a), Vector(b)
        d = b - a
        L = max(d.length, 1e-5)
        rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
        self.ob.matrix_world = Matrix.Translation(a) @ rot @ Matrix.Diagonal((1, 1, L, 1))
        self.j0.location, self.j1.location = a, b


FR = [(0.0115, 0.0105, 0.0095, 0.0085), (0.0095, 0.0088, 0.0080, 0.0070), (0.0098, 0.0090, 0.0082, 0.0072),
      (0.0093, 0.0085, 0.0078, 0.0068), (0.0085, 0.0078, 0.0070, 0.0062)]
rig = {}
for h in ("L", "R"):
    rig[h] = {"forearm": Seg(f"{h}_forearm", 0.036, 0.026), "upper": Seg(f"{h}_upper", 0.046, 0.038),
              "fingers": [[Seg(f"{h}_f{f}_{s}", FR[f][s], FR[f][s + 1]) for s in range(3)] for f in range(5)]}
    me = bpy.data.meshes.new(f"{h}_palm")
    ob = bpy.data.objects.new(f"{h}_palm", me)
    ob.data.materials.append(M_SKIN)
    COL.objects.link(ob)
    rig[h]["palm"] = ob


def mesh_ob(name):
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    ob.data.materials.append(M_SKIN)
    COL.objects.link(ob)
    return ob


torso = mesh_ob("torso")
neck = Seg("neck", 0.050, 0.045)
head = sphere("head", 1.0, M_SKIN)
nose = bpy.data.objects.new("nose", capsule_mesh("nose", 0.016, 0.004))
nose.data.materials.append(M_SKIN); COL.objects.link(nose)
HEAD_PARTS = [head, nose, neck.ob, neck.j0, neck.j1]
hp = Vector(frames[0]["body"]["hips"])
box("bench", hp.x - 0.36, hp.x + 0.36, hp.y - 0.16, hp.y + 0.16, -0.79, hp.z - 0.06, M_CASE)
for sgn in (-1, 1):
    hip, knee = hp + Vector((sgn * 0.10, 0.02, -0.02)), hp + Vector((sgn * 0.13, 0.40, -0.04))
    Seg(f"thigh{sgn}", 0.070, 0.055).set(hip, knee)
    Seg(f"shin{sgn}", 0.050, 0.040).set(knee, knee + Vector((0, 0.04, -0.48)))
if VIEW == "pov":
    for ob in HEAD_PARTS:
        ob.hide_render = True


def set_hull(ob, pts):
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(Vector(p))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(ob.data); bm.free()


def set_palm(ob, pts):
    bm = bmesh.new()
    for p in pts:
        for dz in (0.010, -0.010):
            bm.verts.new(Vector(p) + Vector((0, 0, dz)))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(ob.data); bm.free()


# ------------------------------------------------------------------ camera, light, render
cx = anim["centre_x"]
cam_d = bpy.data.cameras.new("cam")
cam_d.lens, cam_d.sensor_width = float(opts.get("--lens", 30)), 36.0
cam = bpy.data.objects.new("cam", cam_d)
COL.objects.link(cam)
FIXED = {"three4": ((cx + 0.95, 0.55, 0.80), (cx - 0.05, -0.30, 0.22)),
         "side": ((cx + 1.30, -0.30, 0.30), (cx, -0.30, 0.22))}
if VIEW in FIXED:
    eye, look = map(Vector, FIXED[VIEW])
    cam.location = eye
    cam.rotation_euler = (look - eye).to_track_quat("-Z", "Y").to_euler()
scn.camera = cam


def look_matrix(pos, fwd, up):
    f = Vector(fwd).normalized()
    r = f.cross(Vector(up)).normalized()
    u = r.cross(f)
    m = Matrix(((r.x, u.x, -f.x), (r.y, u.y, -f.y), (r.z, u.z, -f.z))).to_4x4()
    m.translation = Vector(pos)
    return m

scn.render.resolution_x, scn.render.resolution_y = W, H
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = "PNG"
world = bpy.data.worlds.new("w"); scn.world = world
world.color = (0.03, 0.03, 0.03)
if ENGINE == "CYCLES":
    scn.render.engine = "CYCLES"
    scn.cycles.samples = 16
    scn.cycles.use_denoising = True
    sun = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    sun.data.energy, sun.data.size = 150, 1.5
    sun.location = (cx - 0.3, -0.6, 1.4)
    sun.rotation_euler = (Vector((cx, 0.1, 0)) - sun.location).to_track_quat("-Z", "Y").to_euler()
    COL.objects.link(sun)
else:
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_shadows, sh.show_cavity = True, True
    sh.cavity_type = "BOTH"
    sh.background_type = "WORLD"
    scn.display.shadow_shift = 0.05
    scn.display.light_direction = (-0.35, -0.45, 0.82)
    scn.display.render_aa = "16"

for i in range(fa, min(fb, len(frames))):
    fr = frames[i]
    for p, ob in keys.items():
        d = fr["keys"].get(p, 0.0)
        ob.rotation_euler = (-(d * 0.010) / ob["length"], 0, 0)
    for h, hd in fr["hands"].items():
        R = rig[h]
        R["forearm"].set(hd["elbow"], hd["wrist"])
        for f, chain in enumerate(hd["fingers"]):
            for s in range(3):
                R["fingers"][f][s].set(chain[s], chain[s + 1])
        w = Vector(hd["wrist"])
        side = Vector((0.026, 0, 0))
        knuckles = [hd["fingers"][f][0] for f in range(1, 5)]
        set_palm(R["palm"], knuckles + [w + side, w - side, hd["fingers"][0][1]])
        R["upper"].set(fr["body"]["shoulders"][h], hd["elbow"])
    b = fr["body"]
    Rt = Matrix(b["torso_rot"])
    hips, chest = Vector(b["hips"]), Vector(b["chest"])
    pts = []
    for sx, sz, sy in ((0.15, 0.0, 0.10), (0.21, 0.40, 0.11), (0.17, 0.53, 0.08)):
        for sgn in (-1, 1):
            for dy in (-sy, sy):
                pts.append(hips + Rt @ Vector((sgn * sx, dy, sz)))
    set_hull(torso, pts)
    neck.set(chest, b["neck"])
    Rh = Matrix(b["head_rot"])
    head.matrix_world = Matrix.Translation(Vector(b["head"])) @ Rh.to_4x4() @ Matrix.Diagonal((0.078, 0.095, 0.115, 1))
    nb = Vector(b["head"]) + Rh @ Vector((0, 0.085, 0.0))
    nose.matrix_world = (Matrix.Translation(nb) @ (Rh @ Matrix.Rotation(-math.pi / 2, 3, "X")).to_4x4()
                         @ Matrix.Diagonal((1, 1, 0.035, 1)))
    if VIEW == "pov":
        cam.matrix_world = look_matrix(b["eye"], b["cam_fwd"], b["cam_up"])
    scn.render.filepath = os.path.join(out_dir, f"f_{i:05d}.png")
    bpy.ops.render.render(write_still=True)
print("done", fa, fb)

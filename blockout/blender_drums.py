"""Blender (bpy) renderer for a drum blockout JSON from blockout/drums.py. Grey matte primitives, Workbench.

    blender -b --factory-startup -P blockout/blender_drums.py -- ANIM.json OUT_DIR [--res 1280x704]
           [--frames a:b] [--engine WORKBENCH|CYCLES] [--view front|three4|side|over|top] [--lens mm]
           [--cam ex,ey,ez,lx,ly,lz]

front is the audience view, three4 a stage-left three-quarter, side a debug profile for checking strokes,
over an over-the-shoulder view, top a debug plan view. --cam sets any eye and look-at point (kit metres:
x to the drummer's right, y toward the audience, z up) to match a shot's camera.

Writes OUT_DIR/f_00000.png ... (one per frame). Everything the scene needs is in the JSON; this file only
builds geometry and poses it.
"""
import json, math, os, sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
anim_path, out_dir = argv[0], os.path.abspath(argv[1])
opts = dict(zip(argv[2::2], argv[3::2]))
W, H = map(int, opts.get("--res", "1280x704").split("x"))
ENGINE = opts.get("--engine", "WORKBENCH")
VIEW = opts.get("--view", "front")
anim = json.load(open(anim_path))
frames, KIT = anim["frames"], anim["kit"]
fa, fb = map(int, opts.get("--frames", f"0:{len(frames)}").split(":"))
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
COL = scn.collection


def mat(name, g):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (g, g, g, 1.0)
    m.roughness = 0.6
    return m


M = {k: mat(k, g) for k, g in (("skin", 0.42), ("stick", 0.86), ("head", 0.80), ("shell", 0.20), ("hoop", 0.50),
                               ("metal", 0.58), ("hw", 0.33), ("floor", 0.06), ("rug", 0.13), ("seat", 0.12))}


def link(ob, m=None, parent=None):
    if m:
        ob.data.materials.append(m)
    COL.objects.link(ob)
    if parent:
        ob.parent = parent
    return ob


def mesh_from(name, build):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    build(bm)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me); bm.free()
    return bpy.data.objects.new(name, me)


def cyl(name, r0, r1, z0, z1, m, parent=None, segs=40, caps=True):
    """Cylinder/cone along local z from z0 (radius r0) to z1 (radius r1)."""
    ob = mesh_from(name, lambda bm: bmesh.ops.create_cone(
        bm, cap_ends=caps, segments=segs, radius1=r0, radius2=r1, depth=abs(z1 - z0),
        matrix=Matrix.Translation((0, 0, (z0 + z1) / 2)) @ (Matrix() if z1 > z0 else Matrix.Rotation(math.pi, 4, "X"))))
    return link(ob, m, parent)


def rod(name, a, b, r, m):
    a, b = Vector(a), Vector(b)
    ob = cyl(name, r, r, 0.0, 1.0, m, segs=12)
    d = b - a
    ob.matrix_world = (Matrix.Translation(a) @ Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
                       @ Matrix.Diagonal((1, 1, max(d.length, 1e-4), 1)))
    return ob


def tripod(name, top, foot_r=0.26, z_split=0.30):
    top = Vector(top)
    hub = Vector((top.x, top.y, z_split))
    rod(name + "_post", hub, top, 0.012, M["hw"])
    for k in range(3):
        a = math.radians(90 + 120 * k)
        rod(f"{name}_leg{k}", hub, (top.x + foot_r * math.cos(a), top.y + foot_r * math.sin(a), 0.01), 0.009, M["hw"])


def sphere(name, r, m, parent=None):
    ob = mesh_from(name, lambda bm: bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=r))
    return link(ob, m, parent)


BASE = {}    # object name -> (rest matrix_world pieces) for the per-frame pose


def pivot(name, c, n):
    e = bpy.data.objects.new(name, None)
    COL.objects.link(e)
    BASE[name] = (Vector(c), Vector((0, 0, 1)).rotation_difference(Vector(n)).to_matrix().to_4x4())
    e.matrix_world = Matrix.Translation(BASE[name][0]) @ BASE[name][1]
    return e


# ------------------------------------------------------------------ kit
pivots = {}
for p, k in KIT.items():
    c, n, r = Vector(k["c"]), Vector(k["n"]), k["r"]
    e = pivots[p] = pivot(p, c, n)
    if k["kind"] in ("drum", "kick"):
        d = k["depth"]
        cyl(p + "_head", r, r, -0.004, 0.0, M["head"], e)
        cyl(p + "_shell", r * 0.995, r * 0.995, -d, -0.004, M["shell"], e, caps=False)
        cyl(p + "_hoop", r + 0.008, r + 0.008, -0.012, 0.006, M["hoop"], e, caps=False)
        cyl(p + "_hoop2", r + 0.008, r + 0.008, -d - 0.006, -d + 0.012, M["hoop"], e, caps=False)
        if k["kind"] == "kick":
            cyl(p + "_reso", r, r, -d, -d + 0.004, M["head"], e)
    else:
        if p.rstrip("0123456789") == "china":      # edge turned up (drum_collide.piece_parts)
            cyl(p + "_bow", r, 0.03, 0.012, 0.0, M["metal"], e, segs=48)
        else:
            cyl(p + "_bow", r, 0.03, -0.003, 0.009, M["metal"], e, segs=48)
        cyl(p + "_bell", 0.055, 0.02, 0.006, 0.026, M["metal"], e)
        if k["kind"] == "hihat":
            cyl(p + "_bottom", r, 0.03, -0.010, -0.004, M["metal"], e, segs=48)
# stands, tom mounts and floor-tom legs: the hitbox rods (drum_collide.stands); a post gets tripod feet
FEET = {"snare": 0.22, "hihat": 0.24, "tom": 0.24}
if "stands" not in anim:
    print("blender_drums: this JSON has no 'stands' (made before drums.py wrote them); drawing no hardware")
for name, a, b, r in anim.get("stands", ()):
    if name.endswith("_stand"):
        owner = name[:-len("_stand")].rstrip("0123456789")
        tripod(name, b, FEET.get(owner, 0.28), a[2])
    else:
        rod(name, a, b, r, M["hw"])
# hi-hat top cymbal (bow + bell) rides up when the pedal opens
hat_moving = [ob for ob in bpy.data.objects if ob.name in ("hihat_bow", "hihat_bell")]

# pedals
kp, hp = anim["kick_pedal"], anim["hat_pedal"]


def pedal(name, heel, dirxy, L):
    yaw = math.atan2(-dirxy[0], dirxy[1])
    base = Matrix.Translation(Vector(heel)) @ Matrix.Rotation(yaw, 4, "Z")
    frame = mesh_from(name + "_frame", lambda bm: bmesh.ops.create_cube(bm, size=1.0))
    link(frame, M["hw"])
    frame.matrix_world = base @ Matrix.Translation((0, L / 2, -0.015)) @ Matrix.Diagonal((0.09, L + 0.04, 0.02, 1))
    bd = mesh_from(name + "_board", lambda bm: bmesh.ops.create_cube(bm, size=1.0))
    link(bd, M["hw"])
    BASE[bd.name] = base
    bd.data.transform(Matrix.Translation((0, L / 2, 0.006)) @ Matrix.Diagonal((0.075, L, 0.012, 1)))
    return bd


kick_board = pedal("kick_pedal", kp["heel"], (0, 1), kp["len"])
hat_board = pedal("hat_pedal", hp["heel"], hp["dir"], hp["len"])
bp = Vector(kp["beater_pivot"])
rod("beater_post_l", (bp.x - 0.05, bp.y, 0.0), (bp.x - 0.05, bp.y, bp.z + 0.02), 0.008, M["hw"])
rod("beater_post_r", (bp.x + 0.05, bp.y, 0.0), (bp.x + 0.05, bp.y, bp.z + 0.02), 0.008, M["hw"])
beater = bpy.data.objects.new("beater", None); COL.objects.link(beater)
cyl("beater_shaft", 0.005, 0.005, 0.0, kp["beater_len"] - 0.02, M["hw"], beater, segs=12)
cyl("beater_ball", 0.032, 0.032, kp["beater_len"] - 0.03, kp["beater_len"] + 0.01, M["head"], beater, segs=24)
beater2 = None
if "heel2" in kp:       # double pedal: second footboard, its beater beside the first, a drive shaft between
    kick_board2 = pedal("kick_pedal2", kp["heel2"], (0, 1), kp["len"])
    bp2 = Vector(kp["beater_pivot2"])
    beater2 = bpy.data.objects.new("beater2", None); COL.objects.link(beater2)
    cyl("beater2_shaft", 0.005, 0.005, 0.0, kp["beater_len"] - 0.02, M["hw"], beater2, segs=12)
    cyl("beater2_ball", 0.032, 0.032, kp["beater_len"] - 0.03, kp["beater_len"] + 0.01, M["head"], beater2, segs=24)
    h2 = Vector(kp["heel2"])
    rod("drive_shaft", (h2.x, h2.y + kp["len"] + 0.02, 0.05), (bp2.x - 0.05, bp2.y, bp2.z), 0.006, M["hw"])

# throne, rug, floor
hips0 = Vector(frames[0]["body"]["hips"])
cyl("throne_seat", 0.19, 0.19, 0.46, 0.53, M["seat"]).location = (hips0.x, hips0.y, 0)
tripod("throne", (hips0.x, hips0.y, 0.46), 0.28, 0.18)
cyl("rug", 1.35, 1.35, 0.0, 0.006, M["rug"], segs=64).location = (0.0, 0.25, 0)
fl = mesh_from("floor", lambda bm: bmesh.ops.create_cube(bm, size=1.0)); link(fl, M["floor"])
fl.matrix_world = Matrix.Translation((0, 1.0, -0.01)) @ Matrix.Diagonal((12, 12, 0.02, 1))


# ------------------------------------------------------------------ performer
def capsule_mesh(name, r0, r1):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=20, radius1=r0, radius2=r1, depth=1.0,
                          matrix=Matrix.Translation((0, 0, 0.5)))
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me); bm.free()
    return me


class Seg:
    """A tapered limb segment between two points, with sphere joints at both ends."""

    def __init__(self, name, r0, r1, m=None):
        m = m or M["skin"]
        self.ob = link(bpy.data.objects.new(name, capsule_mesh(name, r0, r1)), m)
        self.j0, self.j1 = sphere(name + "_j0", r0, m), sphere(name + "_j1", r1, m)

    def set(self, a, b):
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix().to_4x4()
        self.ob.matrix_world = Matrix.Translation(a) @ rot @ Matrix.Diagonal((1, 1, max(d.length, 1e-5), 1))
        self.j0.location, self.j1.location = a, b


def hull_ob(name, m=None):
    return link(bpy.data.objects.new(name, bpy.data.meshes.new(name)), m or M["skin"])


def set_hull(ob, pts):
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(Vector(p))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(ob.data); bm.free()


# ------------------------------------------------------------------ MPFB2 hands (blockout/hand_model)
# The piano's rigged, skinned CC0 hand. drum_hands.py solves the grip on the same rig (blockout/rig), so each
# frame's "root" and "angles" pose the bones by pure forward kinematics, as blender_piano.pose_rig does.
HANDS = "angles" in frames[0]["hands"]["R"]         # older JSONs get a fist
HAND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_model")


def load_mpfb():
    with bpy.data.libraries.load(os.path.join(HAND_DIR, "mpfb_hands.blend")) as (src, dst):
        dst.objects = ["hands", "Human.rig"]
    mesh, arm = dst.objects
    for o in (mesh, arm):
        COL.objects.link(o)
    mesh.data.materials.clear()
    mesh.data.materials.append(M["skin"])
    for p in mesh.data.polygons:
        p.use_smooth = True
    sub = mesh.modifiers.new("smooth", "SUBSURF")      # after the armature: smooth the posed mesh
    sub.levels, sub.render_levels = 1, 2
    arm.hide_render = True
    bones = {s: {b.name[:-2]: b for b in arm.data.bones if b.name.endswith("." + s)} for s in "LR"}
    pts = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    w, e = bones["R"]["wrist"].head_local, bones["R"]["lowerarm01"].head_local
    u = (w - e).normalized()
    along = [(p - e).dot(u) for p in pts]
    near = [i for i, a in enumerate(along) if a > -0.01 and (pts[i] - e - u * along[i]).length < 0.08]
    cut = min(along[i] for i in near)       # where the skinned forearm starts, m from the elbow
    full = (w - e).length - cut > 0.2       # the skin runs (almost) to the elbow
    cut_r = max((pts[i] - e - u * along[i]).length for i in near if along[i] < cut + 0.01)
    end_r = min(((pts[i] - e - u * along[i]).length for i in near if abs(along[i] - cut - 0.012) < 0.008),
                default=cut_r)              # the skin's width where the sleeve ends: the sleeve tucks under it
    return arm, bones, full, (cut, cut_r, end_r)


_SK = {}


def pose_hand(arm, B, s, hd):
    """The wrist bone takes the solved root pose, every finger bone its joint rotation about its own rest axes,
    and the two forearm bones run from the elbow to the wrist, sharing some of the hand's roll."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from blockout.rig.skeleton import Skeleton
    if s not in _SK:
        _SK[s] = Skeleton(s)
    sk = _SK[s]
    B = B[s]
    wrist, elbow = Vector(hd["wrist"]), Vector(hd["elbow"])
    u = (wrist - elbow).normalized()
    root = Matrix([list(r) for r in hd["root"]["rot"]]).to_4x4()
    root.translation = Vector(hd["root"]["pos"])
    want = {"wrist": root @ Matrix.Diagonal((sk.scale, sk.scale, sk.scale, 1.0))}   # the rig's hand scale
    for nm, back, share in (("lowerarm02", 0.0, 0.6), ("lowerarm01", B["lowerarm02"].length, 0.2)):
        head = wrist - u * (B["lowerarm02"].length + back)
        r3 = B[nm].matrix_local.to_3x3()
        r3 = r3.col[1].normalized().rotation_difference(u).to_matrix() @ r3
        rigid = root.to_3x3() @ B["wrist"].matrix_local.to_3x3().inverted() @ B[nm].matrix_local.to_3x3()
        za, zb = r3.col[2] - u * r3.col[2].dot(u), rigid.col[2] - u * rigid.col[2].dot(u)
        if za.length > 1e-6 and zb.length > 1e-6:
            ang = za.normalized().angle(zb.normalized())
            if za.normalized().cross(zb.normalized()).dot(u) < 0:
                ang = -ang
            r3 = Matrix.Rotation(ang * share, 3, u) @ r3
        m = r3.to_4x4()
        m.translation = head
        want[nm] = m

    def basis(nm):
        b = B[nm]
        if b.parent is None:
            return b.matrix_local.inverted() @ want[nm]
        pn = b.parent.name[:-2]
        return (b.parent.matrix_local.inverted() @ b.matrix_local).inverted() @ want[pn].inverted() @ want[nm]

    for nm in ("lowerarm01", "lowerarm02", *sk.names):
        pb = arm.pose.bones[f"{nm}.{s}"]
        if nm in want:
            pb.matrix_basis = basis(nm)
        else:
            pb.matrix_basis = Matrix([list(r) for r in sk.local_rot(nm, hd["angles"].get(nm, {}))]).to_4x4()


FULL_FOREARM = False
if HANDS:
    MPFB_ARM, MPFB_BONES, FULL_FOREARM, (SKIN_CUT, SKIN_CUT_R, SKIN_END_R) = load_mpfb()
rig = {}
for h in ("L", "R"):
    # a skinned forearm that runs to the elbow is cut off just short of it: a sleeve a little wider than the cut
    # covers the elbow end and the cut edge, its tapered end tucked into the skin so no open edge shows
    fore = Seg(f"{h}_fore", SKIN_CUT_R + 0.003, max(SKIN_END_R - 0.002, 0.008)) if FULL_FOREARM \
        else Seg(f"{h}_fore", 0.036, 0.028)
    rig[h] = {"upper": Seg(f"{h}_upper", 0.046, 0.038), "fore": fore,
              "stick": Seg(f"{h}_stick", 0.0075, 0.0055, M["stick"]),
              "thigh": Seg(f"{h}_thigh", 0.075, 0.058), "shin": Seg(f"{h}_shin", 0.055, 0.042),
              "foot": hull_ob(f"{h}_foot")}
    if not HANDS:
        rig[h]["fist"] = sphere(f"{h}_fist", 1.0, M["skin"])
torso, pelvis = hull_ob("torso"), hull_ob("pelvis")
neck = Seg("neck", 0.050, 0.045)
head = sphere("head", 1.0, M["skin"])
nose = link(bpy.data.objects.new("nose", capsule_mesh("nose", 0.016, 0.004)), M["skin"])

# ------------------------------------------------------------------ camera, light, render
cam_d = bpy.data.cameras.new("cam")
cam_d.lens, cam_d.sensor_width = float(opts.get("--lens", 32)), 36.0
cam = bpy.data.objects.new("cam", cam_d)
COL.objects.link(cam)
VIEWS = {"front": ((0.10, 2.60, 1.50), (0.0, 0.10, 0.80)),
         "three4": ((-2.00, 2.30, 1.65), (0.0, 0.10, 0.78)),
         "side": ((2.60, 0.10, 0.95), (0.0, 0.10, 0.75)),
         "over": ((0.75, -1.10, 1.95), (-0.10, 0.45, 0.70)),
         "top": ((0.0, 0.15, 3.6), (0.0, 0.16, 0.0))}
if "--cam" in opts:
    v = [float(x) for x in opts["--cam"].split(",")]
    eye, look = Vector(v[:3]), Vector(v[3:6])
else:
    eye, look = map(Vector, VIEWS[VIEW])
cam.location = eye
cam.rotation_euler = (look - eye).to_track_quat("-Z", "Y" if VIEW != "top" else "Y").to_euler()
scn.camera = cam

scn.render.resolution_x, scn.render.resolution_y = W, H
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = "PNG"
world = bpy.data.worlds.new("w"); scn.world = world
world.color = (0.03, 0.03, 0.03)
if ENGINE == "CYCLES":
    scn.render.engine = "CYCLES"
    scn.cycles.samples = 16
    scn.cycles.use_denoising = True
    key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
    key.data.energy, key.data.size = 400, 2.0
    key.location = (-1.0, 1.8, 2.6)
    key.rotation_euler = (Vector((0, 0.2, 0.7)) - key.location).to_track_quat("-Z", "Y").to_euler()
    COL.objects.link(key)
else:
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_shadows, sh.show_cavity = True, True
    sh.cavity_type = "BOTH"
    sh.background_type = "WORLD"
    scn.display.shadow_shift = 0.05
    scn.display.light_direction = (-0.35, 0.45, 0.82)
    scn.display.render_aa = "16"


def foot_pts(fd, side):
    across = Vector((0.045, 0, 0))
    pts = []
    for key, dz in (("heel", 0.0), ("ankle", -0.02), ("ball", 0.0), ("toe", 0.0)):
        p = Vector(fd[key])
        w = 0.6 if key == "ankle" else 1.0
        for s in (-1, 1):
            pts += [p + across * s * w + Vector((0, 0, dz - 0.02)), p + across * s * w + Vector((0, 0, dz + 0.02))]
    return pts


for i in range(fa, min(fb, len(frames))):
    fr = frames[i]
    for p, e in pivots.items():
        c, base = BASE[p]
        aa = Vector(fr["tilt"].get(p, (0, 0, 0)))
        R = Matrix.Rotation(aa.length, 4, aa.normalized()) if aa.length > 1e-6 else Matrix()
        e.matrix_world = Matrix.Translation(c) @ R @ base
    lift = fr["hihat"]["open"] * anim["hihat_open"]
    for ob in hat_moving:
        ob.location = (0, 0, lift)
    kick_board.matrix_world = BASE[kick_board.name] @ Matrix.Rotation(fr["kick"]["board"], 4, "X")
    hat_board.matrix_world = BASE[hat_board.name] @ Matrix.Rotation(fr["hihat"]["board"], 4, "X")
    beater.matrix_world = Matrix.Translation(bp) @ Matrix.Rotation(-fr["kick"]["beater"], 4, "X")
    if beater2:
        kick_board2.matrix_world = BASE[kick_board2.name] @ Matrix.Rotation(fr["kick"]["board2"], 4, "X")
        beater2.matrix_world = Matrix.Translation(bp2) @ Matrix.Rotation(-fr["kick"]["beater2"], 4, "X")
    for h, hd in fr["hands"].items():
        R = rig[h]
        R["upper"].set(hd["shoulder"], hd["elbow"])
        R["fore"].set(hd["elbow"], hd["wrist"])
        R["stick"].set(hd["butt"], hd["tip"])
        if HANDS:
            el, wr = Vector(hd["elbow"]), Vector(hd["wrist"])
            if FULL_FOREARM:        # the skinned forearm runs to the elbow: a short sleeve covers its cut edge
                R["fore"].set(el, el + (wr - el).normalized() * (SKIN_CUT + 0.012))
                R["fore"].j1.hide_render = True
            else:                   # the primitive forearm stops short and swallows the hand's forearm stub
                R["fore"].set(el, wr - (wr - el).normalized() * 0.05)
            pose_hand(MPFB_ARM, MPFB_BONES, h, hd)
        else:
            d = Vector(hd["hand_dir"]) if "hand_dir" in hd else (Vector(hd["tip"]) - Vector(hd["butt"])).normalized()
            R["fist"].matrix_world = (Matrix.Translation(Vector(hd["grip"]) - d * 0.012)
                                      @ Vector((0, 0, 1)).rotation_difference(d).to_matrix().to_4x4()
                                      @ Matrix.Diagonal((0.042, 0.038, 0.055, 1)))
        fd = fr["feet"][h]
        R["thigh"].set(fd["hip"], fd["knee"])
        R["shin"].set(fd["knee"], fd["ankle"])
        set_hull(R["foot"], foot_pts(fd, h))
    b = fr["body"]
    Rt = Matrix(b["torso_rot"])
    hips = Vector(b["hips"])
    pts = []
    for sx, sz, sy in ((0.15, 0.0, 0.10), (0.21, 0.40, 0.11), (0.17, 0.53, 0.08)):
        for sgn in (-1, 1):
            for dy in (-sy, sy):
                pts.append(hips + Rt @ Vector((sgn * sx, dy, sz)))
    set_hull(torso, pts)
    pp = []
    for side in ("L", "R"):
        hj = Vector(fr["feet"][side]["hip"])
        for dx, dy, dz in ((0.07, 0, 0), (-0.07, 0, 0), (0, 0.07, 0), (0, -0.09, 0), (0, 0, 0.08), (0, 0, -0.06)):
            pp.append(hj + Vector((dx, dy, dz)))
    set_hull(pelvis, pp)
    neck.set(b["chest"], b["neck"])
    Rh = Matrix(b["head_rot"])
    head.matrix_world = Matrix.Translation(Vector(b["head"])) @ Rh.to_4x4() @ Matrix.Diagonal((0.078, 0.095, 0.115, 1))
    nb = Vector(b["head"]) + Rh @ Vector((0, 0.085, 0.0))
    nose.matrix_world = (Matrix.Translation(nb) @ (Rh @ Matrix.Rotation(-math.pi / 2, 3, "X")).to_4x4()
                         @ Matrix.Diagonal((1, 1, 0.035, 1)))
    scn.render.filepath = os.path.join(out_dir, f"f_{i:05d}.png")
    bpy.ops.render.render(write_still=True)
print("done", fa, fb)

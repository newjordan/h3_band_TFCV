"""Blender (bpy) renderer for a drum blockout JSON from blockout/drums.py. Grey matte, Workbench.

The drummer is the CC0 MPFB2 man in blockout/hand_model/mpfb_body.blend (mpfb_make_body.py), dressed, posed bone by
bone onto the solved joints; without that file (or with DRUM_BODY=prims) he is the older primitive mannequin with
the MPFB2 hands.

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


# ------------------------------------------------------------------ MPFB2 whole body (blockout/hand_model/mpfb_body.*)
# The same CC0 man as the hands, shaped toward the solver's performer (mpfb_make_body.py). Every bone the solver
# has a joint for is unparented in that file, so each takes a world pose here: its head on the solved joint, its
# length stretched to the solved segment, its roll from the limb's bend plane. Fingers and toes ride their parents.
def _frame(a, b):
    a = a.normalized()
    b = (b - a * b.dot(a)).normalized()
    return Matrix((a, b, a.cross(b))).transposed()


def _minrot(a, b):
    return a.normalized().rotation_difference(b.normalized()).to_matrix()


def _signed(a, b, axis):
    a, b = a - axis * a.dot(axis), b - axis * b.dot(axis)
    if a.length < 1e-9 or b.length < 1e-9:
        return 0.0
    ang = a.angle(b)
    return ang if a.cross(b).dot(axis) >= 0 else -ang


def _slerp(Q, f):
    return Matrix.Identity(3).to_quaternion().slerp(Q.to_quaternion(), f).to_matrix()


ZONES = (("skin", None), ("shirt", 0.16), ("pants", 0.10), ("shoes", 0.05), ("eyes", 0.26))
WORN = {"shirt": (0.13, 0.13, 0.14), "pants": (0.06, 0.07, 0.10), "shoes": (0.55, 0.55, 0.55),
        "hair": (0.045, 0.04, 0.035), "brows": (0.05, 0.045, 0.04)}


def _zone(bone):
    if bone.startswith(("spine", "clavicle", "shoulder01", "breast", "upperarm01")):
        return 1
    if bone.startswith(("root", "pelvis", "upperleg", "lowerleg")):
        return 2
    if bone.startswith(("foot", "toe")):
        return 3
    return 0


class MPFBBody:
    SPINE = (("spine05", 0.15), ("spine04", 0.35), ("spine03", 0.6), ("spine02", 0.85), ("spine01", 1.0))
    NECK = (("neck01", 0.25), ("neck02", 0.5), ("neck03", 0.75), ("head", 1.0))
    TOES = [f"toe{t}-1" for t in range(1, 6)]

    def __init__(self, fr0):
        with bpy.data.libraries.load(os.path.join(HAND_DIR, "mpfb_body.blend")) as (src, dst):
            dst.objects = list(src.objects)
        obs = {o.name: o for o in dst.objects}
        self.mesh, self.arm = obs.pop("body"), obs.pop("Human.rig")
        self.worn = obs                     # suit, shoes, hair, brows (mpfb_make_body.py with the asset pack)
        for o in (self.mesh, self.arm, *self.worn.values()):
            COL.objects.link(o)
        self.arm.hide_render = True
        me = self.mesh.data
        me.materials.clear()
        for k, g in ZONES:
            me.materials.append(M["skin"] if g is None else mat(k, g))
        names = {g.index: g.name for g in self.mesh.vertex_groups}
        zone = []
        for v in me.vertices:
            gs = [(e.weight, names[e.group]) for e in v.groups if e.weight > 0]
            # dressed: the skin is skin; bare: shirt, shorts and shoes painted on by the bones under them
            zone.append(4 if any(n == "eyes" for _, n in gs) else 0 if self.worn or not gs else _zone(max(gs)[1]))
        for p in me.polygons:
            z = [zone[i] for i in p.vertices]
            p.material_index = max(set(z), key=z.count)
            p.use_smooth = True
        for o in (self.mesh, *self.worn.values()):
            if o is not self.mesh:
                for slot in o.material_slots:
                    m = slot.material
                    m.diffuse_color = (*WORN.get(m.name.split(".")[0], (0.3, 0.3, 0.3)), 1.0)
                    m.roughness = 0.7
                for p in o.data.polygons:
                    p.use_smooth = True
            sub = o.modifiers.new("smooth", "SUBSURF")
            sub.levels, sub.render_levels = 1, 2 if o is self.mesh else 1
        bones = self.arm.data.bones
        self.Lc = {b.name: b.matrix_local.copy() for b in bones}
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from blockout.rig.skeleton import Skeleton
        self.sk = {s: Skeleton(s) for s in "LR"}
        # rest hip-joint centre onto the solver's, turned to face the audience (+y)
        hj = (bones["upperleg01.L"].head_local + bones["upperleg01.R"].head_local) / 2
        self.hc = (Vector(fr0["feet"]["L"]["hip"]) + Vector(fr0["feet"]["R"]["hip"])) / 2
        self.A = Matrix.Translation(self.hc) @ Matrix.Rotation(math.pi, 4, "Z") @ Matrix.Translation(-hj)
        self.Ai = self.A.inverted()
        self.arm.matrix_world = self.A
        W = {n: self.A @ m for n, m in self.Lc.items()}
        self.h0 = {n: m.translation.copy() for n, m in W.items()}
        self.R0 = {n: m.to_3x3() for n, m in W.items()}
        # the solver's torso stands straight over its hips pivot; the rest body leans a little forward of that,
        # so it sits back about its hip joints by the difference (a seated pelvis rolls back anyway)
        b0 = fr0["body"]
        Rt0 = Matrix(b0["torso_rot"])
        sm = (Vector(b0["shoulders"]["L"]) + Vector(b0["shoulders"]["R"])) / 2 - Vector(b0["hips"])
        sm_neutral = Vector(b0["hips"]) + Rt0.transposed() @ sm
        sm0 = (self.h0["upperarm01.L"] + self.h0["upperarm01.R"]) / 2
        self.C = _minrot(sm0 - self.hc, sm_neutral - self.hc)
        # foot landmarks from the sole: heel (back), ball (under the toe joints), toe (front)
        mw = self.A
        pts = {s: [] for s in "LR"}
        for o in [self.mesh] + [o for nm, o in self.worn.items() if nm == "shoes"]:      # the shoe sole if worn
            gn = {g.index: g.name for g in o.vertex_groups}
            for v in o.data.vertices:
                gs = [(e.weight, gn[e.group]) for e in v.groups if e.weight > 0.5]
                if gs and max(gs)[1].startswith(("foot", "toe")):
                    pts[max(gs)[1][-1]].append(mw @ v.co)
        self.foot0 = {}
        for s, P in pts.items():
            zmin = min(p.z for p in P)
            sole = [p for p in P if p.z < zmin + 0.012]
            heel = min(sole, key=lambda p: p.y)
            toe = max(P, key=lambda p: p.y)
            ball = sum((self.h0[f"{t}.{s}"] for t in self.TOES), Vector()) / 5
            ball = Vector((ball.x, ball.y, heel.z))
            self.foot0[s] = (heel, ball, Vector((toe.x, toe.y, heel.z)))
        self.n_prev = {}

    def _put(self, n, pos, Q, scale=(1.0, 1.0, 1.0)):
        """World pose of an unparented bone: head at pos, rest frame turned by Q, scaled in its own axes."""
        P = Matrix.Translation(pos) @ (Q @ self.R0[n]).to_4x4() @ Matrix.Diagonal((*scale, 1.0))
        self.arm.pose.bones[n].matrix_basis = self.Lc[n].inverted() @ self.Ai @ P
        return P

    def _bend_axis(self, key, a, b, fallback):
        n = a.cross(b)
        if n.length < 0.03 * a.length * b.length:
            n = self.n_prev.get(key, fallback)
        n = n.normalized()
        self.n_prev[key] = n
        return n

    def pose(self, fr):
        b = fr["body"]
        Rt, Rh = Matrix(b["torso_rot"]), Matrix(b["head_rot"])
        C, hc = self.C, self.hc

        def rigid(n, Q, piv):
            return self._put(n, piv + Q @ (self.h0[n] - piv), Q)
        rigid("root", C, hc)
        Qs = {}
        p = hc + C @ (self.h0["spine05"] - hc)
        prev = None
        for n, f in self.SPINE:
            if prev:
                p = p + Qs[prev] @ (self.h0[n] - self.h0[prev])
            Qs[n] = _slerp(Rt, f) @ C
            self._put(n, p, Qs[n])
            prev = n
        top, Qtop = p, Qs["spine01"]
        for n, f in self.NECK:
            p = p + Qs[prev] @ (self.h0[n] - self.h0[prev])
            Qs[n] = Qtop.to_quaternion().slerp(Rh.to_quaternion(), f).to_matrix()
            self._put(n, p, Qs[n])
            prev = n
        for s, hd in fr["hands"].items():
            self._arm(s, hd, top, Qtop)
        for s, fd in fr["feet"].items():
            rigid(f"pelvis.{s}", C, hc)
            self._leg(s, fd)

    def _arm(self, s, hd, top, Qtop):
        n = lambda k: f"{k}.{s}"
        # clavicle + shoulder01 swing (and stretch a little) to put the shoulder joint on the solved one
        pc = top + Qtop @ (self.h0[n("clavicle")] - self.h0["spine01"])
        S, E, Wr = Vector(hd["shoulder"]), Vector(hd["elbow"]), Vector(hd["wrist"])
        ps0 = pc + Qtop @ (self.h0[n("upperarm01")] - self.h0[n("clavicle")])
        Qc = _minrot(ps0 - pc, S - pc) @ Qtop
        k = (S - pc).length / max((ps0 - pc).length, 1e-6)
        self._put(n("clavicle"), pc, Qc, (k, k, k))
        self._put(n("shoulder01"), pc + k * (Qc @ (self.h0[n("shoulder01")] - self.h0[n("clavicle")])), Qc, (k, k, k))
        # upper arm: aimed at the elbow, rolled so the rest elbow hinge lines up with the solved one
        d0 = self.h0[n("lowerarm01")] - self.h0[n("upperarm01")]
        f0 = self.h0[n("wrist")] - self.h0[n("lowerarm01")]
        d, f = E - S, Wr - E
        n0 = d0.cross(f0).normalized()
        nn = self._bend_axis("arm" + s, d, f, Qc @ n0)
        Qu = _frame(d, nn) @ _frame(d0, n0).inverted()
        Qsw = _minrot(Qc @ d0, d) @ Qc
        tw = _signed(Qsw @ n0, Qu @ n0, d.normalized())
        ku = d.length / d0.length
        self._put(n("upperarm01"), S, Matrix.Rotation(0.5 * tw, 3, d.normalized()) @ Qsw, (1.0, ku, 1.0))
        self._put(n("upperarm02"), S + ku * (Qu @ (self.h0[n("upperarm02")] - self.h0[n("upperarm01")])), Qu,
                  (1.0, ku, 1.0))
        # forearm: hinged at the elbow, then pronated part of the way toward the hand's own roll
        Qf = _minrot(Qu @ f0, f) @ Qu
        root = Matrix([list(r) for r in hd["root"]["rot"]])
        Qh = root @ self.R0[n("wrist")].inverted()
        u = f.normalized()
        ps = _signed(Qf @ n0, Qh @ n0, u)
        kf = f.length / f0.length
        self._put(n("lowerarm01"), E, Matrix.Rotation(0.2 * ps, 3, u) @ Qf, (1.0, kf, 1.0))
        self._put(n("lowerarm02"), E + kf * (Qf @ (self.h0[n("lowerarm02")] - self.h0[n("lowerarm01")])),
                  Matrix.Rotation(0.6 * ps, 3, u) @ Qf, (1.0, kf, 1.0))
        # the hand: the solved wrist pose, fingers by their joint angles (as pose_hand)
        sk = self.sk[s]
        self._put(n("wrist"), Vector(hd["root"]["pos"]), Qh, (sk.scale,) * 3)
        for nm in sk.names:
            if nm != "wrist":
                self.arm.pose.bones[n(nm)].matrix_basis = Matrix(
                    [list(r) for r in sk.local_rot(nm, hd["angles"].get(nm, {}))]).to_4x4()

    def _leg(self, s, fd):
        n = lambda k: f"{k}.{s}"
        H, K, A = Vector(fd["hip"]), Vector(fd["knee"]), Vector(fd["ankle"])
        d0 = self.h0[n("lowerleg01")] - self.h0[n("upperleg01")]
        s0 = self.h0[n("foot")] - self.h0[n("lowerleg01")]
        d, sh = K - H, A - K
        n0 = (self.A.to_3x3() @ Vector((1, 0, 0)))            # the knee's hinge at rest: the body's own left
        n0 = (n0 - d0.normalized() * n0.dot(d0.normalized())).normalized()
        nn = self._bend_axis("leg" + s, d, sh, self.C @ n0)
        Qt = _frame(d, nn) @ _frame(d0, n0).inverted()
        Qsw = _minrot(self.C @ d0, d) @ self.C
        tw = _signed(Qsw @ n0, Qt @ n0, d.normalized())
        kt = d.length / d0.length
        self._put(n("upperleg01"), H, Matrix.Rotation(0.5 * tw, 3, d.normalized()) @ Qsw, (1.0, kt, 1.0))
        self._put(n("upperleg02"), H + kt * (Qt @ (self.h0[n("upperleg02")] - self.h0[n("upperleg01")])), Qt,
                  (1.0, kt, 1.0))
        Qs = _minrot(Qt @ s0, sh) @ Qt
        ks = sh.length / s0.length
        self._put(n("lowerleg01"), K, Qs, (1.0, ks, 1.0))
        self._put(n("lowerleg02"), K + ks * (Qs @ (self.h0[n("lowerleg02")] - self.h0[n("lowerleg01")])), Qs,
                  (1.0, ks, 1.0))
        # foot: from the ankle, its sole turned onto the solved heel -> ball line, sized to it; toes bend at the ball
        heel0, ball0, toe0 = self.foot0[s]
        heel, ball, toe = Vector(fd["heel"]), Vector(fd["ball"]), Vector(fd["toe"])
        up = Vector((0, 0, 1))
        Qft = _frame(ball - heel, up) @ _frame(ball0 - heel0, up).inverted()
        kft = (ball - heel).length / (ball0 - heel0).length
        Pf = self._put(n("foot"), A, Qft, (kft,) * 3)
        Qtoe = _minrot(Qft @ (toe0 - ball0), toe - ball) @ Qft
        for t in self.TOES:
            tn = n(t)
            pt = A + kft * (Qft @ (self.h0[tn] - self.h0[n("foot")]))
            Pt = Matrix.Translation(pt) @ (Qtoe @ self.R0[tn]).to_4x4() @ Matrix.Diagonal((kft, kft, kft, 1.0))
            rel = self.Lc[n("foot")].inverted() @ self.Lc[tn]
            self.arm.pose.bones[tn].matrix_basis = rel.inverted() @ Pf.inverted() @ Pt


BODY = HANDS and "shoulders" in frames[0]["body"] and os.environ.get("DRUM_BODY", "mpfb") == "mpfb" \
    and os.path.exists(os.path.join(HAND_DIR, "mpfb_body.blend"))
FULL_FOREARM = False
if BODY:
    MPFB_BODY = MPFBBody(frames[0])
elif HANDS:
    MPFB_ARM, MPFB_BONES, FULL_FOREARM, (SKIN_CUT, SKIN_CUT_R, SKIN_END_R) = load_mpfb()
rig = {h: {"stick": Seg(f"{h}_stick", 0.0075, 0.0055, M["stick"])} for h in ("L", "R")}
for h in ("L", "R") if not BODY else ():
    # a skinned forearm that runs to the elbow is cut off just short of it: a sleeve a little wider than the cut
    # covers the elbow end and the cut edge, its tapered end tucked into the skin so no open edge shows
    fore = Seg(f"{h}_fore", SKIN_CUT_R + 0.003, max(SKIN_END_R - 0.002, 0.008)) if FULL_FOREARM \
        else Seg(f"{h}_fore", 0.036, 0.028)
    rig[h].update({"upper": Seg(f"{h}_upper", 0.046, 0.038), "fore": fore,
                   "thigh": Seg(f"{h}_thigh", 0.075, 0.058), "shin": Seg(f"{h}_shin", 0.055, 0.042),
                   "foot": hull_ob(f"{h}_foot")})
    if not HANDS:
        rig[h]["fist"] = sphere(f"{h}_fist", 1.0, M["skin"])
if not BODY:
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
        rig[h]["stick"].set(hd["butt"], hd["tip"])
    if BODY:
        MPFB_BODY.pose(fr)
        scn.render.filepath = os.path.join(out_dir, f"f_{i:05d}.png")
        bpy.ops.render.render(write_still=True)
        continue
    for h, hd in fr["hands"].items():
        R = rig[h]
        R["upper"].set(hd["shoulder"], hd["elbow"])
        R["fore"].set(hd["elbow"], hd["wrist"])
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

"""Blender (bpy) renderer for a piano blockout JSON from blockout/piano.py. Grey matte primitives, Workbench.

    blender -b --factory-startup -P blockout/blender_piano.py -- ANIM.json OUT_DIR [--res 1280x704]
           [--frames a:b] [--engine WORKBENCH|CYCLES] [--view pov|three4|side|handcam [--hand R|L --angle front3q|side|top|front --dist m]] [--lens mm]

--hands mpfb|skin|capsule: mpfb (default) is a real rigged, skinned CC0 hand from MPFB2, driven bone by
bone; skin is a Skin-modifier tube stopgap over the joint graph; capsule is the original mannequin.

--mark sides: the side and front faces of each pressed key light up in its finger's colour (thumb red .. pinky blue),
brighter the deeper the key is down. A check/scoring pass; keep H3 plates unmarked unless testing it.

pov rides the performer's eyes (head hidden); three4 is a concert three-quarter from the treble side;
side is a debug profile view for checking strikes and head nods; edit follows fr["cam"] from cameras.py
(a cut sequence: wide, hand tracking, close-ups, overhead, hero angles, first person).

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
if opts.get("--pass") == "chrome":     # target-material plate: Cycles, chrome hands, black gloss piano
    ENGINE = "CYCLES"
MARK = opts.get("--mark", "none")
HANDS = opts.get("--hands", "mpfb")   # mpfb: the rigged MPFB2 hand; skin: Skin-modifier tubes; capsule: mannequin
anim = json.load(open(anim_path))
frames = anim["frames"]
_fr = opts.get("--frames", f"0:{len(frames)}")
if "," in _fr or ":" not in _fr:                                   # an explicit list, e.g. one frame per shot for a contact sheet
    FRAME_IDS = [int(x) for x in _fr.split(",")]
else:
    fa, fb = map(int, _fr.split(":"))
    FRAME_IDS = list(range(fa, min(fb, len(frames))))
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
    if MARK == "sides":
        base = ob.data.materials[0]
        ob.data.materials.append(base)
        for poly in ob.data.polygons:
            if abs(poly.normal.x) > 0.9 or poly.normal.y < -0.9:     # sides and the front face
                poly.material_index = 1
        ob.material_slots[1].link = "OBJECT"
        ob.material_slots[1].material = base
        ob["base"] = base.name

box("keybed", X0 - 0.04, X1 + 0.04, -0.02, 0.30, -0.10, -0.019, M_CASE)
box("front_rail", X0 - 0.04, X1 + 0.04, -0.03, -0.004, -0.10, -0.012, M_CASE)
box("cheek_l", X0 - 0.06, X0 - 0.005, -0.03, 0.30, -0.10, 0.045, M_CASE)
box("cheek_r", X1 + 0.005, X1 + 0.06, -0.03, 0.30, -0.10, 0.045, M_CASE)
box("fallboard", X0 - 0.005, X1 + 0.005, 0.153, 0.20, -0.02, 0.055, M_CASE)
box("lid_body", X0 - 0.06, X1 + 0.06, 0.20, 0.32, -0.10, 0.07, M_CASE)
VIEW = opts.get("--view", "pov")
HC_HAND, HC_ANGLE, HC_DIST = opts.get("--hand", "R"), opts.get("--angle", "front3q"), float(opts.get("--dist", 0.34))
if VIEW == "pov":
    box("music_desk", X0 + 0.45, X1 - 0.45, 0.22, 0.25, 0.07, 0.30, M_CASE)
box("floor", X0 - 3, X1 + 3, -3, 4, -0.80, -0.79, M_FLOOR)

# ------------------------------------------------------------------ hands
FINGER_HUE = [(1.0, 0.15, 0.10), (1.0, 0.60, 0.05), (0.15, 0.90, 0.20), (0.05, 0.75, 1.0), (0.35, 0.25, 1.0)]
MARKS = {}


def mark_mat(f, level):
    """Bright finger colour for a pressed key's sides (level 0..4 = how far down)."""
    key = (f, level)
    if key not in MARKS:
        c = FINGER_HUE[f]
        g = 0.35 + 0.65 * (level + 1) / 5
        m = bpy.data.materials.new(f"mark{f}_{level}")
        m.diffuse_color = (c[0] * g, c[1] * g, c[2] * g, 1.0)
        MARKS[key] = m
    return MARKS[key]


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


FR = anim.get("finger_radii") or [(0.0115, 0.0105, 0.0095, 0.0085), (0.0095, 0.0088, 0.0080, 0.0070),
                                  (0.0098, 0.0090, 0.0082, 0.0072), (0.0093, 0.0085, 0.0078, 0.0068),
                                  (0.0085, 0.0078, 0.0070, 0.0062)]
M_NAIL = mat("nail", 0.62)


def skin_hand(h):
    """Fingers as one continuous skinned mesh: wrist -> each knuckle (metacarpals, inside the palm) -> PIP ->
    DIP -> tip, every finger a smooth tapered tube with natural joint swellings (Skin + Subdivision)."""
    me = bpy.data.meshes.new(f"{h}_hand")
    verts = [(0, 0, 0)] + [(0, 0, 0)] * 20
    edges = []
    for f in range(5):
        b = 1 + 4 * f
        edges += [(0, b), (b, b + 1), (b + 1, b + 2), (b + 2, b + 3)]
    me.from_pydata(verts, edges, [])
    ob = bpy.data.objects.new(f"{h}_hand", me)
    ob.data.materials.append(M_SKIN)
    COL.objects.link(ob)
    sk = ob.modifiers.new("skin", "SKIN")
    sk.use_smooth_shade = True
    sk.branch_smoothing = 0.6
    sub = ob.modifiers.new("smooth", "SUBSURF")
    sub.levels, sub.render_levels = 2, 2
    sv = me.skin_vertices[0].data
    sv[0].use_root = True
    sv[0].radius = (0.022, 0.016)
    for f in range(5):
        for k in range(4):
            r = FR[f][k] * (1.05 if k == 0 else 1.0)
            sv[1 + 4 * f + k].radius = (r * 1.06, r * 0.86)        # fingers are a little flatter than round
    return ob


def nail(name):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=1.0)
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(M_NAIL)
    COL.objects.link(ob)
    return ob


def set_skin_hand(R, wrist, fingers):
    me = R["skin"].data
    me.vertices[0].co = Vector(wrist)
    for f, chain in enumerate(fingers):
        for k in range(4):
            me.vertices[1 + 4 * f + k].co = Vector(chain[k])
        tip, dip = Vector(chain[3]), Vector(chain[2])
        along = (tip - dip).normalized()
        side = along.cross(Vector((0, 0, 1)))
        if side.length < 1e-4:
            side = Vector((1, 0, 0))
        side.normalize()
        up = side.cross(along).normalized()
        if up.z < 0:
            up = -up
        r = FR[f][3]
        c = tip + up * (r * 0.62) - along * (r * 0.35)               # on the back of the fingertip
        rot = Matrix((side, along, up)).transposed().to_4x4()
        R["nails"][f].matrix_world = Matrix.Translation(c) @ rot @ Matrix.Diagonal((r * 0.75, r * 0.95, r * 0.22, 1))
    me.update()


# ------------------------------------------------------------------ MPFB2 hand (--hands mpfb)
# A CC0 MakeHuman hand from MPFB2 (blockout/hand_model/mpfb_make_hand.py): skinned mesh + MPFB's default rig.
# Every frame each bone is posed from our joints: wrist -> metacarpal -> MCP -> PIP -> DIP -> tip for the fingers,
# CMC -> MCP -> IP -> tip for the thumb. piano.py's kinematics use this rig's own bone lengths and knuckle layout
# (piano.set_hand_model("mpfb")), so the bones land on the joints; the residual is reported at the end.
HAND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_model")
FINGER_BONES = [[f"finger{f}-{s}" for s in (1, 2, 3)] for f in range(1, 6)]     # thumb .. pinky
MPFB_RESID = []


def mpfb_frame(wrist, mcps, side):
    """Same frame as piano.mpfb_frame: centre of the four MCPs; across, forward, up (back of hand)."""
    c = sum(mcps, Vector()) / 4
    fwd = (c - wrist).normalized()
    ac = mcps[3] - mcps[0]
    ac = (ac - fwd * ac.dot(fwd)).normalized()
    up = ac.cross(fwd) * (1 if side == "R" else -1)
    return c, Matrix((ac, fwd, up))                    # rows: world -> hand-local


def load_mpfb():
    with bpy.data.libraries.load(os.path.join(HAND_DIR, "mpfb_hands.blend")) as (src, dst):
        dst.objects = ["hands", "Human.rig"]
    mesh, arm = dst.objects
    for o in (mesh, arm):
        COL.objects.link(o)
    mesh.data.materials.clear()
    mesh.data.materials.append(M_SKIN)
    for p in mesh.data.polygons:
        p.use_smooth = True
    sub = mesh.modifiers.new("smooth", "SUBSURF")      # after the armature: smooth the posed mesh
    sub.levels, sub.render_levels = 1, 2
    arm.hide_render = True
    global MPFB_MESH
    MPFB_MESH = mesh
    rest = {}
    for s in "LR":
        B = {b.name[:-2]: b for b in arm.data.bones if b.name.endswith("." + s)}
        c, M = mpfb_frame(B["wrist"].head_local, [B[f"finger{f}-1"].head_local for f in range(2, 6)], s)
        rest[s] = {"c": c, "M": M, "bones": B}
    return arm, rest


_SK = {}


def pose_rig(arm, rest, s, hd):
    """Pure forward kinematics from the hand rig (blockout/rig): the wrist bone gets the solved root pose and every
    other bone its joint rotation about its own rest axes -- exactly the pose the rig solved, nothing re-aimed."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from blockout.rig.skeleton import Skeleton
    if s not in _SK:
        _SK[s] = Skeleton(s)
    sk = _SK[s]
    B = rest[s]["bones"]
    wrist, elbow = Vector(hd["wrist"]), Vector(hd["elbow"])
    u = (wrist - elbow).normalized()
    want = {}
    root = Matrix([list(r) for r in hd["root"]["rot"]]).to_4x4()
    root.translation = Vector(hd["root"]["pos"])
    for nm, back, share in (("lowerarm02", 0.0, 0.6), ("lowerarm01", B["lowerarm02"].length, 0.2)):
        head = wrist - u * (B["lowerarm02"].length + back)
        r3 = B[nm].matrix_local.to_3x3()
        r3 = r3.col[1].normalized().rotation_difference(u).to_matrix() @ r3
        # forearm pronation: the two forearm bones take a share of the hand's own roll about the forearm axis
        rigid = root.to_3x3() @ B["wrist"].matrix_local.to_3x3().inverted() @ B[nm].matrix_local.to_3x3()
        za, zb = r3.col[2] - u * r3.col[2].dot(u), rigid.col[2] - u * rigid.col[2].dot(u)
        if za.length > 1e-6 and zb.length > 1e-6:
            ang = za.normalized().angle(zb.normalized())
            if za.normalized().cross(zb.normalized()).dot(u) < 0:
                ang = -ang
            r3 = Matrix.Rotation(ang * share, 3, u) @ r3
        m = r3.to_4x4(); m.translation = head
        want[nm] = m
    want["wrist"] = root @ Matrix.Diagonal((sk.scale, sk.scale, sk.scale, 1.0))   # a pianist-sized hand
    ang = hd["angles"]
    for nm in sk.names:
        pb = arm.pose.bones[f"{nm}.{s}"]
        b = B[nm]
        if nm in want:
            m = want[nm]
            if b.parent is None:
                pb.matrix_basis = b.matrix_local.inverted() @ m
            else:
                pn = b.parent.name[:-2]
                pb.matrix_basis = ((b.parent.matrix_local.inverted() @ b.matrix_local).inverted()
                                   @ want[pn].inverted() @ m)
        else:
            R = sk.local_rot(nm, ang.get(nm, {}))
            pb.matrix_basis = Matrix([list(r) for r in R]).to_4x4()
    for nm in ("lowerarm01", "lowerarm02"):
        b = B[nm]; pb = arm.pose.bones[f"{nm}.{s}"]
        if b.parent is None:
            pb.matrix_basis = b.matrix_local.inverted() @ want[nm]
        else:
            pn = b.parent.name[:-2]
            pb.matrix_basis = ((b.parent.matrix_local.inverted() @ b.matrix_local).inverted() @ want[pn].inverted() @ want[nm])


def pose_mpfb(arm, rest, s, hd):
    """Pose one hand's bones from the frame's joints (armature space = world: the rig sits at the origin)."""
    R = rest[s]
    B = R["bones"]
    wrist, elbow = Vector(hd["wrist"]), Vector(hd["elbow"])
    ch = [[Vector(q) for q in c] for c in hd["fingers"]]
    c, M = mpfb_frame(wrist, [ch[f][0] for f in range(1, 5)], s)
    rot = M.transposed() @ R["M"]                       # rest hand frame -> this frame's hand frame

    want = {}

    def aim(name, head, target):
        """Carry the bone with its parent (or the rigid hand), then swing it onto its segment: no twist is
        added, so a hinge chain posed in its own plane turns exactly about its hinge axis."""
        b = B[name]
        pn = b.parent.name[:-2] if b.parent else None
        if pn in want and pn not in ("lowerarm01", "lowerarm02"):
            r3 = want[pn].to_3x3() @ b.parent.matrix_local.to_3x3().inverted() @ b.matrix_local.to_3x3()
        else:
            r3 = rot @ b.matrix_local.to_3x3()
        d0 = r3.col[1].normalized()
        if target is not None:
            r3 = d0.rotation_difference((target - head).normalized()).to_matrix() @ r3
        m = r3.to_4x4()
        m.translation = head
        return m

    u = (wrist - elbow).normalized()
    want["lowerarm02"] = aim("lowerarm02", wrist - u * B["lowerarm02"].length, wrist)
    want["lowerarm01"] = aim("lowerarm01", wrist - u * (B["lowerarm02"].length + B["lowerarm01"].length),
                             wrist - u * B["lowerarm02"].length)
    want["wrist"] = aim("wrist", wrist, None)
    for m in range(1, 5):
        head = c + rot @ (B[f"metacarpal{m}"].head_local - R["c"])
        want[f"metacarpal{m}"] = aim(f"metacarpal{m}", head, ch[m][0])
    for f in range(5):
        for k, name in enumerate(FINGER_BONES[f]):
            want[name] = aim(name, ch[f][k], ch[f][k + 1])
            tail = want[name] @ Vector((0, B[name].length, 0))
            MPFB_RESID.append((tail - ch[f][k + 1]).length)
    for name, m in want.items():                        # pose matrix -> local basis, through the parent
        b = B[name]
        pb = arm.pose.bones[f"{name}.{s}"]
        if b.parent is None:
            pb.matrix_basis = b.matrix_local.inverted() @ m
        else:
            pn = b.parent.name[:-2]
            pb.matrix_basis = ((b.parent.matrix_local.inverted() @ b.matrix_local).inverted()
                               @ want[pn].inverted() @ m)


if HANDS == "mpfb":
    MPFB_ARM, MPFB_REST = load_mpfb()
    _ys = [MPFB_MESH.matrix_world @ v.co for v in MPFB_MESH.data.vertices]
    _w = MPFB_ARM.data.bones["wrist.R"].head_local; _e = MPFB_ARM.data.bones["lowerarm01.R"].head_local
    MPFB_FULL_FOREARM = max(((p - _w).dot((_e - _w).normalized()) for p in _ys)) > 0.2
    if "--gauntlet" in sys.argv:        # gauntlet geometry on the hand: a ring at every finger joint and a cuff at the
        def _gaunt():                   # wrist, parented to the bones (so they follow the solved pose exactly); the
            vg = {g.index: g.name for g in MPFB_MESH.vertex_groups}      # normal/canny control then shows plated
            pts = {}                                                    # fingers, not a bare hand
            for v in MPFB_MESH.data.vertices:
                for g in v.groups:
                    if g.weight > 0.6:
                        pts.setdefault(vg[g.group], []).append(MPFB_MESH.matrix_world @ v.co)
            for s_ in "LR":
                names = [f"finger{f}-{k}" for f in range(1, 6) for k in (1, 2, 3)] + ["wrist"]
                for nm in names:
                    b = MPFB_ARM.data.bones.get(f"{nm}.{s_}")
                    if b is None or f"{nm}.{s_}" not in pts:
                        continue
                    a, d = b.head_local, (b.tail_local - b.head_local).normalized()
                    rr = sorted(((p - a) - d * (p - a).dot(d)).length for p in pts[f"{nm}.{s_}"])
                    r = rr[len(rr) // 2] * (1.25 if nm == "wrist" else 1.12)
                    depth = (0.022 if nm == "wrist" else min(0.006, 0.3 * b.length))
                    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, vertices=28)
                    o = bpy.context.active_object; o.name = f"ring_{nm}_{s_}"
                    o.data.materials.append(M_SKIN)
                    for pl in o.data.polygons:
                        pl.use_smooth = True
                    o.parent, o.parent_type, o.parent_bone = MPFB_ARM, "BONE", f"{nm}.{s_}"
                    o.matrix_parent_inverse = Matrix.Identity(4)
                    o.location = (0, -b.length + (0.012 if nm == "wrist" else 0.002), 0)   # at the joint (bone head)
                    o.rotation_euler = (-math.pi / 2, 0, 0)                               # cylinder axis -> bone
        _gaunt()

rig = {}
for h in ("L", "R"):
    if HANDS == "mpfb":     # the primitive forearm stops short and swallows the cut edge of the hand's forearm stub
        rig[h] = {"forearm": Seg(f"{h}_forearm", 0.036, 0.031), "upper": Seg(f"{h}_upper", 0.046, 0.038)}
        continue
    rig[h] = {"forearm": Seg(f"{h}_forearm", 0.036, 0.026), "upper": Seg(f"{h}_upper", 0.046, 0.038)}
    if HANDS == "capsule":
        rig[h]["fingers"] = [[Seg(f"{h}_f{f}_{s}", FR[f][s], FR[f][s + 1]) for s in range(3)] for f in range(5)]
    else:
        rig[h]["skin"] = skin_hand(h)
        rig[h]["nails"] = [nail(f"{h}_nail{f}") for f in range(5)]
    me = bpy.data.meshes.new(f"{h}_palm")
    ob = bpy.data.objects.new(f"{h}_palm", me)
    ob.data.materials.append(M_SKIN)
    COL.objects.link(ob)
    sub = ob.modifiers.new("round", "SUBSURF")          # a hull of points, rounded into a hand
    sub.levels, sub.render_levels = 2, 2
    rig[h]["palm"] = ob
    rig[h]["wrist"] = sphere(f"{h}_wristball", 0.030, M_SKIN)


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


def set_palm(ob, pts, thick):
    """Convex hull of the palm landmarks, each given a top and bottom (thick = [(up, down), ...])."""
    bm = bmesh.new()
    for p, (tu, td) in zip(pts, thick):
        bm.verts.new(Vector(p) + Vector((0, 0, tu)))
        bm.verts.new(Vector(p) - Vector((0, 0, td)))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(ob.data); bm.free()


# ------------------------------------------------------------------ camera, light, render
cx = anim["centre_x"]
cam_d = bpy.data.cameras.new("cam")
cam_d.lens, cam_d.sensor_width = float(opts.get("--lens", 30)), 36.0
cam_d.clip_start = 0.01
cam = bpy.data.objects.new("cam", cam_d)
COL.objects.link(cam)
FIXED = {"three4": ((cx + 0.95, 0.55, 0.80), (cx - 0.05, -0.30, 0.22)),
         "overhead": ((cx, -0.30, 1.05), (cx, 0.02, 0.0)),
         "side": ((cx + 1.30, -0.30, 0.30), (cx, -0.30, 0.22))}
if "--eye" in opts:                                   # debug close-ups: --eye x,y,z --look x,y,z
    FIXED[VIEW] = (tuple(map(float, opts["--eye"].split(","))), tuple(map(float, opts["--look"].split(","))))
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

PASS = opts.get("--pass")           # depth: an inverse-depth control image (near = white) for H3 Fun ControlNet
                                    # mask: the hands (and forearms) white, everything else black, flat (for metrics)
if PASS == "depth":
    near, far = (float(v) for v in opts.get("--depth-range", "0.12,2.5").split(","))
    scn.view_settings.view_transform = "Standard"
    bpy.context.view_layer.use_pass_z = True
    scn.use_nodes = True
    nt = scn.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    rl = nt.nodes.new("CompositorNodeRLayers")
    inv = nt.nodes.new("CompositorNodeMath"); inv.operation = "DIVIDE"; inv.inputs[0].default_value = 1.0
    mr = nt.nodes.new("CompositorNodeMapRange"); mr.use_clamp = True
    mr.inputs[1].default_value, mr.inputs[2].default_value = 1.0 / far, 1.0 / near
    mr.inputs[3].default_value, mr.inputs[4].default_value = 0.0, 1.0
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Depth"], inv.inputs[1]); nt.links.new(inv.outputs[0], mr.inputs[0])
    nt.links.new(mr.outputs[0], comp.inputs["Image"])

if PASS == "normal":                  # shaded by surface normal: every finger, knuckle and key edge a distinct colour
    scn.display.shading.light = "MATCAP"
    scn.display.shading.studio_light = "check_normal+y.exr"
    scn.display.shading.color_type = "SINGLE"
    scn.display.shading.show_shadows = False
    scn.display.shading.show_cavity = False
    scn.view_settings.view_transform = "Standard"
if PASS == "chrome":                  # the plate in the look H3 should render (mirror chrome gauntlets, glossy
    def _pbr(m, base, metal, rough):    # black grand, warm low light): plate pixels then carry the material
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = (*base, 1.0)
        b.inputs["Metallic"].default_value = metal
        b.inputs["Roughness"].default_value = rough
    _pbr(M_SKIN, (0.86, 0.86, 0.9), 1.0, 0.12)
    _pbr(M_NAIL, (0.86, 0.86, 0.9), 1.0, 0.12)
    _pbr(M_CASE, (0.004, 0.004, 0.004), 0.0, 0.18)
    _pbr(M_BLACK, (0.006, 0.006, 0.006), 0.0, 0.1)
    _pbr(M_WHITE, (0.78, 0.75, 0.68), 0.0, 0.25)
    world.use_nodes = True
    _pbr(M_FLOOR, (0.012, 0.007, 0.004), 0.0, 0.5)            # dark wood, a dark smoky room
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.028, 0.014, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = float(opts.get("--world", 0.12))
    scn.cycles.samples = int(opts.get("--samples", 64))
    scn.cycles.use_denoising = False                # this Blender build has no OpenImageDenoise
    for nm, loc, e, col in (("lamp", (cx + 0.6, 0.5, 1.1), 60, (1.0, 0.62, 0.3)),
                            ("rim", (cx - 0.9, 0.9, 0.7), 30, (1.0, 0.8, 0.6))):
        L = bpy.data.objects.new(nm, bpy.data.lights.new(nm, "AREA"))
        L.data.energy, L.data.size, L.data.color = e, 0.8, col
        L.location = loc
        L.rotation_euler = (Vector((cx, 0.1, 0)) - L.location).to_track_quat("-Z", "Y").to_euler()
        COL.objects.link(L)
    for nm, loc, sz, st in (("softbox", (cx, -0.2, 1.6), (0.7, 0.25), 2.0), ("bounce", (cx, 1.2, 0.6), (1.2, 0.3), 0.5)):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc)
        pl = bpy.context.active_object; pl.scale = (*sz, 1)
        pl.rotation_euler = (Vector((cx, 0.1, 0.1)) - pl.location).to_track_quat("Z", "Y").to_euler()
        em = bpy.data.materials.new(nm); em.use_nodes = True
        nt = em.node_tree; nt.nodes.clear()
        es = nt.nodes.new("ShaderNodeEmission"); es.inputs[0].default_value = (1.0, 0.75, 0.5, 1); es.inputs[1].default_value = st
        lp = nt.nodes.new("ShaderNodeLightPath"); mx = nt.nodes.new("ShaderNodeMixShader"); tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0]); nt.links.new(es.outputs[0], mx.inputs[1])
        nt.links.new(tr.outputs[0], mx.inputs[2]); nt.links.new(mx.outputs[0], out.inputs[0])
        pl.data.materials.append(em)
if PASS == "mask":
    scn.display.shading.light = "FLAT"
    scn.display.shading.color_type = "OBJECT"
    scn.display.shading.show_shadows = False
    scn.display.shading.show_cavity = False
    scn.display.render_aa = "OFF"
    scn.view_settings.view_transform = "Standard"
    world.color = (0, 0, 0)
    for ob in COL.objects:
        ob.color = (0, 0, 0, 1)
    if globals().get("MPFB_MESH") is not None:
        MPFB_MESH.color = (1, 1, 1, 1)

MESHQA = opts.get("--meshqa")      # OUT.json: no render; count skinned-mesh self-intersections per frame, by body part
if MESHQA:
    import numpy as np
    from mathutils.bvhtree import BVHTree
    me0 = MPFB_MESH.data
    MPFB_MESH.modifiers["smooth"].show_viewport = False
    me0.calc_loop_triangles()
    TRIS = [tuple(t.vertices) for t in me0.loop_triangles]
    gname = {g.index: g.name for g in MPFB_MESH.vertex_groups}

    def part(vn):
        b, sd = vn.rsplit(".", 1)
        if b.startswith("finger"):
            return f"{sd}{b[6]}"                          # R1 = right thumb ... R5 = right pinky
        return f"{sd}{'arm' if b.startswith('lowerarm') else 'palm'}"
    VPART = []
    for v in me0.vertices:
        g = max(v.groups, key=lambda e: e.weight, default=None)
        VPART.append(part(gname[g.group]) if g else "?")
    TPART = [VPART[t[0]] for t in TRIS]
    TSET = [set(t) for t in TRIS]
    NV = len(me0.vertices)

    KB = np.array([(k["x"] - k["w"] / 2, k["x"] + k["w"] / 2, k["y0"], 0.150, k["top"] - (0.012 if k["black"] else 0.018),
                    k["top"], k["pitch"]) for k in anim["keyboard"]])
    LAST_CO = {}
    PARTS_HIT = {}

    def key_penetration(co, keys):
        """Skinned-mesh vertices inside a key box (keys tilted down by their depth): count > 1 mm and max depth."""
        depth = np.zeros(len(KB))
        for i, k in enumerate(KB):
            depth[i] = float(keys.get(str(int(k[6])), 0.0))
        worst, n = 0.0, 0
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        near = (z < 0.013) & (y > -0.01) & (y < 0.155)
        for i, (x0, x1, y0, y1, zb, zt, _p) in enumerate(KB):
            m = near & (x > x0) & (x < x1) & (y > y0) & (y < y1)
            if not m.any():
                continue
            top = zt - depth[i] * 0.010 * (0.150 - y[m]) / (0.150 - y0)     # pivot at the back
            d = np.minimum(top - z[m], z[m] - zb)
            idx = np.nonzero(m)[0][d > 0.001]
            for vi in idx:
                PARTS_HIT[VPART[vi]] = PARTS_HIT.get(VPART[vi], 0) + 1
            if os.environ.get("KEYPEN_DUMP") and len(idx):
                dd = d[d > 0.001]
                print("KP key", int(_p), "depth", round(float(depth[i]), 2), "n", len(idx), "max mm", round(float(dd.max()) * 1000, 1),
                      "parts", sorted({VPART[v] for v in idx}), "y", round(float(y[idx].min()), 3), round(float(y[idx].max()), 3),
                      "z", round(float(z[idx].min()), 4), "x", round(float(x[idx].min()), 3), round(float(x[idx].max()), 3), "key x", round(x0, 3), round(x1, 3))
            d = d[d > 0]
            if len(d):
                worst = max(worst, float(d.max())); n += int((d > 0.001).sum())
        return n, worst

    def intersections():
        dg = bpy.context.evaluated_depsgraph_get()
        ev = MPFB_MESH.evaluated_get(dg)
        m = ev.to_mesh()
        co = np.empty(NV * 3); m.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        LAST_CO["co"] = co.reshape(-1, 3)
        tree = BVHTree.FromPolygons([tuple(c) for c in co.reshape(-1, 3)], TRIS)
        out = set()
        for a, b in tree.overlap(tree):
            if a < b and not (TSET[a] & TSET[b]):
                out.add((a, b))
        return out
    for pb in MPFB_ARM.pose.bones:
        pb.matrix_basis = Matrix()
    bpy.context.view_layer.update()
    REST_PAIRS = intersections()
    MQ = {"rest_pairs": len(REST_PAIRS), "frames": {}}

for i in FRAME_IDS:
    fr = frames[i]
    for p, ob in keys.items():
        d = fr["keys"].get(p, 0.0)
        ob.rotation_euler = ((d * 0.010) / ob["length"], 0, 0)          # pivot at the back: + tilts the front down
        if MARK == "sides":
            who = fr.get("press", {}).get(p)
            ob.material_slots[1].material = (mark_mat(int(who[1]) - 1, min(4, int(d * 5))) if who and d > 0.05
                                             else bpy.data.materials[ob["base"]])
    for h, hd in fr["hands"].items():
        R = rig[h]
        R["forearm"].set(hd["elbow"], hd["wrist"])
        if HANDS == "mpfb":
            el, wr = Vector(hd["elbow"]), Vector(hd["wrist"])
            R["forearm"].set(el, wr - (wr - el).normalized() * 0.05)
            if "angles" in hd and MPFB_FULL_FOREARM:          # the skinned forearm runs (almost) to the elbow:
                R["forearm"].set(el, el + (wr - el).normalized() * 0.06)   # a short sleeve covers its cut edge
                R["forearm"].j1.hide_render = True
            (pose_rig if "angles" in hd else pose_mpfb)(MPFB_ARM, MPFB_REST, h, hd)
            if "angles" in hd and opts.get("--fkcheck"):
                bpy.context.view_layer.update()
                for f in range(5):
                    tl = MPFB_ARM.pose.bones[f"finger{f + 1}-3.{h}"].tail
                    MPFB_RESID.append((Vector(hd["fingers"][f][3]) - tl).length)
            R["upper"].set(fr["body"]["shoulders"][h], hd["elbow"])
            continue
        if HANDS == "capsule":
            for f, chain in enumerate(hd["fingers"]):
                for s in range(3):
                    R["fingers"][f][s].set(chain[s], chain[s + 1])
        else:
            set_skin_hand(R, hd["wrist"], hd["fingers"])
        w = Vector(hd["wrist"])
        side = Vector((0.033, 0, 0))
        knuckles = [hd["fingers"][f][0] for f in range(1, 5)]
        thenar = [hd["fingers"][0][0], hd["fingers"][0][1]]
        set_palm(R["palm"], knuckles + [w + side, w - side] + thenar,
                 [(0.011, 0.009)] * 4 + [(0.014, 0.012)] * 2 + [(0.012, 0.014), (0.010, 0.010)])
        R["wrist"].location = w
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
    elif VIEW == "handcam":                            # review: follow one hand from its own joints, fixed distance
        hd = fr["hands"][HC_HAND]
        pts = [Vector(hd["wrist"])] + [Vector(c[k]) for c in hd["fingers"] for k in (0, 3)]
        ctr = sum(pts, Vector()) / len(pts)
        sg = 1.0 if HC_HAND == "R" else -1.0
        dirv, upv = {"front3q": ((0.35 * sg, -0.75, 0.80), (0, 0, 1)), "side": ((sg, -0.05, 0.22), (0, 0, 1)),
                     "top": ((0, -0.08, 1.0), (0, 1, 0)), "front": ((0.20 * sg, 1.0, 0.45), (0, 0, 1))}[HC_ANGLE]
        eye = ctr + Vector(dirv).normalized() * HC_DIST
        cam.matrix_world = look_matrix(eye, ctr - eye, upv)
    elif VIEW == "edit":
        c = fr["cam"]
        fwd = Vector(c["look"]) - Vector(c["pos"])
        cam.matrix_world = look_matrix(c["pos"], fwd, c["up"])
        cam_d.lens = c["lens"]
        for ob in HEAD_PARTS:
            ob.hide_render = c["hide_head"]
        if "hide_body" in c:                           # keyboard shots: only the arms, hands and piano render
            for ob in COL.objects:
                if ob.name.split("_")[0].rstrip("-1") in ("torso", "head", "nose", "neck", "thigh", "shin", "bench"):
                    ob.hide_render = bool(c["hide_body"]) or (ob in HEAD_PARTS and c["hide_head"])
    if "case" in opts.get("--hide", ""):              # debug: see the fingers from the far side of the piano
        for ob in COL.objects:
            if ob.name in ("fallboard", "lid_body", "music_desk", "cheek_l", "cheek_r"):
                ob.hide_render = True
    if "body" in opts.get("--hide", ""):              # debug: see the hands from the player's side
        for ob in COL.objects:
            if ob.name.split("_")[0].rstrip("-1") in ("torso", "head", "nose", "neck", "thigh", "shin", "bench") \
                    or ob.name.startswith(("L_upper", "R_upper")):
                ob.hide_render = True
    if MESHQA:
        bpy.context.view_layer.update()
        cnt = {}
        for a, b in intersections() - REST_PAIRS:
            k = "-".join(sorted((TPART[a], TPART[b])))
            cnt[k] = cnt.get(k, 0) + 1
        kn, kw = key_penetration(LAST_CO["co"], fr["keys"])
        if kn:
            cnt["keys"] = kn
            MQ.setdefault("keys_worst_mm", {})[i] = round(kw * 1000, 1)
        if cnt:
            MQ["frames"][i] = cnt
        continue
    scn.render.filepath = os.path.join(out_dir, f"f_{i:05d}.png")
    bpy.ops.render.render(write_still=True)
if MESHQA:
    MQ["key_hits_by_part"] = PARTS_HIT
    json.dump(MQ, open(MESHQA, "w"))
    print("meshqa frames with intersections:", len(MQ["frames"]), "of", len(FRAME_IDS))
if MPFB_RESID:
    MPFB_RESID.sort()
    print(f"mpfb joint residual: median {MPFB_RESID[len(MPFB_RESID) // 2] * 1000:.2f} mm, "
          f"max {MPFB_RESID[-1] * 1000:.2f} mm over {len(MPFB_RESID)} bone tails")
print("done", len(FRAME_IDS), "frames")

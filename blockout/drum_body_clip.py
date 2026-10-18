"""Clip check for the drawn MPFB2 drummer, not the solver's capsules. Every Nth frame poses the body exactly as
blender_drums.py does, takes the deformed mesh cages (skin, clothes, shoes) and measures how deep any vertex sits
inside the kit's hitboxes (drum_collide.Kit) or either stick, and how far each forearm and hand goes into the head,
the torso, the thighs and the other arm. Allowed: the playing hand on a piece it is choking or pinching, and a hand
on its own stick.

    blender -b --factory-startup -P blockout/drum_body_clip.py -- ANIM.json OUT.json [--step 4] [--tol 0.004]
           [--only 10,11,12]

Writes a JSON report: frames checked, frames clipping, and the deepest frame of each body-part x obstacle pair.
"""
import json, os, runpy, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:]
anim_path, out_json = os.path.abspath(argv[0]), os.path.abspath(argv[1])
opt = dict(zip(argv[2::2], argv[3::2]))
STEP, TOL = int(opt.get("--step", 4)), float(opt.get("--tol", 0.004))
# blender_drums.py builds the scene; an empty frame range skips its render loop
sys.argv = [sys.argv[0], "--", anim_path, tempfile.mkdtemp(prefix="drum_clip_"), "--frames", "0:0"]
g = runpy.run_path(os.path.join(HERE, "blender_drums.py"))
import bpy
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(HERE))
from blockout import drum_collide as dc

body = g.get("MPFB_BODY")
if body is None:
    sys.exit("drum_body_clip: needs the MPFB2 body (blockout/hand_model/mpfb_body.blend and a take with shoulders)")
anim, frames = g["anim"], g["frames"]
kit = dc.Kit(anim["kit"], anim.get("hihat_open", 0.018))
allow = dc.allowed_windows(anim["hits"], anim.get("chokes", ()), anim["fps"])
PARTS = (("lowerarm", "fore"), ("upperarm", "upper"), ("shoulder", "upper"), ("clavicle", "torso"),
         ("wrist", "hand"), ("finger", "hand"), ("metacarpal", "hand"), ("upperleg", "thigh"), ("lowerleg", "shin"),
         ("foot", "foot"), ("toe", "foot"), ("head", "head"), ("neck", "head"), ("jaw", "head"), ("eye", "head"),
         ("tongue", "head"), ("spine", "torso"), ("pelvis", "torso"), ("root", "torso"), ("breast", "torso"))


def part_of(bone):
    side = "L" if bone.endswith(".L") else "R" if bone.endswith(".R") else "-"
    b = bone.lower()
    return side, next((part for key, part in PARTS if key in b), "head")


objs = [body.mesh] + [o for n, o in body.worn.items() if not n.startswith(("hair", "brow", "eyebrow"))]
flat, src, polys, off = [], [], [], 0
for o in objs:
    for m in o.modifiers:
        if m.type == "SUBSURF":
            m.show_viewport = False         # the cage: on or outside the smoothed surface
    names = {vg.index: vg.name for vg in o.vertex_groups}
    for v in o.data.vertices:
        gs = [(e.weight, names[e.group]) for e in v.groups if e.weight > 0 and names[e.group] != "eyes"]
        flat.append(part_of(max(gs)[1]) if gs else ("-", "torso"))
    src += [o.name.split(".")[0]] * len(o.data.vertices)
    polys += [[off + i for i in p.vertices] for p in o.data.polygons]
    off += len(o.data.vertices)
side = np.array([s for s, _ in flat])
part = np.array([p for _, p in flat])
src = np.array(src)


def group(p):
    keys = [(side[i], part[i]) for i in p]
    return max(set(keys), key=keys.count)


pg = [group(p) for p in polys]
SELF = {f"{h}.arm": [p for p, (s, q) in zip(polys, pg) if s == h and q in ("fore", "hand")] for h in "LR"}
for name, q0 in (("head", "head"), ("torso", "torso"), ("thighs", "thigh")):
    SELF[name] = [p for p, (s, q) in zip(polys, pg) if q == q0]
SELF_PAIRS = [(f"{h}.arm", t) for h in "LR" for t in ("head", "torso", "thighs")] + [("L.arm", "R.arm")]
SELF_VERTS = {k: {v for ps in pp for v in ps} for k, pp in SELF.items()}


def posed_points(dg):
    P = []
    for o in objs:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(e.matrix_world)
        P.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        e.to_mesh_clear()
    return np.concatenate(P)


def self_depth(Pv, trees, a, b):
    """Deepest vertex of either part behind the other's surface (by the nearest face's normal)."""
    d = 0.0
    for tree, verts in ((trees[b], SELF_VERTS[a]), (trees[a], SELF_VERTS[b])):
        for v in verts:
            hit = tree.find_nearest(Pv[v], 0.1)
            if hit[0] is not None and (np.array(Pv[v]) - np.array(hit[0])) @ np.array(hit[1]) < 0:
                d = max(d, hit[3])
    return d


dg = bpy.context.evaluated_depsgraph_get()
stats, worst = {}, []
todo = [int(x) for x in opt["--only"].split(",")] if "--only" in opt else range(0, len(frames), STEP)
for i in todo:
    fr = frames[i]
    body.pose(fr)
    dg.update()
    P = posed_points(dg)
    kit.pose(fr)
    hits = {}
    for key, dist in kit.sd(P).items():
        depth = -dist
        for h in "LR":
            if any(t0 <= fr["t"] <= t1 and key[0] == p and key[1] in ps for t0, t1, p, ps in allow.get(h, ())):
                depth = np.where((side == h) & np.isin(part, ["hand", "fore"]), -1.0, depth)
        for j in np.nonzero(depth > TOL)[0]:
            k = (f"{side[j]}.{part[j]}", f"{key[0]}.{key[1]}")
            hits[k] = max(hits.get(k, 0.0), float(depth[j]))
    for h, hd in fr["hands"].items():
        a, b = np.array(hd["butt"]), np.array(hd["tip"])
        r = dc._sd_capsule_pts(P, a, b, dc.STICK_R[0], dc.STICK_R[1])
        depth = np.where((side == h) & np.isin(part, ["hand", "fore"]), -1.0, -r)
        for j in np.nonzero(depth > TOL)[0]:
            k = (f"{side[j]}.{part[j]}({src[j]})", f"{h}.stick")
            hits[k] = max(hits.get(k, 0.0), float(depth[j]))
    Pv = [tuple(v) for v in P]
    trees = {k: BVHTree.FromPolygons(Pv, ps, all_triangles=False) for k, ps in SELF.items()}
    for a, b in SELF_PAIRS:
        if trees[a].overlap(trees[b]):
            d = self_depth(Pv, trees, a, b)
            if d > TOL:
                hits[(a, b)] = d
    for k, d in hits.items():
        n, mx, at = stats.get(k, (0, 0.0, -1))
        stats[k] = (n + 1, max(mx, d), i if d > mx else at)
        worst.append((d, i, f"{k[0]} x {k[1]}"))
    if i % 96 == 0:
        print("frame", i, len(hits), flush=True)

worst.sort(reverse=True)
rep = {"anim": os.path.basename(anim_path), "frames_checked": len(todo),
       "frames_clipping": len({i for _, i, _ in worst}), "tol_mm": TOL * 1000,
       "pairs": {f"{a} x {b}": {"frames": n, "max_mm": round(1000 * mx, 1), "worst_frame": at}
                 for (a, b), (n, mx, at) in sorted(stats.items(), key=lambda x: -x[1][1])},
       "worst": [{"frame": i, "pair": k, "mm": round(1000 * d, 1)} for d, i, k in worst[:12]]}
with open(out_json, "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep, indent=1), flush=True)

"""Body gate for the animated drummer: torso, head and arm measures against real drummers in video.

Scores a drums anim JSON (blockout/drums.py) against blockout/drum_body_ref.json: distributions of the same measures
taken from SAM 3D Body mocap of drummers in footage (h3band/sam3d_mocap.py writes the BVH). Both sides are reduced to
the same points and head axes, and only angles are measured, so the rigs' scale and units drop out:

    pelvis, hip L/R, knee L/R, neck, head; shoulder, elbow, wrist L/R; the head's forward and up axes

Posture measures are per frame. The head and arms are measured in the trunk frame (up = pelvis -> neck, right = left
-> right shoulder); the trunk in the pelvis frame (right = left -> right hip, forward = along the thighs). Motion
measures are per 2 s window inside one camera shot: the p5-p95 range of a posture measure, or how far the trunk's
direction and the shoulder line move in the world.

In drum footage the hips and thighs are behind the kit, so SAM 3D Body guesses them; the pelvis-frame measures are
reported but not gated.

    python -m blockout.drum_body_score ANIM.json [--ref blockout/drum_body_ref.json] [--json OUT.json] [--curves OUT.png]
                                                 [--groove]
    python -m blockout.drum_body_score --build-ref SOURCES.json OUT_REF.json     # rebuild the reference

SOURCES.json: {"sources": [{"id", "bvh", "content", "shots": [{"id", "t": [t0, t1], "view"}]}]}; BVH paths are
relative to the sources file. Shot edges lose TRIM seconds (the mocap smoother runs across cuts).

Gate per measure, as blockout/human_score.py: OUT if our median is outside the reference 25-75% band, or our 5th / 95th
percentile is outside the reference 5-95% range. Overall PASS = no gated measure OUT.

The footage is a groove: the right hand on the hi-hat and the left on the snare (GROOVE), and no ride, toms or crashes.
--groove compares like with like: posture on the frames where each hand's last and next strokes are on its GROOVE
piece, and motion on the windows that fit inside those stretches. Without it, a take's fills and cymbals count too.
"""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "drum_body_ref.json")
PCTS = (5, 25, 50, 75, 95)
TRIM = 0.2                              # s cut from each end of a shot
WIN, STEP = 2.0, 0.5                    # motion windows (s)

# MHR joint indices (SAM 3D Body's rig; the BVH names them joint_###, the root "Hips" is joint 1)
MHR = {"pelvis": 1, "hipL": 2, "kneeL": 3, "hipR": 18, "kneeR": 19, "neck": 110, "head": 113,
       "shR": 39, "elR": 40, "wrR": 42, "shL": 75, "elL": 76, "wrL": 78}

ARM = ("elbow_flex", "ua_elev", "ua_abd", "ua_fwd")
GROOVE = {"R": "hihat", "L": "snare"}   # what each hand plays in the reference footage (--groove)
# name: (unit, gated, description)
MEASURES = {
    "head_pitch": ("deg", True, "head forward axis below the trunk's forward, + = looking down"),
    "head_yaw": ("deg", True, "head turned from the trunk's forward, + = toward the drummer's right"),
    "head_roll": ("deg", True, "head up axis tipped from the trunk's up, + = toward the right shoulder"),
    "trunk_flex": ("deg", False, "pelvis -> neck tipped forward of the pelvis frame's up (hips hidden in footage)"),
    "trunk_side": ("deg", False, "pelvis -> neck tipped toward the right (hips hidden in footage)"),
    "trunk_twist": ("deg", False, "shoulder line turned from the hip line, + = chest toward the right (hips hidden)"),
    **{f"{k}_{s}": ("deg", True, d.format(s)) for s in "RL" for k, d in (
        ("elbow_flex", "{} elbow flexion: upper arm to forearm, 0 = straight"),
        ("ua_elev", "{} upper arm from hanging (trunk down), 90 = level"),
        ("ua_abd", "{} upper arm out to the side of the trunk, + = elbow away from the body"),
        ("ua_fwd", "{} upper arm forward of the trunk, + = elbow in front"))},
    # where the hands work, in arm lengths (shoulder -> elbow -> wrist); set by the kit layout, so not gated
    **{f"wrist_{k}_{s}": ("arm", False, f"{s} wrist {d} of the shoulder in the trunk frame, in arm lengths (kit layout)")
       for s in "RL" for k, d in (("fwd", "forward"), ("up", "up"), ("out", "out to the side"))},
    # motion, per 2 s window
    "nod": ("deg", True, "head_pitch p5-p95 range in a 2 s window"),
    "head_turn": ("deg", True, "head_yaw p5-p95 range in a 2 s window"),
    "head_tilt": ("deg", True, "head_roll p5-p95 range in a 2 s window"),
    "trunk_sway": ("deg", True, "p95 angle of the trunk's up axis from its 2 s mean, in the world"),
    "shoulder_turn": ("deg", True, "p5-p95 range of the shoulder line's heading about the trunk axis, in the world"),
    **{f"{k}_{s}": ("deg", True, f"{s} {base} p5-p95 range in a 2 s window") for s in "RL"
       for k, base in (("elbow_lift", "ua_elev"), ("elbow_bend", "elbow_flex"))},
}
MOTION = ("nod", "head_turn", "head_tilt", "trunk_sway", "shoulder_turn",
          "elbow_lift_R", "elbow_lift_L", "elbow_bend_R", "elbow_bend_L")
POSTURE = tuple(k for k in MEASURES if k not in MOTION)


def _n(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _orth(v, ax):
    return _n(v - ax * np.sum(v * ax, -1, keepdims=True))


def _dot(a, b):
    return np.sum(a * b, -1)


def _deg(y, x):
    return np.degrees(np.arctan2(y, x))


# ---------------------------------------------------------------- BVH -> points

def read_bvh(path):
    """-> names, parents, offsets (J,3), channel names per joint, motion (F, C), frame time."""
    lines = open(path).read().splitlines()
    names, parents, offsets, chans, stack, i, end = [], [], [], [], [], 0, 0
    while not lines[i].strip().startswith("MOTION"):
        w = lines[i].split()
        if w and w[0] in ("ROOT", "JOINT"):
            names.append(w[1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); chans.append([])
            stack.append(len(names) - 1)
        elif w and w[0] == "End":
            end += 1
        elif w and w[0] == "OFFSET" and not end:
            offsets[stack[-1]] = [float(x) for x in w[1:4]]
        elif w and w[0] == "CHANNELS":
            chans[stack[-1]] = w[2:]
        elif w and w[0] == "}":
            if end:
                end -= 1
            else:
                stack.pop()
        i += 1
    nf = int(lines[i + 1].split()[1])
    ft = float(lines[i + 2].split()[2])
    motion = np.loadtxt(lines[i + 3:i + 3 + nf], dtype=float, ndmin=2)
    return names, parents, np.asarray(offsets, float), chans, motion, ft


def _axis_rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i = "XYZ".index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    R[..., i, i] = 1
    R[..., j, j], R[..., j, k], R[..., k, j], R[..., k, k] = c, -s, s, c
    return R


def bvh_fk(path):
    """Global joint positions (F, J, 3) and rotations (F, J, 3, 3), names, fps."""
    names, parents, offsets, chans, motion, ft = read_bvh(path)
    F, J = len(motion), len(names)
    P, G = np.zeros((F, J, 3)), np.zeros((F, J, 3, 3))
    col = 0
    for j in range(J):
        R, pos = np.broadcast_to(np.eye(3), (F, 3, 3)).copy(), None
        for c in chans[j]:
            v = motion[:, col]; col += 1
            if c.endswith("position"):
                pos = np.zeros((F, 3)) if pos is None else pos
                pos[:, "XYZ".index(c[0])] = v
            else:
                R = R @ _axis_rot(c[0], v)
        p = parents[j]
        if p < 0:
            P[:, j] = pos if pos is not None else offsets[j]
            G[:, j] = R
        else:
            P[:, j] = P[:, p] + np.einsum("fij,j->fi", G[:, p], offsets[j])
            G[:, j] = G[:, p] @ R
    return P, G, names, 1.0 / ft


def mhr_points(path):
    """BVH from h3band/sam3d_mocap.py -> (points {name: (F,3)}, head forward (F,3), head up (F,3), fps, report)."""
    P, G, names, fps = bvh_fk(path)
    idx = {(1 if n == "Hips" else int(n.split("_")[1])): j for j, n in enumerate(names)}
    pts = {k: P[:, idx[v]] for k, v in MHR.items()}
    # left/right: the thighs point forward from a seated drummer, so forward x up must point to the right shoulder
    up = _n(pts["neck"] - pts["pelvis"])
    thigh = _n(_n(pts["kneeL"] - pts["hipL"]) + _n(pts["kneeR"] - pts["hipR"]))
    chir = float(np.median(_dot(np.cross(thigh, up), _n(pts["shR"] - pts["shL"]))))
    if chir < 0:
        for a in ("hip", "knee", "sh", "el", "wr"):
            pts[a + "L"], pts[a + "R"] = pts[a + "R"], pts[a + "L"]
    # head axes: the joint frame's signed local axes that best follow the trunk's forward and up
    right = _orth(pts["shR"] - pts["shL"], up)
    fwd = np.cross(up, right)
    Gh = G[:, idx[MHR["head"]]]
    cand = [(a, s) for a in range(3) for s in (1, -1)]
    fa = max(cand, key=lambda c: np.mean(_dot(c[1] * Gh[:, :, c[0]], fwd)))
    ua = max((c for c in cand if c[0] != fa[0]), key=lambda c: np.mean(_dot(c[1] * Gh[:, :, c[0]], up)))
    hf, hu = fa[1] * Gh[:, :, fa[0]], ua[1] * Gh[:, :, ua[0]]
    rep = {"frames": len(P), "fps": round(fps, 3), "lr_check": round(chir, 3), "lr_swapped": chir < 0,
           "head_fwd_axis": f"{'+' if fa[1] > 0 else '-'}{'xyz'[fa[0]]}", "head_up_axis": f"{'+' if ua[1] > 0 else '-'}{'xyz'[ua[0]]}",
           "bone_cv": {b: round(float(np.std(L) / np.mean(L)), 4) for b, L in (
               ("upper_arm_R", np.linalg.norm(pts["elR"] - pts["shR"], axis=1)),
               ("forearm_R", np.linalg.norm(pts["wrR"] - pts["elR"], axis=1)),
               ("trunk", np.linalg.norm(pts["neck"] - pts["pelvis"], axis=1)))}}
    return pts, hf, hu, fps, rep


def anim_points(anim):
    """drums.py anim JSON -> (points, head forward, head up, fps). The arm points are the drawn ones (hands[side])."""
    d = json.load(open(anim)) if isinstance(anim, str) else anim
    fr = d["frames"]
    A = lambda f: np.array([f(x) for x in fr], float)
    pts = {"pelvis": A(lambda x: x["body"]["hips"]), "neck": A(lambda x: x["body"]["neck"]),
           "head": A(lambda x: x["body"]["head"])}
    for s in "LR":
        pts["hip" + s] = A(lambda x: x["feet"][s]["hip"])
        pts["knee" + s] = A(lambda x: x["feet"][s]["knee"])
        pts["sh" + s] = A(lambda x: x["hands"][s]["shoulder"])
        pts["el" + s] = A(lambda x: x["hands"][s]["elbow"])
        pts["wr" + s] = A(lambda x: x["hands"][s]["wrist"])
    Rh = A(lambda x: x["body"]["head_rot"])
    return pts, Rh[:, :, 1], Rh[:, :, 2], float(d["fps"])


# ---------------------------------------------------------------- measures

def posture(pts, hf, hu):
    """Per-frame posture measures {name: (F,)} plus the trunk axes (for the motion measures)."""
    up = _n(pts["neck"] - pts["pelvis"])
    right = _orth(pts["shR"] - pts["shL"], up)
    fwd = np.cross(up, right)
    hf, hu = _n(hf), _n(hu)
    m = {"head_pitch": _deg(-_dot(hf, up), _dot(hf, fwd)),
         "head_yaw": _deg(_dot(hf, right), _dot(hf, fwd)),
         "head_roll": _deg(_dot(hu, right), _dot(hu, up))}
    xh = _n(pts["hipR"] - pts["hipL"])
    thigh = _n(_n(pts["kneeL"] - pts["hipL"]) + _n(pts["kneeR"] - pts["hipR"]))
    fp = _orth(thigh, xh)
    upp = np.cross(xh, fp)
    S = pts["neck"] - pts["pelvis"]
    m["trunk_flex"] = _deg(_dot(S, fp), _dot(S, upp))
    m["trunk_side"] = _deg(_dot(S, xh), _dot(S, upp))
    a, b = _orth(xh, up), right
    m["trunk_twist"] = -_deg(_dot(np.cross(a, b), up), _dot(a, b))
    for s, out in (("R", 1.0), ("L", -1.0)):
        ua = _n(pts["el" + s] - pts["sh" + s])
        fa = _n(pts["wr" + s] - pts["el" + s])
        m[f"elbow_flex_{s}"] = np.degrees(np.arccos(np.clip(_dot(ua, fa), -1, 1)))
        m[f"ua_elev_{s}"] = np.degrees(np.arccos(np.clip(-_dot(ua, up), -1, 1)))
        m[f"ua_abd_{s}"] = _deg(out * _dot(ua, right), -_dot(ua, up))
        m[f"ua_fwd_{s}"] = _deg(_dot(ua, fwd), -_dot(ua, up))
        L = (np.linalg.norm(pts["el" + s] - pts["sh" + s], axis=-1)
             + np.linalg.norm(pts["wr" + s] - pts["el" + s], axis=-1))
        w = pts["wr" + s] - pts["sh" + s]
        m[f"wrist_fwd_{s}"] = _dot(w, fwd) / L
        m[f"wrist_up_{s}"] = _dot(w, up) / L
        m[f"wrist_out_{s}"] = out * _dot(w, right) / L
    return m, (up, right)


def _rng(x):
    p5, p95 = np.percentile(x, (5, 95))
    return float(p95 - p5)


def motion(m, axes, fps, spans):
    """Per-window motion measures {name: [values]} over windows of WIN s inside each (f0, f1) frame span."""
    up, right = axes
    w, st = int(round(WIN * fps)), max(1, int(round(STEP * fps)))
    out = {k: [] for k in MOTION}
    for f0, f1 in spans:
        for a in range(f0, f1 - w + 1, st):
            sl = slice(a, a + w)
            out["nod"].append(_rng(m["head_pitch"][sl]))
            out["head_turn"].append(_rng(m["head_yaw"][sl]))
            out["head_tilt"].append(_rng(m["head_roll"][sl]))
            mu = _n(up[sl].mean(0))
            out["trunk_sway"].append(float(np.percentile(np.degrees(np.arccos(np.clip(up[sl] @ mu, -1, 1))), 95)))
            r0 = _orth(right[sl].mean(0)[None], mu[None])[0]
            rr = _orth(right[sl], mu[None])
            out["shoulder_turn"].append(_rng(_deg(np.cross(r0, rr) @ mu, rr @ r0)))
            for s in "RL":
                out[f"elbow_lift_{s}"].append(_rng(m[f"ua_elev_{s}"][sl]))
                out[f"elbow_bend_{s}"].append(_rng(m[f"elbow_flex_{s}"][sl]))
    return out


def wpct(x, q=PCTS):
    return [float(v) for v in np.percentile(np.asarray(x, float), q)]


def _band(vals):
    return {f"p{q}": round(v, 2) for q, v in zip(PCTS, wpct(vals))} | {"n": len(vals)}


# ---------------------------------------------------------------- building the reference

def build_ref(sources_json, out_json):
    src = json.load(open(sources_json))
    root = os.path.dirname(os.path.abspath(sources_json))
    pooled, per_shot, series = {k: [] for k in MEASURES}, {}, {}
    for s in src["sources"]:
        pts, hf, hu, fps, rep = mhr_points(os.path.join(root, s["bvh"]))
        s["mocap"] = rep
        m, axes = posture(pts, hf, hu)
        F = len(m["head_pitch"])
        for sh in s["shots"]:
            f0 = max(0, int(math.ceil((sh["t"][0] + TRIM) * fps)))
            f1 = min(F, int(math.floor((sh["t"][1] - TRIM) * fps)))
            key = f"{s['id']}:{sh['id']}"
            mo = motion(m, axes, fps, [(f0, f1)])
            vals = {k: m[k][f0:f1].tolist() for k in POSTURE} | mo
            per_shot[key] = {"view": sh.get("view"), "frames": f1 - f0, "windows": len(mo["nod"]),
                             "bands": {k: _band(v) for k, v in vals.items() if len(v)}}
            series[key] = {"t0": round(f0 / fps, 3), "fps": round(fps, 3),
                           **{k: [round(x, 3 if MEASURES[k][0] == "arm" else 1) for x in m[k][f0:f1]] for k in POSTURE}}
            for k, v in vals.items():
                pooled[k].extend(v)
    ref = {k: v for k, v in src.items() if k != "sources"} | {"sources": src["sources"], "trim_s": TRIM, "window_s": WIN,
                                                               "step_s": STEP, "measures": {}}
    for k, (unit, gated, desc) in MEASURES.items():
        ref["measures"][k] = {"unit": unit, "gated": gated, "desc": desc, "pooled": _band(pooled[k]),
                              "per_shot": {sk: v["bands"][k] for sk, v in per_shot.items() if k in v["bands"]}}
    ref["shots"] = {sk: {kk: v[kk] for kk in ("view", "frames", "windows")} for sk, v in per_shot.items()}
    ref["n_frames"] = len(pooled["head_pitch"])
    ref["n_windows"] = len(pooled["nod"])
    ref["series"] = series
    json.dump(ref, open(out_json, "w"), indent=1)
    print(f"wrote {out_json}: {ref['n_frames']} frames, {ref['n_windows']} windows from {len(per_shot)} shots")
    for s in src["sources"]:
        print(f"  {s['id']}: {json.dumps(s['mocap'])}")


# ---------------------------------------------------------------- scoring an anim

def groove_frames(anim):
    """(F,) bool: frames where each hand's last and next strokes are on the piece it plays in the footage (GROOVE)."""
    d = json.load(open(anim)) if isinstance(anim, str) else anim
    t = np.array([f["t"] for f in d["frames"]])
    ok = np.ones(len(t), bool)
    for hand, piece in GROOVE.items():
        hs = sorted((h["t"], h["piece"].split("_")[0].rstrip("0123456789")) for h in d["hits"] if h["hand"] == hand)
        if not hs:
            return np.zeros(len(t), bool)
        on = np.array([p == piece for _, p in hs])
        i = np.clip(np.searchsorted([x for x, _ in hs], t, "right") - 1, 0, len(hs) - 1)
        ok &= on[i] & on[np.minimum(i + 1, len(hs) - 1)]
    return ok


def anim_measures(anim, groove=False):
    """Posture {name: (F,)} and motion {name: [values]} of an anim; groove: posture on groove_frames only, and motion
    on the windows inside them."""
    d = json.load(open(anim)) if isinstance(anim, str) else anim
    pts, hf, hu, fps = anim_points(d)
    m, axes = posture(pts, hf, hu)
    F = len(m["head_pitch"])
    if not groove:
        return m, motion(m, axes, fps, [(0, F)]), fps
    k = groove_frames(d)
    edges = np.flatnonzero(np.diff(np.r_[0, k.astype(int), 0]))
    mo = motion(m, axes, fps, list(zip(edges[::2], edges[1::2])))
    return {n: v[k] for n, v in m.items()}, mo, fps


def score(anim, ref_path=REF, out_json=None, quiet=False, groove=False):
    ref = json.load(open(ref_path))
    m, mo, fps = anim_measures(anim, groove)
    if not len(m["head_pitch"]):
        raise SystemExit(f"{anim}: no frames with the right hand on the hi-hat and the left on the snare (--groove)")
    ours = {k: m[k] for k in POSTURE} | mo
    res, fails = {}, []
    say = (lambda *a: None) if quiet else print
    say(f"anim {anim}: {len(m['head_pitch'])} {'groove ' if groove else ''}frames, {len(mo['nod'])} windows  |  "
        f"reference: {ref['n_frames']} frames, {ref['n_windows']} windows from {len(ref['shots'])} shots")
    hdr = f"{'measure':14s} {'ours p5/p25/p50/p75/p95':>36s}   {'ref p5/p25/p50/p75/p95':>36s}  {'%out':>5s}  verdict"
    say(hdr); say("-" * len(hdr))
    for k, spec in ref["measures"].items():
        h = spec["pooled"]
        o = wpct(ours[k])
        dp = 2 if spec["unit"] == "arm" else 1
        hp = [h[f"p{q}"] for q in PCTS]
        x = np.asarray(ours[k], float)
        frac = float(np.mean((x < h["p5"]) | (x > h["p95"])))
        why = []
        if not (h["p25"] <= o[2] <= h["p75"]):
            why.append(f"median {o[2]:.{dp}f} outside ref IQR [{h['p25']:.{dp}f},{h['p75']:.{dp}f}]")
        if o[0] < h["p5"]:
            why.append(f"p5 {o[0]:.{dp}f} < ref p5 {h['p5']:.{dp}f}")
        if o[4] > h["p95"]:
            why.append(f"p95 {o[4]:.{dp}f} > ref p95 {h['p95']:.{dp}f}")
        out = bool(why) and spec["gated"]
        if out:
            fails.append(k)
        tag = "OUT" if out else ("ok" if spec["gated"] else "info")
        say(f"{k:14s} {''.join(f'{v:7.{dp}f}' for v in o):>36s}   {''.join(f'{v:7.{dp}f}' for v in hp):>36s}  "
            f"{100 * frac:5.1f}  {tag}{'  (' + '; '.join(why) + ')' if why else ''}")
        res[k] = {"ours": dict(zip([f"p{q}" for q in PCTS], o)), "ref": h, "frac_outside_ref5_95": frac,
                  "out": out, "why": why}
    say("\nper shot (info): ref p50 by shot, ours p50")
    shots = list(ref["shots"])
    say(f"  {'measure':14s} " + " ".join(f"{s.split(':')[-1]:>9s}" for s in shots) + f" {'ours':>9s}")
    for k, spec in ref["measures"].items():
        dp = 2 if spec["unit"] == "arm" else 1
        say(f"  {k:14s} " + " ".join(f"{spec['per_shot'][s]['p50']:9.{dp}f}" if s in spec["per_shot"] else f"{'-':>9s}"
                                    for s in shots) + f" {np.median(ours[k]):9.{dp}f}")
    gated = [k for k in res if ref["measures"][k]["gated"]]
    verdict = "PASS" if not fails else "FAIL"
    say(f"\nOVERALL: {verdict}  ({len(fails)} of {len(gated)} gated measures OUT{': ' + ', '.join(fails) if fails else ''})")
    if out_json:
        json.dump({"anim": anim, "groove": groove, "verdict": verdict, "out": fails, "measures": res},
                  open(out_json, "w"), indent=1)
    return verdict, res


def curves(anims, ref_path, out_png, keys=("head_pitch", "head_yaw", "head_roll", "elbow_flex_R", "elbow_flex_L",
                                           "ua_elev_R", "ua_elev_L", "ua_abd_R", "ua_abd_L")):
    """Per-frame curves: the reference shots (left) next to each anim (right), shared y-range per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = json.load(open(ref_path))
    runs = [(os.path.splitext(os.path.basename(a))[0], anim_measures(a)) for a in anims]
    fig, axs = plt.subplots(len(keys), 1 + len(runs), figsize=(4.2 * (1 + len(runs)), 1.55 * len(keys)),
                            sharey="row", squeeze=False)
    for r, k in enumerate(keys):
        ax = axs[r][0]
        for sk, s in ref["series"].items():
            t = s["t0"] + np.arange(len(s[k])) / s["fps"]
            ax.plot(t, s[k], lw=0.9, label=f"{sk.split(':')[-1]} ({ref['shots'][sk]['view']})")
        ax.set_ylabel(k, fontsize=8)
        for c, (name, (m, _, fps)) in enumerate(runs, 1):
            t = np.arange(len(m[k])) / fps
            axs[r][c].plot(t, m[k], lw=0.9, color="k")
        for ax in axs[r]:
            h = ref["measures"][k]["pooled"]
            ax.axhspan(h["p25"], h["p75"], color="0.85", zorder=0)
            ax.tick_params(labelsize=7)
    axs[0][0].set_title("reference (SAM 3D Body), clip time s", fontsize=9)
    axs[0][0].legend(fontsize=6, loc="upper right")
    for c, (name, _) in enumerate(runs, 1):
        axs[0][c].set_title(f"{name}, s", fontsize=9)
    fig.text(0.5, 0.002, "grey band: reference p25-p75", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    fig.savefig(out_png, dpi=110)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anim", nargs="*")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--json")
    ap.add_argument("--curves", help="PNG of the reference curves next to each anim's")
    ap.add_argument("--build-ref", nargs=2, metavar=("SOURCES_JSON", "OUT_REF"))
    ap.add_argument("--groove", action="store_true",
                    help="only the frames where the hands play what the footage's do (right hi-hat, left snare)")
    a = ap.parse_args()
    if a.build_ref:
        build_ref(*a.build_ref)
    elif a.anim:
        if a.curves:
            curves(a.anim, a.ref, a.curves)
        ok = True
        for an in a.anim:
            v, _ = score(an, a.ref, a.json if len(a.anim) == 1 else None, groove=a.groove)
            ok &= v == "PASS"
            print()
        sys.exit(0 if ok else 1)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

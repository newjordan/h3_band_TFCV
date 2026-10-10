"""Hitboxes for the drum blockout: the kit and the drummer's arms and sticks as signed-distance shapes.

The shapes match what blender_drums.py draws. Every round piece is a solid of revolution about its normal:
its parts are 2D shapes (rectangles, or thick segments for cymbal bows) in the (radial q, height d) plane of
the piece, posed per frame by the piece's tilt (and the hi-hat's lift). Stands and tom mounts are rods. The
drummer is capsules: stick, fist (the hand around the stick: palm and knuckles, drum_hands.hitbox), forearm,
upper arm, thigh, head.

    python -m blockout.drum_collide ANIM.json        # report every clip in an animation from drums.py

A contact is allowed when it is playing: the stick's tip on the piece the hand is playing (sunk at most
TIP_SINK), the shaft on the hoop for a rimshot or a cross-stick, the fist and stick on a cymbal being choked.
Everything else deeper than TOL is a clip.
"""
import json, math

import numpy as np

from . import drum_hands

TOL = 0.002                     # m of overlap ignored (mesh faceting)
UNDER_BAND = 0.04               # m under a played cymbal's bow where the tip counts as caught underneath
TIP_SINK = 0.006                # m below the playing surface the tip's centre may go (head give + tip radius)
STICK_R = (0.0075, 0.0055)      # butt, tip radius (blender_drums.py)
ARM_R = {"upper": (0.046, 0.038), "fore": (0.036, 0.028)}
THIGH_R, SHIN_R, HEAD_R, TORSO_R = (0.075, 0.058), (0.055, 0.042), 0.095, 0.11
HOOP_W, HOOP_UP, HOOP_DOWN = 0.008, 0.006, 0.012
CHINA_EDGE = 0.012              # m a china's edge stands above its bell's base (upturned)
BOOM_RISE = 0.35                # m a boom arm climbs from the top of its post to the cymbal


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _rot(aa):
    """Rotation matrix for an axis-angle vector."""
    aa = np.asarray(aa, float)
    a = float(np.linalg.norm(aa))
    if a < 1e-9:
        return np.eye(3)
    k = aa / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# ---------------------------------------------------------------- 2D shapes in the (q, d) plane
def _sd_rect(q, d, q0, q1, d0, d1):
    cq, cd = (q0 + q1) / 2, (d0 + d1) / 2
    hq, hd = (q1 - q0) / 2, (d1 - d0) / 2
    x, y = np.abs(q - cq) - hq, np.abs(d - cd) - hd
    return np.minimum(np.maximum(x, y), 0) + np.hypot(np.maximum(x, 0), np.maximum(y, 0))


def _sd_seg(q, d, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((q - a[0]) * ab[0] + (d - a[1]) * ab[1]) / float(ab @ ab), 0, 1)
    return np.hypot(q - a[0] - t * ab[0], d - a[1] - t * ab[1]) - r


def _sd_stack(q, d, q0, q1, lo, hi, r, lift):
    """The region from q0 to q1 between a lower line (heights lo at q0 and q1) and an upper one (hi, raised by
    lift), r thick: the max of its four sides' distances (exact inside, a little short outside its corners)."""
    out = [q - q1, q0 - q]
    for (h0, h1), sgn, up in ((lo, -1, 0.0), (hi, 1, lift)):
        t = _unit([q1 - q0, h1 - h0])
        out.append(sgn * (-(q - q0) * t[1] + (d - h0 - up) * t[0]))
    return np.maximum.reduce(out) - r


def piece_parts(p, k):
    """[(part name, kind, args)] for piece p of the kit dict (drums.py KIT / anim['kit'])."""
    r = k["r"]
    if k["kind"] in ("drum", "kick"):
        dep = k["depth"]
        return [("head", "rect", (0.0, r, -dep, 0.0)),
                ("shell", "rect", (r, r + HOOP_W, -dep - 0.006, 0.0)),
                ("hoop", "rect", (r - 0.001, r + HOOP_W + 0.001, -HOOP_DOWN, HOOP_UP))]
    edge = CHINA_EDGE if _ptype(p) == "china" else -0.003     # a china's edge turns up
    inner = 0.009 if edge < 0 else 0.0
    bell = ("bell", "seg", ((0.02, 0.026), (0.055, 0.006), 0.004))
    if k["kind"] == "hihat":
        # both cymbals and the gap between them (however far the pedal opens it) as one solid, so a stick caught
        # between them is pushed out over the top or under the bottom instead of back and forth
        return [("hats", "stack", (0.03, r, (-0.004, -0.010), (inner, edge), 0.0015)), bell]
    return [("bow", "seg", ((0.03, inner), (r, edge), 0.0015)), bell]


def _ptype(p):
    return p.split("_")[0].rstrip("0123456789")       # drum_kit.piece_type


def on_kick(kit, p):
    """A rack tom close enough over the kick hangs from it; others stand on their own."""
    k, kk = kit[p], kit.get("kick")
    return kk is not None and abs(k["c"][0] - kk["c"][0]) < kk["r"] + 0.10 and k["c"][1] > kk["c"][1] - 0.05


def stands(kit):
    """[(name, a, b, radius)] rods, as blender_drums.py builds them."""
    out = []
    for p, k in kit.items():
        c, n = np.array(k["c"], float), np.array(k["n"], float)
        t = _ptype(p)
        if t == "snare" or (t == "tom" and not on_kick(kit, p)):
            b = c - n * k["depth"]
            out.append((p + "_stand", np.array([b[0], b[1], 0.25 if t == "snare" else 0.30]), b, 0.012))
        elif t == "tom":
            kk = kit["kick"]
            kc = np.array(kk["c"]) + np.array([0, kk["depth"] * 0.4, kk["r"]])
            out.append((p + "_mount", c - n * k["depth"], kc, 0.011))
        elif t == "floor":
            for a in (30, 150, 270):
                o = np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0]) * (k["r"] + 0.02)
                top = c + o + np.array([0, 0, -0.05])
                out.append((f"{p}_leg{a}", top, np.array([*(c + o * 1.25)[:2], 0.0]), 0.008))
        elif k["kind"] == "hihat":
            top = c - np.array([0, 0, 0.012])
            out.append((p + "_stand", np.array([top[0], top[1], 0.30]), top, 0.012))
            out.append((p + "_rod", top, c + np.array([0, 0, 0.06]), 0.004))
        elif k["kind"] == "cymbal":
            top = c - n * 0.02
            split = 0.35 if c[2] > 1.1 else 0.30
            if "stand" in k:        # boom stand: post at k["stand"] (x, y), arm up to the cymbal
                base = np.array([*k["stand"], split])
                joint = np.array([base[0], base[1], max(split + 0.25, top[2] - BOOM_RISE)])
                out.append((p + "_stand", base, joint, 0.012))
                out.append((p + "_boom", joint, top, 0.010))
            else:
                out.append((p + "_stand", np.array([top[0], top[1], split]), top, 0.012))
    return out


class Kit:
    """The kit's hitboxes. pose(frame) moves them to that frame; sd(points) gives per-part signed distances."""

    def __init__(self, kit, hihat_open=0.018):
        self.kit, self.hihat_open = kit, hihat_open
        self.parts = {p: piece_parts(p, k) for p, k in kit.items()}
        self.rods = stands(kit)
        self.pose({})

    def pose(self, fr):
        tilt = fr.get("tilt", {}) if fr else {}
        lift = (fr.get("hihat", {}).get("open", 0.0) * self.hihat_open) if fr else 0.0
        self.frame = {}
        for p, k in self.kit.items():
            c, n = np.array(k["c"], float), np.array(k["n"], float)
            if p in tilt:
                n = _rot(tilt[p]) @ n
            self.frame[p] = (c, _unit(n), lift if p == "hihat" else 0.0)     # only the main hat is on the pedal
        return self

    def sd(self, pts, pieces=None):
        """{(piece, part): (M,) signed distance} for points (M, 3)."""
        pts = np.atleast_2d(pts)
        out = {}
        for p in pieces or self.kit:
            c, n, lift = self.frame[p]
            v = pts - c
            d = v @ n
            q = np.linalg.norm(v - d[:, None] * n, axis=1)
            for name, kind, args in self.parts[p]:
                dd = d - lift if name in ("bow", "bell") else d
                if kind == "rect":
                    out[(p, name)] = _sd_rect(q, dd, *args)
                elif kind == "stack":
                    out[(p, name)] = _sd_stack(q, d, *args, lift)
                else:
                    out[(p, name)] = _sd_seg(q, dd, *args)
        if pieces is None:
            for name, a, b, r in self.rods:
                out[(name, "rod")] = _sd_capsule_pts(pts, a, b, r)
        return out

    def push(self, pts, pieces=None, skip=()):
        """Per point: (depth, outward unit normal, (piece, part)) of the deepest overlap, depth <= 0 if clear."""
        sd = {k: v for k, v in self.sd(pts, pieces).items() if k not in skip}
        keys = list(sd)
        if not keys:
            return np.zeros(len(pts)), np.zeros((len(pts), 3)), [None] * len(pts)
        S = np.stack([sd[k] for k in keys])
        i = np.argmin(S, axis=0)
        depth = -S[i, np.arange(len(pts))]
        normals = np.zeros((len(pts), 3))
        e = 1e-4
        for j in np.nonzero(depth > 0)[0]:
            k = keys[i[j]]
            g = np.zeros(3)
            for ax in range(3):
                dp = np.zeros(3); dp[ax] = e
                g[ax] = (self._one(pts[j] + dp, k) - self._one(pts[j] - dp, k)) / (2 * e)
            normals[j] = _unit(g)
        return depth, normals, [keys[x] for x in i]

    def _one(self, pt, key):
        p, part = key
        if part == "rod":
            _, a, b, r = next(x for x in self.rods if x[0] == p)
            return float(_sd_capsule_pts(pt[None], a, b, r)[0])
        return float(self.sd(pt[None], [p])[key][0])


def _sd_capsule_pts(pts, a, b, r, r1=None):
    """Distance from points to a capsule a-b of radius r (tapering to r1 at b)."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) - (r if r1 is None else r + t * (r1 - r))


def seg_seg(p1, q1, p2, q2):
    """Closest points (s, t in 0..1) and distance between segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    if a < 1e-12 and e < 1e-12:
        s = t = 0.0
    elif a < 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    elif e < 1e-12:
        s, t = float(np.clip(-c / a, 0, 1)), 0.0
    else:
        den = a * e - b * b
        s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-12 else 0.0
        t = (b * s + f) / e
        if t < 0:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        elif t > 1:
            t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    c1, c2 = p1 + s * d1, p2 + t * d2
    return s, t, float(np.linalg.norm(c1 - c2)), c1, c2


# ---------------------------------------------------------------- the drummer
def samples(a, b, r0, r1, n):
    s = np.linspace(0, 1, n)
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + s[:, None] * (b - a), r0 + s * (r1 - r0), s


def body_parts(hd, hand):
    """Capsules of one hand's chain: [(part, a, b, r0, r1, samples)]. The "fist" is the hand around the stick
    (drum_hands.hitbox: palm, and knuckles with the fingers curled under)."""
    tip, butt, grip = np.array(hd["tip"]), np.array(hd["butt"]), np.array(hd["grip"])
    d = np.array(hd["hand_dir"]) if "hand_dir" in hd else _unit(tip - butt)     # hand_dir: stick in the air
    return [("stick", butt, tip, *STICK_R, 18),
            *(("fist", a, b, r0, r1, 4) for a, b, r0, r1 in drum_hands.hitbox(grip, d, hand)),
            ("fore", np.array(hd["elbow"]), np.array(hd["wrist"]), *ARM_R["fore"], 8),
            ("upper", np.array(hd["shoulder"]), np.array(hd["elbow"]), *ARM_R["upper"], 6)]


def allowed_windows(hits, chokes, fps):
    """Per hand: [(t0, t1, piece, parts)] when touching those parts of that piece is playing, not clipping.
    hits: [{"tf", "hand", "piece", "tech"?}], chokes: [{"piece", "hand", "t0", "t1"}], in one time base."""
    out = {"R": [], "L": []}
    dt = 1.5 / fps
    for h in hits:
        tech = h.get("tech")
        if tech == "rimshot":
            out[h["hand"]].append((h["tf"] - dt, h["tf"] + dt, h["piece"], ("hoop", "head")))
        elif tech == "cross_stick":
            out[h["hand"]].append((h["tf"] - 0.25, h["tf"] + 0.25, h["piece"], ("hoop", "head")))
    for c in chokes:
        out[c["hand"]].append((c["t0"], c["t1"], c["piece"], ("bow", "bell")))
    return out


def _under_bow(kit, p, pts):
    """How far points sit below the top of cymbal p's bow: 0 outside its rim, above it, or more than UNDER_BAND
    below it (a stick that far under the cymbal is not touching it)."""
    c, n, lift = kit.frame[p]
    (q0, z0), (q1, z1), th = next(args for name, _, args in kit.parts[p] if name == "bow")
    v = pts - c
    d = v @ n - lift
    q = np.linalg.norm(v - (v @ n)[:, None] * n, axis=1)
    top = z0 + (z1 - z0) * np.clip((q - q0) / (q1 - q0), 0, 1) + th
    return np.where((q < q1) & (d < top - TIP_SINK) & (d > top - UNDER_BAND), top - d, 0.0)


def _body_obstacles(fr):
    """Capsules of the drummer's own body that sticks and fists must stay out of: [(name, a, b, r0, r1)]."""
    b = fr["body"]
    out = [("head", np.array(b["head"]), np.array(b["head"]), HEAD_R, HEAD_R),
           ("torso", np.array(b["hips"]) + np.array([0, 0, 0.12]), np.array(b["chest"]), TORSO_R, TORSO_R)]
    for side, fd in fr["feet"].items():
        out.append((f"{side}.thigh", np.array(fd["hip"]), np.array(fd["knee"]), *THIGH_R))
    return out


def _seg_contact(a1, b1, r1, a2, b2, r2):
    """Overlap of two tapered capsules: (depth, point on the first, unit normal pushing the first out)."""
    s, u, dist, c1, c2 = seg_seg(a1, b1, a2, b2)
    depth = r1[0] + s * (r1[1] - r1[0]) + r2[0] + u * (r2[1] - r2[0]) - dist
    n = (c1 - c2) / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0])
    return depth, c1, n, s


def _arm_contacts(fr, parts, tol):
    """The two hands' fists and forearms against each other (a hand crossed over the other), each pair once. A
    contact is told from the side whose point is nearer its wrist: the resolver pushes hands, and leaves an
    overlap near an elbow to place_elbows."""
    arms = {h: [(p, a, b, r0, r1) for p, a, b, r0, r1, _ in body_parts(fr["hands"][h], h) if p in ("fist", "fore")]
            for h in ("R", "L")}
    out = []
    for pr, ar, br, r0, r1 in arms["R"]:
        for pl, al, bl, l0, l1 in arms["L"]:
            s, u, dist, cr, cl = seg_seg(ar, br, al, bl)
            depth = r0 + s * (r1 - r0) + l0 + u * (l1 - l0) - dist
            if depth <= tol or (pr not in parts and pl not in parts):
                continue
            n = (cr - cl) / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0])
            # "fist" capsules count as all wrist end
            sr, sl = (1.0 if pr == "fist" else s), (1.0 if pl == "fist" else u)
            if (sr >= sl and pr in parts) or pl not in parts:
                c = {"hand": "R", "part": pr, "other": f"L.{pl}", "point": cr, "s": float(s), "normal": n}
            else:
                c = {"hand": "L", "part": pl, "other": f"R.{pr}", "point": cl, "s": float(u), "normal": -n}
            c["depth"] = float(depth)
            out.append(c)
    return out


def contacts(kit, fr, allow=None, tol=TOL, parts=("stick", "fist", "fore", "upper"), normals=False):
    """Clips in one frame (kit already posed): [{"hand", "part", "other", "depth", "point", "s", "normal"?}].
    s is where along the part (0 at its base: butt, elbow, shoulder) the overlap is deepest. The normal (when
    asked for) pushes the part out of the obstacle."""
    allow = allow or {}
    t = fr["t"]
    out = []
    obst = _body_obstacles(fr)
    for hand, hd in fr["hands"].items():
        playing = hd.get("piece")
        win = [(p, ps) for t0, t1, p, ps in allow.get(hand, ()) if t0 <= t <= t1]
        for part, a, b, r0, r1, n in body_parts(hd, hand):
            if part not in parts:
                continue
            pts, rad, s = samples(a, b, r0, r1, n)
            for key, dist in kit.sd(pts).items():
                depth = rad - dist
                under = None
                if part == "stick" and key[0] == playing and key[1] in ("head", "bow", "bell", "hats"):
                    # the tip sits on what it plays (its centre may sink TIP_SINK); the rest of the stick may not
                    depth = np.where(s > 0.93, -dist - TIP_SINK, depth)
                    if key[1] == "bow":     # ... but not from underneath, where a swinging cymbal can catch it
                        under = _under_bow(kit, key[0], pts) * (s > 0.93)
                        depth = np.maximum(depth, under)
                if any(key[0] == p and key[1] in ps for p, ps in win):
                    if part in ("stick", "fist"):
                        continue
                    if part == "fore":      # a pinching hand has its wrist at the edge
                        depth = np.where(s > 0.8, -1.0, depth)
                j = int(np.argmax(depth))
                if depth[j] > tol:
                    c = {"hand": hand, "part": part, "other": f"{key[0]}.{key[1]}", "depth": float(depth[j]),
                         "point": pts[j], "s": float(s[j])}
                    if normals and under is not None and under[j] >= depth[j]:
                        c["normal"] = kit.frame[key[0]][1]      # back up through to the top
                    elif normals:
                        g = np.zeros(3)
                        for ax in range(3):
                            dp = np.zeros(3); dp[ax] = 1e-4
                            g[ax] = kit._one(pts[j] + dp, key) - kit._one(pts[j] - dp, key)
                        c["normal"] = _unit(g)
                    out.append(c)
            if part in ("stick", "fist"):
                for name, oa, ob, o0, o1 in obst:
                    depth, pt, nrm, s0 = _seg_contact(a, b, (r0, r1), oa, ob, (o0, o1))
                    if depth > tol:
                        out.append({"hand": hand, "part": part, "other": name, "depth": float(depth), "point": pt,
                                    "s": float(s0), "normal": nrm})
    out += _arm_contacts(fr, parts, tol)
    # sticks against each other, and each stick against the other hand's fist and forearm
    if "stick" not in parts:
        return out
    R, L = fr["hands"]["R"], fr["hands"]["L"]
    for A, B, na, nb in ((R, L, "R", "L"), (L, R, "L", "R")):
        for part, a, b, r0, r1, n in body_parts(B, nb):
            if part == "upper" or (part == "stick" and na == "L"):
                continue
            depth, pt, nrm, s0 = _seg_contact(np.array(A["butt"]), np.array(A["tip"]), STICK_R, a, b, (r0, r1))
            if depth > tol:
                out.append({"hand": na, "part": "stick", "other": f"{nb}.{part}", "depth": float(depth),
                            "point": pt, "s": float(s0), "normal": nrm})
    return out


def report(anim, tol=TOL, worst=8):
    """Clip summary of a whole animation: frames with any clip, per pair counts and worst depths."""
    kit = Kit(anim["kit"], anim.get("hihat_open", 0.018))
    allow = allowed_windows(anim["hits"], anim.get("chokes", ()), anim["fps"])
    pairs, bad, rows = {}, 0, []
    for j, fr in enumerate(anim["frames"]):
        cl = contacts(kit.pose(fr), fr, allow, tol)
        if cl:
            bad += 1
        for c in cl:
            k = f"{c['hand']}.{c['part']} x {c['other']}"
            n, mx, at = pairs.get(k, (0, 0.0, 0))
            pairs[k] = (n + 1, max(mx, c["depth"]), j if c["depth"] > mx else at)
            rows.append((c["depth"], j, k))
    rows.sort(reverse=True)
    return {"frames": len(anim["frames"]), "frames_clipping": bad,
            "pairs": {k: {"frames": n, "max_mm": round(1000 * mx, 1), "worst_frame": at}
                      for k, (n, mx, at) in sorted(pairs.items(), key=lambda x: -x[1][0])},
            "worst": [{"frame": j, "pair": k, "mm": round(1000 * d, 1)} for d, j, k in rows[:worst]]}


# ---------------------------------------------------------------- resolving clips
MARGIN = 0.003          # m of clearance a push leaves
OVER = 1.3              # pushes overshoot a little: the spread and the moving contact normals undershoot
BODY = ("head", "torso")
POLISH = 3              # last iterations spread over a third of SPREAD_S, to clear what is left locally
SPREAD_S = 0.08         # s: a correction eases in and out over about this long, so hands move around obstacles
SWIVEL = tuple(math.radians(a) for a in range(0, 360, 5))      # elbow candidates about the shoulder-wrist axis
SWIVEL_MOVE = 160.0     # cost per (rad per 1/24 s)^2 of turning the swivel: the elbow drifts, it doesn't follow
                        # strokes (high enough that the 5-degree candidate steps don't read as elbow pops)
ELBOW_UP = 0.05         # m below the shoulder an elbow rises to before it costs (3 per m)
ELBOW_SMOOTH_S = 0.05   # s: the elbow's swivel is smoothed over about this long


def spread(r, fps, s=SPREAD_S):
    """Each frame takes the largest nearby correction, faded by distance, then a light blur: a one-frame push
    becomes a smooth reach around the obstacle, starting before it."""
    r = np.asarray(r, float)
    vec = r.ndim == 2
    sig = max(s * fps, 1e-3)
    W = int(math.ceil(2.5 * sig))
    out = r.copy()
    mag = np.linalg.norm(out, axis=1) if vec else np.abs(out)
    for tau in range(-W, W + 1):
        if tau == 0:
            continue
        f = math.exp(-0.5 * (tau / sig) ** 2)
        sh = np.roll(r, tau, axis=0) * f
        if tau > 0:
            sh[:tau] = 0
        else:
            sh[tau:] = 0
        m = np.linalg.norm(sh, axis=1) if vec else np.abs(sh)
        k = m > mag
        out[k], mag[k] = sh[k], m[k]
    k = np.exp(-0.5 * (np.arange(-W, W + 1) / (0.4 * sig)) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(out[:1], W, 0), out, np.repeat(out[-1:], W, 0)])
    if vec:
        return np.stack([np.convolve(pad[:, i], k, "valid") for i in range(out.shape[1])], 1)
    return np.convolve(pad, k, "valid")


def _set_chain(hd, base, v, stick, tip_len):
    """Moves one hand's chain by v from its unresolved pose; a pinned tip (base['pin'] 1) stays put and the stick
    pivots about it."""
    c = base["pin"]
    grip = base["grip"] + v
    tip = base["tip"] + (1 - c) * v
    d = _unit(tip - grip)
    if c > 0:
        grip = tip - tip_len * d
    else:
        tip = grip + tip_len * d
    hd["tip"], hd["grip"] = tip, grip
    hd["butt"] = grip - (stick - tip_len) * d
    hd["wrist"] = drum_hands.wrist(grip, d, hd["side"])


def _swivel_refs(sh, wr):
    """(N, 3) shoulders and wrists -> each frame's swivel reference, square to the shoulder-wrist axis: the
    world's down at the first frame, then carried along by the least turn that follows the axis. A reference
    fixed in the world spins about an arm that points along it (down: a wrist straight under its shoulder),
    and would swing the elbow round with it."""
    U = np.asarray(wr, float) - np.asarray(sh, float)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    out = np.zeros_like(U)
    r = np.array([0.0, 0.0, -1.0])
    for j, u in enumerate(U):
        r = r - (r @ u) * u
        if r @ r < 1e-8:
            r = np.array([0.0, -1.0, 0.0]) - u[1] * u
        r = out[j] = _unit(r)
    return out


def _elbow_circle(sh, el, wr, ref):
    """The circle an elbow swings on about the shoulder-wrist axis (through el): centre, radius, and in-plane axes
    e1 (ref, square to the axis: _swivel_refs) and e2."""
    u = _unit(wr - sh)
    c = sh + float((el - sh) @ u) * u
    e1 = _unit(ref - (ref @ u) * u)
    return c, float(np.linalg.norm(el - c)), e1, np.cross(u, e1)


def resolve(frames, kit_dict, hihat_open, allow, fps, stick, tip_len, arm_len, arm_ik, iters=10):
    """Removes clips in place. frames: animate()'s unrounded frames, each hand with "pin" (0..1, how pinned the
    tip is to its strike point) and "side" (+1 R, -1 L). Three stages: (1) push each hand's chain (grip, wrist,
    stick) out of the kit, the body and the other hand, spread over neighbouring frames; (2) place_elbows;
    (3) POLISH more pushes with the elbows kept on their swivel, for what the moved forearms now touch.
    Returns per hand the mean and max push (m)."""
    kit = Kit(kit_dict, hihat_open)
    N = len(frames)
    base = [{h: {k: np.array(fr["hands"][h][k], float) for k in ("tip", "grip")} | {"pin": fr["hands"][h]["pin"]}
             for h in ("R", "L")} for fr in frames]
    C = {h: np.zeros((N, 3)) for h in ("R", "L")}
    swivel = {}

    def apply(j):
        fr = frames[j]
        for h in ("R", "L"):
            hd = fr["hands"][h]
            _set_chain(hd, base[j][h], C[h][j], stick, tip_len)
            sh = np.asarray(hd["shoulder"], float)
            v = hd["wrist"] - sh
            over = float(np.linalg.norm(v)) - (arm_len - 0.002)
            if over > 0:
                shift = over * _unit(v)
                for key in ("tip", "grip", "butt", "wrist"):
                    hd[key] = hd[key] - shift
            el = arm_ik(sh, hd["wrist"], hd["side"], drum_hands.forward(hd["tip"] - hd["butt"], hd["side"]))
            if swivel:
                th, refs = swivel[h]
                c, r, e1, e2 = _elbow_circle(sh, el, hd["wrist"], refs[j])
                el = c + r * (math.cos(th[j]) * e1 + math.sin(th[j]) * e2)
            hd["elbow"] = el

    def push(iters, polish):
        for it in range(iters):
            res = {h: np.zeros((N, 3)) for h in ("R", "L")}
            worst = 0.0
            for j, fr in enumerate(frames):
                apply(j)
                kit.pose(fr)
                for c in contacts(kit, fr, allow, tol=0.0, parts=("stick", "fist", "fore"), normals=True):
                    if c["part"] == "fore" and c["s"] < 0.75:
                        continue          # the elbow end is the swivel's job
                    nrm = c["normal"]
                    if c["other"] in BODY:
                        # hands work in front of the body: a radial push off the head or chest would lift them
                        flat = np.array([nrm[0], nrm[1], 0.0])
                        nrm = _unit(flat) if flat @ flat > 0.04 else np.array([0.0, 1.0, 0.0])   # +y: toward the kit
                    mv = OVER * (c["depth"] + MARGIN) * nrm
                    oh, _, op = c["other"].partition(".")
                    if oh in ("R", "L") and op in ("stick", "fist", "fore"):
                        # both hands give way: the upper stick more, and a hand half of a stick in its fist or
                        # arm; a hand with its tip on a drum less (pushed, it could only drag the tip off)
                        mine = (0.7 if nrm[2] >= 0 else 0.3) if op == "stick" else 0.5
                        give = (1 - mine) * (1 - base[j][oh]["pin"])
                        res[oh][j] -= mv * give
                        mv = mv * (1 - give)
                    cur = res[c["hand"]][j]
                    if mv @ mv > cur @ cur:
                        res[c["hand"]][j] = mv
                    worst = max(worst, c["depth"])
            if worst < TOL:
                break
            s = SPREAD_S if it < iters - polish else SPREAD_S / 3
            for h in ("R", "L"):
                C[h] += spread(res[h], fps, s)
        for j in range(N):
            apply(j)

    push(iters, POLISH)
    swivel.update(place_elbows(frames, kit, fps))
    push(POLISH, POLISH)
    return {h: {"mean_mm": round(1000 * float(np.linalg.norm(C[h], axis=1).mean()), 2),
                "max_mm": round(1000 * float(np.linalg.norm(C[h], axis=1).max()), 1)} for h in ("R", "L")}


def swivel_path(cost, move):
    """Viterbi: the candidate per frame that minimises the summed frame costs (N, K) plus move (K, K) between
    consecutive frames' candidates. Returns (N,) indices."""
    N, K = cost.shape
    acc = cost[0].copy()
    back = np.zeros((N, K), int)
    cols = np.arange(K)
    for j in range(1, N):
        tot = acc[:, None] + move
        back[j] = np.argmin(tot, axis=0)
        acc = tot[back[j], cols] + cost[j]
    path = np.zeros(N, int)
    path[-1] = int(np.argmin(acc))
    for j in range(N - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    return path


def place_elbows(frames, kit, fps):
    """Each elbow, around its shoulder-wrist axis, where it keeps the arm out of the kit (a Kit), the body and
    the other hand's stick, fist and forearm, and is least awkward (drum_hands.arm_cost: the wrist's bend, the elbow's
    hang), over the whole take at once: turning the swivel costs SWIVEL_MOVE per squared rate, so the elbow
    drifts to where the strokes average out instead of swinging with each one, and moves early and smoothly
    out of the way of what is coming. The angle is measured from _swivel_refs, so it doesn't drag the
    elbow when the wrist moves. Returns each hand's (angles (N,) rad, references (N, 3))."""
    N = len(frames)
    sig = ELBOW_SMOOTH_S * fps
    W = int(math.ceil(2.5 * sig))
    ker = np.exp(-0.5 * (np.arange(-W, W + 1) / sig) ** 2)
    ker /= ker.sum()
    A = np.array(SWIVEL)
    K, n = len(A), 8
    s = np.linspace(0, 1, n)
    rad = np.tile(np.concatenate([r0 + s * (r1 - r0) for r0, r1 in (ARM_R["upper"], ARM_R["fore"])]), K)
    far = np.tile(np.concatenate([s, s]) > 0.2, K)        # the shoulder end hangs off the torso: not a clip
    dA = (A[:, None] - A[None, :] + math.pi) % (2 * math.pi) - math.pi
    move = SWIVEL_MOVE * (fps / 24.0) ** 2 * dA ** 2
    out = {}
    for h, oh in (("R", "L"), ("L", "R")):
        cost = np.zeros((N, K))
        circ = []
        refs = _swivel_refs(*(np.array([fr["hands"][h][k] for fr in frames], float) for k in ("shoulder", "wrist")))
        for j, fr in enumerate(frames):
            hd = fr["hands"][h]
            kit.pose(fr)
            sh, el0, wr = (np.asarray(hd[k], float) for k in ("shoulder", "elbow", "wrist"))
            c, r, e1, e2 = _elbow_circle(sh, el0, wr, refs[j])
            circ.append((c, r, e1, e2))
            # hand_dir: the empty hand's line while its stick is in the air
            d = np.asarray(hd["hand_dir"] if "hand_dir" in hd else np.asarray(hd["tip"]) - np.asarray(hd["butt"]))
            el = c + r * (np.cos(A)[:, None] * e1 + np.sin(A)[:, None] * e2)
            pts = np.concatenate([sh + s[None, :, None] * (el - sh)[:, None],
                                  el[:, None] + s[None, :, None] * (wr - el)[:, None]], 1).reshape(-1, 3)
            sd = np.min(np.stack(list(kit.sd(pts).values())), axis=0)
            for name, a, b, r0, _ in _body_obstacles(fr):
                if name in BODY:
                    sd = np.minimum(sd, np.where(far, _sd_capsule_pts(pts, a, b, r0), 1.0))
            for part, a, b, r0, r1, _ in body_parts(fr["hands"][oh], oh):
                if part in ("stick", "fist", "fore"):
                    sd = np.minimum(sd, _sd_capsule_pts(pts, a, b, r0, r1))
            tot = np.clip(rad - sd, 0, None).reshape(K, 2, n).max(2).sum(1)
            cost[j] = (1000 * tot + drum_hands.arm_cost(d, h, sh, el, wr)
                       + 3 * np.maximum(0.0, el[:, 2] - sh[2] + ELBOW_UP))
        th = np.unwrap(A[swivel_path(cost, move)])
        th = np.convolve(np.concatenate([np.repeat(th[:1], W), th, np.repeat(th[-1:], W)]), ker, "valid")
        for j, fr in enumerate(frames):
            c, r, e1, e2 = circ[j]
            fr["hands"][h]["elbow"] = c + r * (math.cos(th[j]) * e1 + math.sin(th[j]) * e2)
        out[h] = (th, refs)
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(report(json.load(open(sys.argv[1]))), indent=1))

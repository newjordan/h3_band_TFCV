"""Drum kits as data: a kit spec (pieces by type and size, placed by hand or by rule) -> the kit dict that drums.py
animates, drum_collide.py turns into hitboxes and blender_drums.py builds.

Spec (JSON):
  {"name": "big", "double_pedal": true,
   "pieces": [{"type": "tom", "size": 8}, {"type": "crash", "size": 19, "id": "crash3"},
              {"type": "ride", "size": 22, "at": [0.66, 0.36, 0.94], "tilt": 10}, ...]}

  type   kick, snare, hihat, tom (rack), floor, crash, splash, china, ride
  size   diameter in inches (default per type, TYPES)
  id     defaults to the type, numbered from the second one on (tom pieces are always numbered: tom1, tom2 ...).
         Ids start with their type, which is how the rest of the rig tells what a piece is (piece_type).
  at     [x, y, z] centre of the playing surface (kit metres: x to the drummer's right, y toward the audience,
         z up); tilt (deg toward the drummer) goes with it. Pieces without "at" are laid out by the rules below.
  stand  [x, y] floor spot of a cymbal's boom stand. Otherwise the stand stands straight under it; a
         rule-placed cymbal gets a boom by rule when a straight stand would go through the kit (one placed with
         "at" keeps its straight stand).

Layout rules, around the seat (O, the drummer's hips seen from above) at azimuth theta (0 straight ahead,
positive to the drummer's right) and distance d:
  kick, snare, hi-hat   where the standard kit has them (one each; a double pedal is "double_pedal": true).
  rack toms             smallest to largest, left to right, on an arc 0.81 m out, centred 1 deg right (3.5 deg
                        further per tom past two), 3 cm between rims: two toms land where the standard kit's do.
  floor toms            from 48 deg outward on an arc 0.60 m out, 3 cm between rims, heads at 0.57 m.
  cymbals               first crash, second crash and ride at the standard kit's spots; further cymbals take
                        SLOTS in order of preference for their type (centre above the toms, far left, right of
                        the ride, low left for a splash).
Then every rule-placed piece that is not exactly a standard piece is nudged (azimuth, distance, height) until
it clears the rest of the kit and its hardware (drum_collide.py hitboxes), the air above every head, the
column a stroke rises through over each drum's strike point (no cymbal hangs there) and the path the stick
takes to it, and pulled in until the stick's tip reaches its strike point from a shoulder. If
that leaves the hi-hat, snare or ride clashing, they give way a little too (the ride further if it has to).
check() reports what is left.

    python -m blockout.drum_kit big [--out kit.json]       # build a preset or a spec file, print the check
"""
import json, math, re

import numpy as np

from . import drum_collide, drum_hands


def _toward(deg):
    """Head normal tilted toward the drummer (-y) by deg."""
    a = math.radians(deg)
    return (0.0, -math.sin(a), math.cos(a))


# c: centre of the playing surface, r: radius, n: surface normal, depth: shell depth,
# strike: how far from the centre toward the drummer the stick lands (fraction of r),
# pitch: stick angle below horizontal at contact.
STANDARD = {
    "kick":   {"kind": "kick", "c": (0.06, 0.41, 0.28), "r": 0.28, "n": (0, -1, 0), "depth": 0.40},
    "snare":  {"kind": "drum", "c": (-0.08, 0.22, 0.68), "r": 0.178, "n": _toward(6), "depth": 0.13,
               "strike": 0.30, "pitch": 14},
    "tom1":   {"kind": "drum", "c": (-0.14, 0.49, 0.84), "r": 0.127, "n": _toward(22), "depth": 0.20,
               "strike": 0.30, "pitch": 20},
    "tom2":   {"kind": "drum", "c": (0.17, 0.50, 0.84), "r": 0.152, "n": _toward(22), "depth": 0.22,
               "strike": 0.30, "pitch": 20},
    "floor":  {"kind": "drum", "c": (0.45, 0.10, 0.63), "r": 0.20, "n": _toward(3), "depth": 0.40,
               "strike": 0.35, "pitch": 16},
    "hihat":  {"kind": "hihat", "c": (-0.40, 0.37, 0.92), "r": 0.178, "n": (0, 0, 1), "strike": 0.55, "pitch": 5},
    "crash":  {"kind": "cymbal", "c": (-0.42, 0.60, 1.22), "r": 0.23, "n": _toward(14), "strike": 0.85, "pitch": 2},
    "crash2": {"kind": "cymbal", "c": (0.42, 0.66, 1.25), "r": 0.23, "n": _toward(14), "strike": 0.85, "pitch": 2},
    "ride":   {"kind": "cymbal", "c": (0.66, 0.36, 0.94), "r": 0.26, "n": _toward(10), "strike": 0.55, "pitch": 5},
}

# per type: kind, default size (in), shell depth (m, or per inch for toms), strike, pitch, tilt (deg)
TYPES = {
    "kick":   {"kind": "kick", "size": 22, "depth": 0.40},
    "snare":  {"kind": "drum", "size": 14, "depth": 0.13, "strike": 0.30, "pitch": 14, "tilt": 6},
    "tom":    {"kind": "drum", "size": 12, "strike": 0.30, "pitch": 20, "tilt": 22},
    "floor":  {"kind": "drum", "size": 16, "strike": 0.35, "pitch": 16, "tilt": 3},
    "hihat":  {"kind": "hihat", "size": 14, "strike": 0.55, "pitch": 5, "tilt": 0},
    "crash":  {"kind": "cymbal", "size": 18, "strike": 0.85, "pitch": 2, "tilt": 14},
    "splash": {"kind": "cymbal", "size": 10, "strike": 0.80, "pitch": 4, "tilt": 14},
    "china":  {"kind": "cymbal", "size": 18, "strike": 0.85, "pitch": 8, "tilt": 10},   # steeper: over the lip
    "ride":   {"kind": "cymbal", "size": 21, "strike": 0.55, "pitch": 5, "tilt": 10},
}
CYMBALS = ("crash", "splash", "china", "ride")
STANDARD_ORDER = ("kick", "snare", "tom", "floor", "hihat", "crash", "splash", "china", "ride")   # kit dict order

SEAT = np.array([0.0, -0.30])           # layouts centre here; drums.HIPS sits a little behind it
SHOULDERS = (np.array([0.19, -0.25, 1.10]), np.array([-0.19, -0.25, 1.10]))
# the drummer's legs at rest as drums.py poses them on the standard pedals (hip, knee, ankle; the right leg also
# with its heel lifted for a kick): rule-placed pieces keep LEG_CLEAR off them, so the knees have room to swivel
LEGS = (((0.12, -0.32, 0.60), (0.173, 0.117, 0.595), (0.08, 0.119, 0.185)),
        ((0.12, -0.32, 0.60), (0.18, 0.114, 0.643), (0.08, 0.119, 0.235)),
        ((-0.12, -0.32, 0.60), (-0.329, 0.064, 0.548), (-0.347, 0.139, 0.135)))
LEG_CLEAR = 0.005
RIM_CLEAR = 0.004                       # m two pieces' rims keep apart (real kits sit this tight)
TIP_REACH = 0.80                        # m from a shoulder to a strike point the stick's tip can reach
CHECK_TOL = 0.005                       # m of summed overlap check() lets pass (a few mm of stick airspace)
TOM_ARC, TOM_MID, TOM_SHIFT, FLOOR_ARC, FLOOR_FROM, GAP = 0.81, 1.0, 3.5, 0.60, 48.0, 0.03
TOM_MAX, FLOOR_MAX = 75.0, 100.0        # deg from straight ahead the rule layout goes; past it a drum is beside
                                        # or behind the drummer
FLOOR_Z = 0.57                          # m: floor tom heads stay at least this high, about the top of the right
                                        # thigh, so a left hand crossing over to play one stays above the leg
# (theta deg, d m, z m) for cymbals after the first two crashes and the ride, by type preference
SLOTS = {"centre": (0.0, 1.0, 1.34), "far_left": (-50.0, 0.92, 1.22), "right_high": (58.0, 0.95, 1.30),
         "low_left": (-40.0, 0.82, 1.10)}
PREFER = {"crash": ("centre", "far_left", "right_high", "low_left"),
          "china": ("right_high", "far_left", "centre", "low_left"),
          "splash": ("low_left", "centre", "far_left", "right_high"),
          "ride": ("right_high", "centre", "far_left", "low_left")}

_FIVE = [{"type": "kick"}, {"type": "snare"}, {"type": "hihat"}, {"type": "tom", "size": 10},
         {"type": "tom", "size": 12}, {"type": "floor"}, {"type": "crash"}, {"type": "crash"}, {"type": "ride"}]
PRESETS = {
    "standard": {"pieces": _FIVE},
    "double_pedal": {"double_pedal": True, "pieces": _FIVE},
    "big": {"double_pedal": True,
            "pieces": [{"type": "kick"}, {"type": "snare"}, {"type": "hihat"},
                       {"type": "tom", "size": 8}, {"type": "tom", "size": 10}, {"type": "tom", "size": 12},
                       {"type": "tom", "size": 13}, {"type": "floor", "size": 14}, {"type": "floor", "size": 16},
                       {"type": "crash", "size": 18}, {"type": "crash", "size": 19}, {"type": "crash", "size": 20},
                       {"type": "splash", "size": 10}, {"type": "china", "size": 18}, {"type": "ride", "size": 21}]},
}


def piece_type(p):
    """Type of a piece id: 'tom3' -> 'tom', 'crash2' -> 'crash', 'hihat_pedal' -> 'hihat'."""
    return p.split("_")[0].rstrip("0123456789")


def strike_point(k, hips, p=None):
    """Where the tip lands on kit piece k (id p): from the centre toward the drummer, in the surface plane (on a
    china, on its upturned bow)."""
    c, n = np.array(k["c"], float), _unit(k["n"])
    t = np.array([hips[0] - c[0], hips[1] - c[1], 0.0])
    u = _unit(t - (t @ n) * n)
    return c + k["strike"] * k["r"] * u + surface_lift(p, k) * n


def surface_lift(p, k):
    """How far piece p's playing surface sits above its plane at the strike radius: a china's bow rises toward
    its edge (drum_collide.piece_parts); everything else is flat enough to say 0."""
    if p is None or piece_type(p) != "china":
        return 0.0
    q = (k["strike"] * k["r"] - 0.03) / (k["r"] - 0.03)
    return drum_collide.CHINA_EDGE * float(np.clip(q, 0, 1)) + 0.0015


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _ids(pieces):
    for pc in pieces:
        if pc["type"] not in TYPES:
            raise ValueError(f"unknown piece type {pc['type']!r}; types: {', '.join(TYPES)}")
    taken = {pc["id"] for pc in pieces if pc.get("id")}
    out, seen = [], {}
    for pc in pieces:
        t = pc["type"]
        pid = pc.get("id")
        while not pid:      # the next free name: tom1, tom2 ... / crash, crash2 ...
            seen[t] = seen.get(t, 0) + 1
            cand = f"tom{seen[t]}" if t == "tom" else t if seen[t] == 1 else f"{t}{seen[t]}"
            pid = cand if cand not in taken else None
        taken.add(pid)
        if piece_type(pid) != t:
            raise ValueError(f"piece id {pid!r} must start with its type {t!r}")
        out.append({**pc, "id": pid, "size": float(pc.get("size", TYPES[t]["size"]))})
    ids = [pc["id"] for pc in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate piece ids in {ids}")
    if sum(pc["type"] == "kick" for pc in out) > 1:
        raise ValueError("one kick per kit: a double bass drum is 'double_pedal': true")
    for t in ("kick", "snare", "hihat"):
        if not any(pc["id"] == t for pc in out):
            raise ValueError(f"a kit needs a {t} (id {t!r})")
        if sum(pc["type"] == t and "at" not in pc for pc in out) > 1:
            raise ValueError(f"more than one {t}: place the extra ones with 'at'")
    return out


def _piece(t, size, c, tilt=None):
    d = TYPES[t]
    r = round(size * 0.0127, 4)
    tilt = d.get("tilt", 0) if tilt is None else tilt
    k = {"kind": d["kind"], "c": tuple(round(float(x), 4) for x in c), "r": r}
    if t == "kick":
        k.update(n=(0, -1, 0), depth=d["depth"])
        return k
    k["n"] = _toward(tilt) if tilt else (0, 0, 1)
    if d["kind"] == "drum":
        k["depth"] = d.get("depth") or round((0.10 + 0.01 * size) if t == "tom" else 0.025 * size, 3)
    k.update(strike=d["strike"], pitch=d["pitch"])
    return k


def _polar(theta, d, z):
    a = math.radians(theta)
    return (SEAT[0] + d * math.sin(a), SEAT[1] + d * math.cos(a), z)


def _arc(sizes, radius, start=None, mid=None):
    """Azimuths (deg) for discs of these sizes along an arc, GAP between rims, from `start` or centred on `mid`."""
    rs = [s * 0.0127 for s in sizes]
    th = [0.0]
    for a, b in zip(rs, rs[1:]):
        th.append(th[-1] + math.degrees((a + b + GAP) / radius))
    off = start if start is not None else mid - (th[0] + th[-1]) / 2
    return [t + off for t in th]


def build(spec):
    """Kit dict from a spec (or a preset name). The standard pieces in their standard sizes are exactly STANDARD."""
    if isinstance(spec, str):
        spec = PRESETS[spec]
    pieces = _ids(spec["pieces"])
    kit, auto, polar = {}, [], {}
    by = {t: [pc for pc in pieces if pc["type"] == t and "at" not in pc] for t in TYPES}
    for pc in pieces:
        if "at" in pc:
            kit[pc["id"]] = _piece(pc["type"], pc["size"], pc["at"], pc.get("tilt"))
            if "stand" in pc and kit[pc["id"]]["kind"] == "cymbal":
                kit[pc["id"]]["stand"] = tuple(float(v) for v in pc["stand"][:2])
    std = {"kick": "kick", "snare": "snare", "hihat": "hihat", "ride": "ride"}
    for t, sid in std.items():
        for pc in by[t][:1]:
            k = dict(STANDARD[sid])
            if pc["size"] != TYPES[t]["size"]:
                k = _piece(t, pc["size"], STANDARD[sid]["c"][:2] + ((pc["size"] * 0.0127,) if t == "kick" else
                                                                     STANDARD[sid]["c"][2:]))
            kit[pc["id"]] = k
    toms = sorted(by["tom"], key=lambda pc: pc["size"])
    mid = TOM_MID + TOM_SHIFT * max(0, len(toms) - 2)      # extra toms push the arc right, off the hi-hat
    for pc, th in zip(toms, _arc([pc["size"] for pc in toms], TOM_ARC, mid=mid)):
        polar[pc["id"]] = (th, TOM_ARC, 0.84)
    floors = sorted(by["floor"], key=lambda pc: pc["size"])
    for pc, th in zip(floors, _arc([pc["size"] for pc in floors], FLOOR_ARC, start=FLOOR_FROM)):
        polar[pc["id"]] = (th, FLOOR_ARC, FLOOR_Z)
    for p, (th, _, _) in polar.items():
        lim = TOM_MAX if piece_type(p) == "tom" else FLOOR_MAX
        if abs(th) > lim:
            raise ValueError(f"too many {piece_type(p)}s to lay out by rule: {p} would sit at {th:.0f} deg, beside "
                             f"or behind the seat; place it with 'at'")
    crashes = by["crash"]
    for pc, sid in zip(crashes[:2], ("crash", "crash2")):
        if pc["size"] == TYPES["crash"]["size"]:
            kit[pc["id"]] = dict(STANDARD[sid])
        else:       # the standard spot, then nudged like any rule-placed piece
            kit[pc["id"]] = _piece("crash", pc["size"], STANDARD[sid]["c"])
            auto.append(pc["id"])
    for pc in by["ride"][:1]:
        if pc["size"] != TYPES["ride"]["size"]:
            auto.append(pc["id"])
    free = list(SLOTS)
    for pc in crashes[2:] + by["splash"] + by["china"] + by["ride"][1:]:
        slot = next(s for s in PREFER[pc["type"]] if s in free) if free else None
        if slot is None:
            raise ValueError("too many cymbals to lay out by rule: place the rest with 'at'")
        free.remove(slot)
        polar[pc["id"]] = SLOTS[slot]
    for pc in pieces:
        if pc["id"] in polar:
            th, d, z = polar[pc["id"]]
            kit[pc["id"]] = _piece(pc["type"], pc["size"], _polar(th, d, z))
            auto.append(pc["id"])
    for p in auto:      # where the rule put a standard piece in its standard size, use the standard numbers
        if p in STANDARD and np.allclose(kit[p]["c"], STANDARD[p]["c"], atol=0.012) and \
                abs(kit[p]["r"] - STANDARD[p]["r"]) < 0.004:
            kit[p] = dict(STANDARD[p])
    movable = [p for p in auto if kit[p] != STANDARD.get(p)]
    rule = {p: kit[p]["c"] for p in movable}
    placed = {pc["id"] for pc in pieces if "at" in pc}
    booms = [p for p in kit if kit[p]["kind"] == "cymbal" and p not in placed]
    for p in booms:
        _place_stand(kit, p)
    _settle(kit, movable)
    ride = next((pc["id"] for pc in by["ride"][:1]), None)
    give = [p for p in ("hihat", "snare", ride) if p and p not in placed and clash(kit, p) > CHECK_TOL]
    if give:        # the hi-hat, snare and ride give way, a little, only for what the rest could not clear
        movable += give
        _settle(kit, movable, scales=(1,))
    stuck = [p for p in movable if p not in give and clash(kit, p) > CHECK_TOL]
    if ride in give and clash(kit, ride) > CHECK_TOL:
        stuck.insert(0, ride)       # a ride on its boom can move further than the hi-hat and snare
    for p in stuck:
        _relocate(kit, p)
    if stuck:
        _settle(kit, movable, scales=(1,))
    _relax(kit, movable, rule)
    for p in booms:
        _place_stand(kit, p)
    if spec.get("double_pedal"):
        kit["kick"] = dict(kit["kick"], double=True)
    order = list(STANDARD_ORDER)
    return {p: kit[p] for p in sorted(kit, key=lambda p: (order.index(piece_type(p)), p))}


# ---------------------------------------------------------------- clearance
def _rim_points(k, n_pts=16):
    c, n, r = np.array(k["c"], float), _unit(k["n"]), k["r"]
    a = _unit(np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0]))
    b = np.cross(n, a)
    ang = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    ring = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
    if k["kind"] in ("drum", "kick"):
        rr = r + drum_collide.HOOP_W
        return np.concatenate([c + rr * ring, c - k["depth"] * n + rr * ring, [c, c - k["depth"] * n]])
    return np.concatenate([c + r * ring, c + 0.5 * r * ring])


def _airspace(k):
    """Points above a drum head (or a hi-hat's strike side) the stick passes through."""
    c, n, r = np.array(k["c"], float), _unit(k["n"]), k["r"]
    a = _unit(np.cross(n, [1.0, 0, 0]))
    b = np.cross(n, a)
    pts = []
    for h in (0.06, 0.14, 0.22):
        for q in (0.0, 0.45 * r):
            for ang in ((0.0,) if q == 0 else np.linspace(0, 2 * math.pi, 6, endpoint=False)):
                pts.append(c + h * n + q * (math.cos(ang) * a + math.sin(ang) * b))
    return np.array(pts)


STICK_PATH = (0.04, 0.30)      # m along the stick from its tip: the shaft and the fist, as it lands
STICK_CLEAR = 0.012            # m: stick radius and a little air


def _stick_path(p, k):
    """Points along the stick as it lands on piece p, pointing back toward the nearer shoulder."""
    s = strike_point(k, SEAT, p)
    sh = min(SHOULDERS, key=lambda v: float(np.linalg.norm(s - v)))
    d = _unit(sh - s)
    return s + np.linspace(*STICK_PATH, 10)[:, None] * d


CHOKE_ARM = 0.036              # m: the forearm behind a fist choking a cymbal


def _stick_way(p, k):
    """(points, radius around each) the stick takes landing on piece p (_stick_path, radius 0), and on a crash,
    splash or china the hand that chokes it at its near edge (drums.choke_pose, drum_hands.hitbox) and the
    forearm reaching it from the nearer shoulder."""
    path = _stick_path(p, k)
    if piece_type(p) not in ("crash", "splash", "china"):
        return path, np.zeros(len(path))
    return _choke_way(tuple(k["c"]), tuple(k["n"]), k["r"], path)


_CHOKE = {}


def _choke_way(c, n, r, path):
    key = (c, n, r)
    if key not in _CHOKE:
        c, n = np.array(c, float), _unit(n)
        t = np.array([SEAT[0] - c[0], SEAT[1] - c[1], 0.0])
        u = _unit(t - (t @ n) * n)
        g = c + (0.97 * r + 0.02) * u - 0.02 * n
        d = _unit(_unit([u[0], u[1], 0.0]) + np.array([0, 0, -0.5]))
        right = np.linalg.norm(g - SHOULDERS[0]) <= np.linalg.norm(g - SHOULDERS[1])
        sh = SHOULDERS[0 if right else 1]
        pts, rad = [], []
        for a, b, r0, r1 in drum_hands.hitbox(g, d, "R" if right else "L"):
            s = np.linspace(0, 1, 5)
            pts.append(a + s[:, None] * (b - a)), rad.append(r0 + s * (r1 - r0))
        w = drum_hands.wrist(g, d, "R" if right else "L")
        s = np.linspace(0.0, 0.25, 6)
        pts.append(w + s[:, None] * _unit(sh - w)), rad.append(np.full(len(s), CHOKE_ARM))
        _CHOKE[key] = (np.concatenate(pts), np.concatenate(rad))
    pts, rad = _CHOKE[key]
    return np.concatenate([path, pts]), np.concatenate([np.zeros(len(path)), rad])


STRIKE_RISE = np.arange(0.04, 0.301, 0.02)    # m above a drum's strike point its strokes' tip rises through ...
STRIKE_CLEAR = 0.03                            # ... which no cymbal comes within this of


def _strike_column(p, k):
    """Points straight up from drum p's strike point (sampled finer than a cymbal is thin, so none slips
    between them)."""
    return strike_point(k, SEAT, p) + STRIKE_RISE[:, None] * np.array([0.0, 0.0, 1.0])


def _leg_samples(n=8):
    """Points along the LEGS' thighs and shins, and their radii (drum_collide.THIGH_R, SHIN_R)."""
    s = np.linspace(0, 1, n)
    pts, rad = [], []
    for hip, knee, ankle in LEGS:
        for a, b, (r0, r1) in ((hip, knee, drum_collide.THIGH_R), (knee, ankle, drum_collide.SHIN_R)):
            a, b = np.array(a), np.array(b)
            pts.append(a + s[:, None] * (b - a)), rad.append(r0 + s * (r1 - r0))
    return np.concatenate(pts), np.concatenate(rad)


LEG_PTS = _leg_samples()


def _over(sd, clear):
    return sum(float(np.clip(clear - v, 0, None).sum()) for v in sd.values())


def _rods_over(rods, pts, clear):
    return sum(float(np.clip(r + clear - drum_collide._sd_capsule_pts(pts, a, b, 0.0), 0, None).sum())
               for _, a, b, r in rods)


def clash(kit, p, detail=False):
    """How badly piece p overlaps the rest of the kit (m, summed): its rims and shell inside other pieces or their
    hardware, other pieces' rims inside it, anything in the air a stick needs above a head, and a cymbal hanging
    over a drum's strike point. detail: the same as {what: amount}."""
    K = drum_collide.Kit(kit)
    out = {}
    own = lambda name: name == p or name.startswith(p + "_")
    rods = {name: (a, b, r) for name, a, b, r in K.rods}
    for key, dist in K.sd(_rim_points(kit[p])).items():
        if key[0] != p and not own(key[0]) and not (key[1] == "rod" and p == "kick" and key[0].endswith("_mount")):
            out[f"rim in {key[0]}.{key[1]}"] = float(np.clip(RIM_CLEAR - dist, 0, None).sum())
    mine = [x for x in K.rods if own(x[0])]
    others = [x for x in K.rods if not own(x[0]) and not (p == "kick" and x[0].endswith("_mount"))]
    played = lambda q: kit[q]["kind"] in ("drum", "hihat")
    struck = lambda q: kit[q]["kind"] in ("drum", "hihat", "cymbal")
    if played(p):
        air = _airspace(kit[p])
        out["hardware in its air"] = _rods_over(others, air, 0.01)
    way = lambda pts, rad, pieces: _over({key: v - rad for key, v in K.sd(pts, pieces).items()}, STICK_CLEAR)
    if struck(p):
        path, prad = _stick_way(p, kit[p])
        out["hardware in its stick's way"] = _rods_over(others, path, STICK_CLEAR + prad)
    for q in kit:
        if q == p:
            continue
        if played(q):
            out[f"in air over {q}"] = _over(K.sd(_airspace(kit[q]), [p]), 0.01) + \
                _rods_over(mine, _airspace(kit[q]), 0.01)
        if struck(q):
            qpath, qrad = _stick_way(q, kit[q])
            out[f"in the way of {q}'s stick"] = way(qpath, qrad, [p]) + _rods_over(mine, qpath, STICK_CLEAR + qrad)
        if played(p):
            out[f"{q} in its air"] = _over(K.sd(air, [q]), 0.01)
        if struck(p):
            out[f"{q} in its stick's way"] = way(path, prad, [q])
        if kit[p]["kind"] == "drum" and kit[q]["kind"] in ("cymbal", "hihat"):
            out[f"{q} over its strike"] = _over(K.sd(_strike_column(p, kit[p]), [q]), STRIKE_CLEAR)
        if kit[q]["kind"] == "drum" and kit[p]["kind"] in ("cymbal", "hihat"):
            out[f"over {q}'s strike"] = _over(K.sd(_strike_column(q, kit[q]), [p]), STRIKE_CLEAR)
        out[f"{q} rim in it"] = _over(K.sd(_rim_points(kit[q]), [p]), RIM_CLEAR)
    for name, (a, b, r) in rods.items():
        if own(name) or (p == "kick" and name.endswith("_mount")):
            continue
        out[f"{name} through it"] = _over(K.sd(a + np.linspace(0, 1, 12)[:, None] * (b - a), [p]), r + 0.01)
    lp, lr = LEG_PTS
    out["in the drummer's legs"] = _over({k: v - lr for k, v in K.sd(lp, [p]).items()}, LEG_CLEAR) + \
        sum(float(np.clip(r + lr + LEG_CLEAR - drum_collide._sd_capsule_pts(lp, a, b, 0.0), 0, None).sum())
            for _, a, b, r in mine)
    out.update(_stand_clash(kit, p, K))
    out = {k: round(v, 4) for k, v in out.items() if v > 1e-6}
    return out if detail else sum(out.values())


def _stand_clash(kit, p, K=None):
    """{what: m} piece p's own stand (or mount, legs) going through other pieces or crossing other hardware."""
    K = K or drum_collide.Kit(kit)
    own = lambda name: name == p or name.startswith(p + "_")
    mine = [(n, a, b, r) for n, a, b, r in K.rods if own(n)]
    others = [(n, a, b, r) for n, a, b, r in K.rods if not own(n)]
    out = {}
    for name, a, b, r in mine:
        pts = a + np.linspace(0, 1, 16)[:, None] * (b - a)
        for q in kit:
            if q == p or (q == "kick" and name.endswith("_mount")):
                continue
            out[f"{name} through {q}"] = _over(K.sd(pts, [q]), r + 0.01)
        for n2, a2, b2, r2 in others:
            if name.endswith("_mount") and n2.endswith("_mount"):
                continue        # tom mounts meet at the kick's holder
            dist = drum_collide.seg_seg(a, b, a2, b2)[2]
            out[f"{name} crosses {n2}"] = max(0.0, r + r2 + 0.01 - dist)
    return out


def _place_stand(kit, p):
    """A cymbal on a straight stand whose post would go through the kit gets a boom: the post moves out (away
    from the seat, around the cymbal's azimuth) to the nearest spot it stands clear, the arm reaching in."""
    k = {q: v for q, v in kit[p].items() if q != "stand"}
    kit[p] = k
    cost = lambda: sum(_stand_clash(kit, p).values())
    if k["kind"] != "cymbal" or cost() <= 1e-6:
        return
    x, y, _ = k["c"]
    th = math.degrees(math.atan2(x - SEAT[0], y - SEAT[1]))
    d = math.hypot(x - SEAT[0], y - SEAT[1])
    best = (cost(), None)
    for dd in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40):
        for dth in (0, -8, 8, -16, 16, -24, 24, -32, 32):
            sx, sy, _ = _polar(th + dth, d + dd, 0.0)
            if dd == 0 and dth == 0:
                continue
            kit[p] = dict(k, stand=(round(float(sx), 4), round(float(sy), 4)))
            c = cost() + 0.002 * (dd / 0.08 + abs(dth) / 8)
            if c < best[0] - 1e-6:
                best = (c, kit[p])
    kit[p] = best[1] or k


def reach(k, hips=(0.0, -0.30), p=None):
    """Distance from the nearer shoulder to piece k's strike point (on a ride, also to its bell, which lies
    further in: drums.technique_pose)."""
    pts = [strike_point(k, hips)]
    if p is not None and piece_type(p) == "ride":
        c, n = np.array(k["c"], float), _unit(k["n"])
        t = np.array([hips[0] - c[0], hips[1] - c[1], 0.0])
        pts.append(c + 0.035 * _unit(t - (t @ n) * n) + 0.016 * n)
    return max(min(float(np.linalg.norm(s - sh)) for sh in SHOULDERS) for s in pts)


def _too_low(p, z):
    return piece_type(p) == "floor" and z < FLOOR_Z - 1e-6


def _settle(kit, movable, iters=40, scales=(3, 1)):
    """Nudges each rule-placed piece until it clears the kit and its stick can reach it."""
    steps = [(s * dth, s * dd, s * dz) for s in scales for dth in (-3, 0, 3) for dd in (-0.03, 0, 0.03)
             for dz in (-0.03, 0, 0.03) if (dth, dd, dz) != (0, 0, 0)]
    for _ in range(iters):
        moved = False
        for p in movable:
            k = kit[p]
            base = clash(kit, p) + 10 * max(0.0, reach(k, p=p) - TIP_REACH)
            if base <= 1e-6:
                continue
            x, y, z = k["c"]
            th = math.degrees(math.atan2(x - SEAT[0], y - SEAT[1]))
            d = math.hypot(x - SEAT[0], y - SEAT[1])
            best = (base, None)
            for dth, dd, dz in steps:
                if _too_low(p, z + dz):
                    continue
                trial = dict(k, c=tuple(round(float(v), 4) for v in _polar(th + dth, d + dd, z + dz)))
                kit[p] = trial
                cost = clash(kit, p) + 10 * max(0.0, reach(trial, p=p) - TIP_REACH) + 0.002 * (abs(dth) / 3 + abs(dz) / 0.03)
                if cost < best[0] - 1e-6:
                    best = (cost, trial)
            kit[p] = best[1] if best[1] is not None else k
            moved = moved or best[1] is not None
        if not moved:
            break


RELOCATE = [(dth, dd, dz) for dth in range(-24, 25, 6) for dd in (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
            for dz in (-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15)]


def _relax(kit, movable, rule):
    """Each nudged piece moves back toward its rule spot as far as it stays as clear as it is: a piece pushed
    aside by one that has since moved on does not stay where it was pushed."""
    for p in movable:
        if p not in rule:
            continue
        k = kit[p]
        now = clash(kit, p)
        c0, c1 = np.array(k["c"], float), np.array(rule[p], float)
        for t in (1.0, 0.75, 0.5, 0.25):
            trial = dict(k, c=tuple(round(float(v), 4) for v in c0 + t * (c1 - c0)))
            kit[p] = trial
            if clash(kit, p) <= now + 1e-6 and reach(trial, p=p) <= max(TIP_REACH, reach(k, p=p)):
                break
            kit[p] = k


def _relocate(kit, p):
    """A piece the nudges left stuck against the rest jumps to the clearest spot of a wider grid around it."""
    k = kit[p]
    x, y, z = k["c"]
    th = math.degrees(math.atan2(x - SEAT[0], y - SEAT[1]))
    d = math.hypot(x - SEAT[0], y - SEAT[1])
    best = (clash(kit, p) + 10 * max(0.0, reach(k, p=p) - TIP_REACH), k)
    for dth, dd, dz in RELOCATE:
        if _too_low(p, z + dz):
            continue
        trial = dict(k, c=tuple(round(float(v), 4) for v in _polar(th + dth, d + dd, z + dz)))
        kit[p] = trial
        cost = clash(kit, p) + 10 * max(0.0, reach(trial, p=p) - TIP_REACH) + \
            0.002 * (abs(dth) / 6 + abs(dd) / 0.04 + abs(dz) / 0.03)
        if cost < best[0] - 1e-6:
            best = (cost, trial)
    kit[p] = best[1]


def azimuth(k):
    """Degrees from straight ahead (positive to the drummer's right) of piece k, seen from the seat."""
    x, y = k["c"][0] - SEAT[0], k["c"][1] - SEAT[1]
    return math.degrees(math.atan2(x, y))


def check(kit):
    """What is left after layout: overlapping pieces, strike points out of the stick's reach, and pieces beside
    or behind the seat (only hand placement can put them there)."""
    over = {p: round(clash(kit, p), 4) for p in kit}
    return {"pieces": len(kit), "overlaps": {p: v for p, v in over.items() if v > CHECK_TOL},
            "out_of_reach": {p: round(reach(k, p=p), 3) for p, k in kit.items()
                             if k["kind"] != "kick" and reach(k, p=p) > TIP_REACH},
            **({"behind": b} if (b := {p: round(azimuth(k)) for p, k in kit.items()
                                      if abs(azimuth(k)) > FLOOR_MAX}) else {})}


def gm_map(kit):
    """General MIDI drum pitch -> piece of this kit. The six GM toms (50 high .. 41 low floor) spread over the
    kit's rack and floor toms by size; 55 is the splash and 52 the china when the kit has them."""
    toms = sorted((p for p in kit if piece_type(p) == "tom"), key=lambda p: kit[p]["r"]) + \
        sorted((p for p in kit if piece_type(p) == "floor"), key=lambda p: kit[p]["r"])
    crashes = [p for p in kit if piece_type(p) == "crash"]
    first = lambda t, alt=None: next((p for p in kit if piece_type(p) == t), alt)
    m = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
         42: "hihat", 44: "hihat_pedal", 46: "hihat"}
    if toms:
        for i, note in enumerate((50, 48, 47, 45, 43, 41)):
            m[note] = toms[int(math.floor(i * (len(toms) - 1) / 5 + 0.5))]
    ride = first("ride", crashes[-1] if crashes else None)
    c1 = crashes[0] if crashes else ride
    c2 = crashes[1] if len(crashes) > 1 else c1
    m.update({49: c1, 57: c2, 55: first("splash", c1), 52: first("china", c2), 51: ride, 53: ride, 59: ride})
    return {k: v for k, v in m.items() if v}


def remap(hits, kit):
    """Points every hit at a piece of this kit: by its GM note when it has one, otherwise by the standard kit's
    piece name (an audio detector's 'tom1' becomes this kit's smallest tom)."""
    m = gm_map(kit)
    std = gm_map(STANDARD)
    by_piece = {}
    for note, p in std.items():
        by_piece.setdefault(p, m.get(note))
    out = []
    for h in hits:
        p = m.get(h["note"]) if "note" in h else (h["piece"] if h["piece"] in kit else by_piece.get(h["piece"]))
        if p or h["piece"] == "hihat_pedal":
            out.append(dict(h, piece=p or h["piece"]))
    return out


# ---------------------------------------------------------------- hand-typed specs
ALIASES = {"bass": "kick", "bass drum": "kick", "kick drum": "kick", "snare drum": "snare", "hi-hat": "hihat",
           "hi hat": "hihat", "hat": "hihat", "hats": "hihat", "rack tom": "tom", "tom tom": "tom", "tom-tom": "tom",
           "mounted tom": "tom", "floor tom": "floor", "crash cymbal": "crash", "ride cymbal": "ride",
           "splash cymbal": "splash", "china cymbal": "china", "chinese": "china"}


SIZE_RANGE = (6.0, 28.0)       # inches


def _first_json(text):
    """The first JSON object or array in text; prose, code fences and stray braces around it are skipped."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                v, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(v, (dict, list)):
                return v
    raise ValueError("no JSON object in the kit spec")


def _numbers(v, n):
    if isinstance(v, (list, tuple)) and len(v) >= n and all(isinstance(x, (int, float)) for x in v[:n]):
        return [float(x) for x in v[:n]]
    return None


def spec_from_text(text):
    """(spec, notes) from a typed kit spec: the first JSON object (or bare piece list) in the text, piece types
    by name or common alias, sizes as numbers or strings like '16"' or '14x5.5' (the diameter comes first),
    id / at / tilt / stand kept. Unknown types are dropped; a missing kick, snare or hi-hat is added in its
    standard size; extra kicks become a double pedal. notes says what was changed."""
    spec = _first_json(text)
    if isinstance(spec, list):
        spec = {"pieces": spec}
    notes, pieces = [], []
    for pc in spec.get("pieces", []):
        if isinstance(pc, str):
            pc = {"type": pc}
        if not isinstance(pc, dict):
            notes.append(f"dropped {pc!r}: not a piece")
            continue
        name = str(pc.get("type", "")).strip().lower()
        t = ALIASES.get(name, name.replace(" ", ""))
        if t not in TYPES:
            notes.append(f"dropped unknown piece {name!r}")
            continue
        out = {"type": t}
        size = pc.get("size")
        m = re.match(r"\s*(\d+(?:\.\d+)?)", str(size)) if size is not None else None
        if m:
            s = float(m.group(1))
            out["size"] = min(max(s, SIZE_RANGE[0]), SIZE_RANGE[1])
            if out["size"] != s:
                notes.append(f"{t} size {s:g} in clamped to {out['size']:g}")
        elif size is not None:
            notes.append(f"{t}: ignored size {size!r}")
        if isinstance(pc.get("id"), str) and pc["id"]:
            out["id"] = pc["id"]
        for key, n in (("at", 3), ("stand", 2)):
            if key in pc:
                v = _numbers(pc[key], n)
                if v is None:
                    notes.append(f"{t}: ignored {key} {pc[key]!r}")
                else:
                    out[key] = v
        if isinstance(pc.get("tilt"), (int, float)):
            out["tilt"] = float(pc["tilt"])
        pieces.append(out)
    for t in ("kick", "snare", "hihat"):
        free = [i for i, pc in enumerate(pieces) if pc["type"] == t and ("at" not in pc or t == "kick")]
        if not any(pc["type"] == t for pc in pieces):
            pieces.append({"type": t})
            notes.append(f"added a {t}")
        elif len(free) > 1:
            pieces = [pc for i, pc in enumerate(pieces) if i not in free[1:]]
            notes.append(f"kept one {t} of {len(free)}" + (" (two kicks: double_pedal)" if t == "kick" else ""))
            if t == "kick":
                spec["double_pedal"] = True
    seen, extra = {}, 0
    for pc in list(pieces):     # the rule layout has standard spots for two crashes and a ride, then SLOTS
        if pc["type"] in CYMBALS and "at" not in pc:
            seen[pc["type"]] = seen.get(pc["type"], 0) + 1
            std = (pc["type"] == "crash" and seen["crash"] <= 2) or (pc["type"] == "ride" and seen["ride"] == 1)
            if not std:
                extra += 1
                if extra > len(SLOTS):
                    pieces.remove(pc)
                    notes.append(f"dropped a {pc['type']}: no free cymbal spot (place it with 'at')")
    return {"double_pedal": bool(spec.get("double_pedal")), "pieces": pieces}, notes


def load(arg):
    """(kit, notes) from a preset name or a kit spec file (read with spec_from_text)."""
    if arg in PRESETS:
        return build(arg), []
    with open(arg, encoding="utf-8") as f:
        spec, notes = spec_from_text(f.read())
    return build(spec), notes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help=f"preset ({', '.join(PRESETS)}) or a kit spec JSON")
    ap.add_argument("--out", help="write the kit dict as JSON")
    a = ap.parse_args()
    kit, notes = load(a.spec)
    for n in notes:
        print(n)
    if a.out:
        json.dump(kit, open(a.out, "w"), indent=1)
    for p, k in kit.items():
        print(f"{p:<8} {k['kind']:<7} r {k['r']:.3f}  c {tuple(round(x, 3) for x in k['c'])}")
    print(json.dumps(check(kit)))

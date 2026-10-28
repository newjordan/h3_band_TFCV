"""Music-driven camera edit for a performer blockout: shot list, cuts on downbeats, tracking cameras.

    python -m blockout.cameras ANIM.json OUT.json [--seed N]

Adds fr["cam"] = {"pos", "look", "up", "lens", "shot", "hide_head"} to every frame; render with
blender_piano.py --view edit. Rules:
- cuts land on downbeats; shots are long (~6 s soft, ~10 s at the climax): the impressive thing is
  seeing him actually play, so the big moments hold still and wide instead of cutting on the action;
- the shot is drawn (seeded) from weights that also follow energy: soft passages favour the wide, the
  first-person view, slow orbits and gentle hand tracking; loud passages favour views that show both
  hands on the keys (hands, wide, overhead, slow orbit) over the helm and single-hand close-ups;
- composed, not chasing: every shot except the first-person view holds one framing, set from where the hands
  are over the whole shot; only the wide's slow push-in, a slow orbit and the closing pull-back move;
- the point is to watch the hands hit the keys: shots that don't show them (the low hero angle on the helm)
  are rare;
- never the same shot twice in a row; opens on the wide, closes on a slow rising pull-back;
- within a shot nothing is locked off: wides push in, orbits drift, tracking shots follow the hands
  through springs, so the camera moves like an operator, not a rail.
"""
import argparse, json, math

import numpy as np

from . import dynamics

#            soft  loud
WEIGHTS = {"wide":       (3.0, 2.0),
           "pov":        (2.0, 1.0),
           "orbit":      (2.0, 2.0),
           "hands":      (2.0, 3.0),
           "overhead":   (1.0, 2.0),
           "side":       (0.6, 0.5),
           "hero_low":   (0.0, 0.0),        # the helm without the hands: kept for other pieces, never drawn here
           "rh_close":   (1.0, 0.4),
           "lh_close":   (1.0, 0.4),
           "keys_low":   (0.5, 0.3)}


LOCKED = ("hands", "rh_close", "lh_close", "overhead", "side", "keys_low", "hero_low")
FOLLOW_MIN = 0.20         # m a hand travels across the keys in a shot before the camera follows it


def _travel(frames, f0, f1, side):
    xs = [_hand(frames[i], side)[0] for i in range(f0, f1, 3) if side in frames[i]["hands"]]
    return (max(xs) - min(xs)) if xs else 0.0


def plan(frames, downbeats, fps, seed=0, beats=()):
    """Shot list [(frame_start, frame_end, shot)]."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    energy = np.array([f.get("energy", 0.3) for f in frames])
    db = [int(round(d * fps)) for d in downbeats if 0 < d * fps < n]
    bt = [int(round(b * fps)) for b in beats if 0 < b * fps < n]
    shots, f0, last = [], 0, None
    while f0 < n:
        e = float(energy[min(f0 + fps, n - 1)])
        target = int((6.0 + 4.0 * e) * fps)                   # the climax holds longest
        nxt = [d for d in db if d >= f0 + target]
        f1 = nxt[0] if nxt else n
        if n - f1 < 2 * fps:
            f1 = n
        if not shots:
            name = "wide"
        elif f1 == n:
            name = "pullback"
        else:
            names = [k for k in WEIGHTS if k != last and sum(WEIGHTS[k]) > 0]
            w = np.array([WEIGHTS[k][0] * (1 - e) + WEIGHTS[k][1] * e for k in names])
            name = names[int(rng.choice(len(names), p=w / w.sum()))]
        if shots and name == shots[-1][2]:              # the same follow again: one continuous shot, no cut
            shots[-1] = (shots[-1][0], f1, name)
        else:
            shots.append((f0, f1, name))
        last, f0 = name, f1
    return shots


def _mid_hands(fr):
    w = [np.mean([c[3] for c in h["fingers"]], axis=0) for h in fr["hands"].values()]
    return np.mean(w, axis=0)


def _hand(fr, side):
    h = fr["hands"].get(side) or next(iter(fr["hands"].values()))
    return np.mean([c[3] for c in h["fingers"]], axis=0)


def camera(name, fr, u, cx, seed):
    """Raw (pos, look, up, lens) for shot `name` at shot progress u (0..1)."""
    head = np.array(fr["body"]["head"])
    mid = _mid_hands(fr)
    up = np.array([0, 0, 1.0])
    rng = np.random.default_rng(seed)
    side = 1 if rng.random() < 0.5 else -1
    if name == "wide":
        p0, look = np.array([cx + 0.95, 0.55, 0.80]), np.array([cx - 0.05, -0.30, 0.22])
        return p0 + (look - p0) * 0.18 * u, look, up, 30.0 + 6 * u
    if name == "pullback":
        look = np.array([cx - 0.05, -0.30, 0.25])
        p = np.array([cx + 0.95, 0.55, 0.80]) + np.array([0.6, 0.9, 1.1]) * (u ** 1.5)
        return p, look, up, 30.0 - 6 * u
    if name == "orbit":
        a0 = rng.uniform(-0.9, 0.9)
        a = a0 + side * 0.12 * u
        centre = np.array([cx, -0.28, 0.28])
        return centre + np.array([1.25 * math.sin(a), 1.25 * math.cos(a), 0.45]), centre, up, 32.0
    if name == "hands":                                   # both hands from the player's side, high: the keys show
        return mid + np.array([0.18 * side, -0.48, 0.40]), mid + np.array([0, 0.04, -0.02]), up, 34.0
    if name in ("rh_close", "lh_close"):
        h = _hand(fr, "R" if name == "rh_close" else "L")
        s = 1 if name == "rh_close" else -1
        return h + np.array([0.26 * s, -0.17 + 0.05 * u, 0.27]), h + np.array([0, 0.02, -0.01]), up, 48.0
    if name.startswith("follow_"):
        h = _hand(fr, name[-1])
        s = 1 if name[-1] == "R" else -1
        return h + np.array([0.16 * s, -0.42, 0.34]), h + np.array([0, 0.03, -0.01]), up, 38.0
    if name == "overhead":
        return np.array([mid[0], 0.05, 0.78 - 0.08 * u]), np.array([mid[0], 0.05, 0.0]), np.array([0, 1.0, 0]), 32.0
    if name == "side":
        return np.array([cx + 1.25, -0.32, 0.32]), np.array([cx, -0.32, 0.26]), up, 40.0 + 8 * u
    if name == "hero_low":
        # beside the player, low (under the shoulder line, above the keybed), looking up at the head
        p = np.array([head[0] + 0.75 * side, head[1] + 0.30 - 0.06 * u, 0.13 + 0.04 * u])
        return p, head + np.array([0, 0, -0.04]), up, 32.0
    if name == "keys_low":
        x0 = mid[0] - side * 0.55
        return np.array([x0, 0.10, 0.045]), mid + np.array([0, 0.02, 0.0]), up, 48.0
    if name == "pov":
        eye, fwd, upv = (np.array(fr["body"][k]) for k in ("eye", "cam_fwd", "cam_up"))
        return eye, eye + fwd, upv, 30.0
    raise ValueError(name)


def _fit(pos, look, pts, up, lens, margin=0.8, aspect=1280 / 704):
    """Move the camera back along its view line until every point is inside `margin` of the frame."""
    hx = 18.0 / lens; hy = hx / aspect
    for s in np.linspace(1.0, 3.0, 41):
        p = look + (pos - look) * s
        f = look - p; f = f / np.linalg.norm(f)
        r = np.cross(f, up); r = r / np.linalg.norm(r); u = np.cross(r, f)
        v = pts - p; z = v @ f
        if (z > 0).all() and (np.abs(v @ r / z) <= margin * hx).all() and (np.abs(v @ u / z) <= margin * hy).all():
            return p
    return p


# ---------------------------------------------------------------- still edit (default): locked-off keyboard shots
STILL_ORDER = ("keys_high", "keys_3q_R", "keys_high", "overhead", "keys_high", "keys_3q_L")
STILL_LEN = 12.0          # s per shot, cut on the next downbeat


def still_plan(frames, downbeats, fps):
    n = len(frames)
    db = [int(round(d * fps)) for d in downbeats if 0 < d * fps < n]
    shots, f0, k = [], 0, 0
    while f0 < n:
        nxt = [d for d in db if d >= f0 + int(STILL_LEN * fps)]
        f1 = nxt[0] if nxt else n
        if n - f1 < 4 * fps:
            f1 = n
        shots.append((f0, f1, STILL_ORDER[k % len(STILL_ORDER)]))
        f0, k = f1, k + 1
    return shots


def still_camera(name, frames, f0, f1):
    """One fixed camera for the shot: it sees the keys both hands play during it, and the hands, all the time."""
    pts = np.array([q[k] for i in range(f0, f1, 2) for h in frames[i]["hands"].values() for q in h["fingers"]
                    for k in (0, 3)] + [h["wrist"] for i in range(f0, f1, 2) for h in frames[i]["hands"].values()])
    x0, x1 = pts[:, 0].min() - 0.06, pts[:, 0].max() + 0.06
    keys = np.array([[x, y, 0.0] for x in np.linspace(x0, x1, 7) for y in (0.0, 0.14)])
    allp = np.vstack([pts, keys])
    kc = 0.5 * (x0 + x1)
    if name.startswith("keys_3q") and x1 - x0 > 0.55:   # both hands spread wide: the oblique would sit far back
        name = "keys_high"
    look = np.array([kc, 0.06, 0.0])
    up = np.array([0, 0, 1.0])
    if name == "keys_high":
        pos = np.array([kc, -0.30, 0.62])
    elif name == "overhead":
        pos, up = np.array([kc, 0.02, 0.70]), np.array([0, 1.0, 0])
    else:
        s = 1 if name.endswith("R") else -1         # high on the player's side, offset: the keys at an angle
        pos = np.array([kc + s * 0.38, -0.30, 0.58])
    lens = 35.0
    return _fit(pos, look, allp, up, lens, margin=0.85), look, up, lens, name


def direct_still(anim):
    frames, fps = anim["frames"], anim["fps"]
    shots = still_plan(frames, anim.get("downbeats", []), fps)
    labels = []
    for f0, f1, name in shots:
        p, l, u, lens, name = still_camera(name, frames, f0, f1)
        labels.append((f0, f1, name))
        for i in range(f0, f1):
            frames[i]["cam"] = {"pos": [round(float(c), 5) for c in p], "look": [round(float(c), 5) for c in l],
                                "up": [float(c) for c in u], "lens": lens, "shot": name, "hide_head": True,
                                "hide_body": True}
    anim["shots"] = [{"start": f0 / fps, "end": f1 / fps, "shot": s} for f0, f1, s in labels]
    return anim


def direct(anim, seed=0):
    frames, fps = anim["frames"], anim["fps"]
    cx = anim["centre_x"]
    shots = plan(frames, anim.get("downbeats", []), fps, seed, anim.get("beats", []))
    for k, (f0, f1, name) in enumerate(shots):
        L = max(f1 - f0, 1)
        raw = [camera(name, frames[i], (i - f0) / L, cx, seed * 1000 + k) for i in range(f0, f1)]
        pos = np.array([r[0] for r in raw]); look = np.array([r[1] for r in raw])
        if name in LOCKED:                                # composed: one framing for the whole shot, backed off
            p0, l0 = pos.mean(0), look.mean(0)           # until both hands stay inside it all the way through
            pts = np.array([q[3] for i in range(f0, f1, 2) for h in frames[i]["hands"].values() for q in h["fingers"]]
                           + [h["wrist"] for i in range(f0, f1, 2) for h in frames[i]["hands"].values()])
            p0 = _fit(p0, l0, pts, raw[0][2], raw[0][3])
            pos[:] = p0; look[:] = l0
        elif name not in ("pov",):                        # the slow moves: smooth, a touch behind the action
            pos = dynamics.filter_track(pos, fps, 1.2, 0.9, 0.0)
            look = dynamics.filter_track(look, fps, 1.6, 0.85, 0.0)
        for i, (p, l, r) in enumerate(zip(pos, look, raw)):
            frames[f0 + i]["cam"] = {"pos": [round(float(c), 5) for c in p], "look": [round(float(c), 5) for c in l],
                                     "up": [float(c) for c in r[2]], "lens": round(float(r[3]), 2),
                                     "shot": name, "hide_head": name == "pov"}
    anim["shots"] = [{"start": f0 / fps, "end": f1 / fps, "shot": s} for f0, f1, s in shots]
    return anim



# ---------------------------------------------------------------- fingers edit: moving cameras that watch the fingers
# The steadicam on the busier hand is the backbone; the other shots break it up. Every camera follows its hand
# through a spring (an operator a beat behind, never locked to the bones), floats a few mm, and backs off on its
# own if the hand it watches would leave the frame. Cuts land on downbeats; opens descending (top-down) from the
# whole keyboard onto the hands, closes rising off them. The body is hidden throughout: only arms, hands and piano.
FINGER_SHOTS = {"steady": (5.0, 5.0),        # front three-quarter on one hand, high (the review handcam, smoothed)
                "steady_low": (1.5, 2.0),    # low from the hand's outer end of the keys: the strikes in profile
                "both": (1.5, 2.5),          # both hands from the player's side
                "overhead": (1.2, 1.2),      # straight down on one hand, turning slowly
                "orbit": (1.5, 1.5),         # round one hand from its outer side to the front
                "across": (1.2, 1.0),        # from inside the piano (case hidden), looking back at the fingers
                "pov": (0.8, 0.5)}           # the player's eyes
SINGLE = ("steady", "steady_low", "overhead", "orbit", "across")
MAX_SHOT = 14.0           # s: a camera that keeps being drawn runs on at most this long before a cut
ARM_CLEAR = 0.08          # m the camera keeps from either arm's axis (forearm, upper arm)
FINGER_DOF = False        # depth of field on the close shots (off: every edge sharp for H3)
#            azimuth deg (start, end; + = the hand's outer side), elevation deg (start, end), distance m, lens mm, f-stop
SHOT_GEOM = {"steady": ((18, 32), (42, 40), 0.36, 32.0, 8.0),
             "steady_low": ((58, 68), (14, 16), 0.34, 35.0, 8.0),
             "both": ((-8, 8), (50, 46), 0.62, 30.0, None),
             "orbit": ((72, -8), (28, 40), 0.38, 32.0, 8.0),
             "across": ((160, 150), (40, 36), 0.40, 32.0, 8.0),
             "overhead": ((0, 0), (90, 90), 0.42, 30.0, None)}


def _track_pts(fr, side):
    h = fr["hands"][side]
    return np.array([c[k] for c in h["fingers"] for k in (0, 3)])


def _onsets(frames, f0, f1, side):
    n, prev = 0, set()
    for i in range(f0, f1):
        cur = {p for p, who in frames[i].get("press", {}).items() if who[0] == side}
        n += len(cur - prev)
        prev = cur
    return n


def _in_frame(pos, look, up, lens, pts, aspect=16 / 9):
    """Largest |screen coordinate| of pts as a fraction of the half frame (1 = on the edge)."""
    hx = 18.0 / lens; hy = hx / aspect
    f = look - pos; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r); u = np.cross(r, f)
    v = pts - pos; z = v @ f
    if (z <= 0.02).any():
        return 9.0
    return float(max(np.abs(v @ r / z).max() / hx, np.abs(v @ u / z).max() / hy))


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + ab * t[..., None]), axis=-1)


def _arm_clear(pos, frames, f0, f1):
    """Closest the camera comes to either arm (forearm or upper arm axis) over the shot, m."""
    d = 9.0
    for i in range(f0, f1, 2):
        fr, p = frames[i], pos[i - f0]
        for h, hd in fr["hands"].items():
            sh = np.array(fr["body"]["shoulders"][h]); el = np.array(hd["elbow"]); wr = np.array(hd["wrist"])
            d = min(d, float(_seg_dist(p, el, wr)), float(_seg_dist(p, sh, el)))
    return d


def _shot_track(name, frames, f0, f1, side, fps, seed, dscale=1.0, energy=0.3, cx=0.0, swing=0.0):
    """pos, look, up, lens, fstop, focus per frame for one shot (swing: deg further round to the hand's outer side)."""
    L = max(f1 - f0, 1)
    t = np.arange(L) / fps
    u = np.clip(np.arange(L) / max(L - 1, 1), 0, 1)
    ease = u * u * (3 - 2 * u)
    sides = [side] if side else ["L", "R"]
    raw = np.array([np.mean([_track_pts(frames[i], s).mean(0) for s in sides], axis=0) for i in range(f0, f1)])
    fq = 0.9 + 0.8 * energy                                 # a busy passage: the operator keeps up a little harder
    tgt = dynamics.filter_track(raw, fps, fq, 1.0, 0.0)
    look = dynamics.filter_track(raw, fps, fq * 1.6, 0.95, 0.0) + np.array([0, 0.012, -0.008])
    up = np.tile([0, 0, 1.0], (L, 1))
    fl = np.stack([dynamics.drift(t, 0.0025, seed + k) for k in range(3)], 1)     # hand-held float, mm
    sg = 1.0 if side == "R" else -1.0
    if name == "pov":
        eye = np.array([frames[i]["body"]["eye"] for i in range(f0, f1)])
        fwd = np.array([frames[i]["body"]["cam_fwd"] for i in range(f0, f1)])
        upv = np.array([frames[i]["body"]["cam_up"] for i in range(f0, f1)])
        eye = dynamics.filter_track(eye, fps, 1.5, 1.0, 0.0)
        return eye, eye + fwd, upv, np.full(L, 30.0), [None] * L, raw
    if name in ("descend", "rise"):                      # top-down: from the whole keyboard onto the hands (or back)
        k = ease if name == "descend" else 1 - ease
        spread = max(abs(_track_pts(frames[i], "R").mean(0)[0] - _track_pts(frames[i], "L").mean(0)[0])
                     for i in range(f0, f1, 4))
        near = max(0.48, 0.20 + 0.85 * spread)              # m above the keys at the hands: both stay in frame
        h = max(0.95, near + 0.3) * (1 - k) + near * k
        x = cx + (tgt[:, 0] - cx) * k
        y = 0.07 + (tgt[:, 1] - 0.045 - 0.07) * k + 0.10 * (1 - k)    # high up, framed toward the keys so the elbows
                                                                      # (where the skinned forearm ends) stay below frame
        yaw = np.radians(14.0) * (1 - k)
        pos = np.stack([x, y - 0.02, h], 1) + fl
        lk = np.stack([x, y, np.zeros(L)], 1)
        up = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(L)], 1)
        return pos, lk, up, np.full(L, 30.0), [None] * L, raw
    a0, e0, d, lens, fstop = SHOT_GEOM[name]
    az = np.radians(a0[0] + (a0[1] - a0[0]) * ease + swing) * (sg if side else 1.0)
    el = np.radians(e0[0] + (e0[1] - e0[0]) * ease)
    if name == "both":
        spread = np.array([np.linalg.norm(_track_pts(frames[i], "R").mean(0) - _track_pts(frames[i], "L").mean(0))
                           for i in range(f0, f1)])
        d = dynamics.filter_track(np.maximum(d, 0.30 + 1.05 * spread)[:, None], fps, 0.5, 1.0, 0.0)[:, 0]
    d = d * dscale
    if name == "overhead":
        rot = np.radians(8.0 * (2 * ease - 1)) * sg
        off = np.stack([np.zeros(L), np.full(L, -0.05), np.full(L, 1.0)], 1) * np.asarray(d).reshape(-1, 1)
        up = np.stack([-np.sin(rot), np.cos(rot), np.zeros(L)], 1)
        look = look - np.array([0, 0.052, 0])          # the whole hand centred, palm and all
        tgt = tgt - np.array([0, 0.040, 0])
    else:
        off = np.stack([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)], 1) * np.asarray(d).reshape(-1, 1)
    return tgt + off + fl, look, up, np.full(L, lens), [fstop] * L, raw


def _best(name, side, frames, f0, f1, fps, seed, cx, out_margin):
    """The planned camera, backed off until its hand stays in frame; if it would pass through an arm, the same camera
    swung further round to the hand's outer side, then the steadicam (likewise), then straight down, then both hands."""
    energy = float(np.mean([frames[i].get("energy", 0.3) for i in range(f0, f1)]))
    busy = "R" if _onsets(frames, f0, f1, "R") >= _onsets(frames, f0, f1, "L") else "L"
    if name in ("pov", "descend", "rise"):
        chain = [(name, 0.0)]
    elif name == "both":
        chain = [("both", 0.0), ("overhead", 0.0)]
    else:
        chain = [(name, 0.0), (name, 18.0), ("steady", 0.0), ("steady", 25.0), ("overhead", 0.0), ("both", 0.0)]
    for nm, swing in chain:
        sd = None if nm in ("both", "pov", "descend", "rise") else (side or busy)
        pts = [np.vstack([_track_pts(frames[i], s) for s in ([sd] if sd else ["L", "R"])]) for i in range(f0, f1)]
        for dscale in (1.0, 1.12, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3):
            tr = _shot_track(nm, frames, f0, f1, sd, fps, seed, dscale, energy, cx, swing)
            worst = max(_in_frame(tr[0][i], tr[1][i], tr[2][i], tr[3][i], pts[i]) for i in range(0, f1 - f0, 2))
            if worst <= out_margin or nm == "pov":
                break
        clear = _arm_clear(tr[0], frames, f0, f1)
        if clear >= ARM_CLEAR or nm in ("pov", "descend", "rise"):
            break
    return {"f0": f0, "f1": f1, "name": nm, "side": sd, "swing": swing, "track": tr, "dscale": dscale, "worst": worst,
            "clear": clear}


def direct_fingers(anim, seed=0, out_margin=0.92):
    frames, fps = anim["frames"], anim["fps"]
    rng = np.random.default_rng(seed)
    n = len(frames)
    energy = np.array([f.get("energy", 0.3) for f in frames])
    db = [int(round(d * fps)) for d in anim.get("downbeats", []) if 0 < d * fps < n]
    cuts, f0 = [], 0
    while f0 < n:                                            # ~8 s soft, ~6 s loud, on the next downbeat
        e = float(energy[f0:min(f0 + 6 * fps, n)].mean())
        nxt = [d for d in db if d >= f0 + int((8.5 - 3.0 * e) * fps)]
        f1 = nxt[0] if nxt else n
        if n - f1 < 4 * fps:
            f1 = n
        cuts.append((f0, f1)); f0 = f1
    shots, last, last_side = [], None, None
    for k, (f0, f1) in enumerate(cuts):
        e = float(energy[f0:f1].mean())
        if k == 0:
            name, side = "descend", None
        elif k == len(cuts) - 1:
            name, side = "rise", None
        else:
            names = [m for m in FINGER_SHOTS if m != last or m == "steady"]
            if last == "steady" and shots[-2:] and all(s[2] == "steady" for s in shots[-2:]):
                names.remove("steady")                       # three steadicams in a row at most
            w = np.array([FINGER_SHOTS[m][0] * (1 - e) + FINGER_SHOTS[m][1] * e for m in names])
            name = names[int(rng.choice(len(names), p=w / w.sum()))]
            side = None
            if name in SINGLE:
                oR, oL = _onsets(frames, f0, f1, "R"), _onsets(frames, f0, f1, "L")
                busy = "R" if oR >= oL else "L"
                close = min(oR, oL) >= 0.7 * max(oR, oL, 1)
                side = ("L" if last_side == "R" else "R") if close and last_side else busy
            if shots and (name, side) == shots[-1][2:] and f1 - shots[-1][0] > MAX_SHOT * fps:
                if name in SINGLE:                                # long enough on this hand: cut to the other one
                    side = "L" if side == "R" else "R"
                else:
                    names.remove(name)
                    w = np.array([FINGER_SHOTS[m][0] * (1 - e) + FINGER_SHOTS[m][1] * e for m in names])
                    name = names[int(rng.choice(len(names), p=w / w.sum()))]
                    side = None if name not in SINGLE else ("L" if last_side == "R" else "R")
        if shots and (name, side) == shots[-1][2:]:            # the same camera again: one continuous shot, no jump cut
            shots[-1] = (shots[-1][0], f1, name, side)
        else:
            shots.append((f0, f1, name, side))
        last, last_side = name, side or last_side
    res = []
    for k, (f0, f1, name, side) in enumerate(shots):
        r = _best(name, side, frames, f0, f1, fps, seed * 1000 + k, anim["centre_x"], out_margin)
        if res and (r["name"], r["side"]) == (res[-1]["name"], res[-1]["side"]) and name not in ("descend", "rise"):
            m = _best(r["name"], r["side"], frames, res[-1]["f0"], f1, fps, seed * 1000 + k, anim["centre_x"], out_margin)
            if (m["name"], m["side"]) == (r["name"], r["side"]):       # a fallback landed on the camera before it:
                res[-1] = m                                          # one continuous shot instead of a jump cut
                continue
        res.append(r)
    out = []
    for r in res:
        f0, f1, name, side, dscale = r["f0"], r["f1"], r["name"], r["side"], r["dscale"]
        pos, look, up, lens, fst, focus = r["track"]
        for i in range(f1 - f0):
            c = {"pos": [round(float(v), 5) for v in pos[i]], "look": [round(float(v), 5) for v in look[i]],
                 "up": [round(float(v), 5) for v in up[i]], "lens": float(lens[i]), "shot": name + (f"_{side}" if side else ""),
                 "hide_head": True, "hide_body": True, "hide_case": name == "across"}
            if fst[i] and FINGER_DOF:
                c["fstop"] = float(fst[i]) * dscale ** 2           # backed off: keep the same depth on the hand
                c["focus"] = [round(float(v), 5) for v in focus[i]]
            frames[f0 + i]["cam"] = c
        out.append({"start": f0 / fps, "end": f1 / fps, "shot": name + (f"_{side}" if side else ""), "swing": r["swing"],
                    "backoff": round(dscale, 2), "in_frame": round(r["worst"], 2), "arm_clear": round(r["clear"], 3)})
    anim["shots"] = out
    return anim

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("anim"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", default="still", choices=["still", "music", "fingers"],
                    help="still: locked-off shots of the keys and hands (default); music: the older energy-driven edit; "
                         "fingers: moving cameras that follow the fingers (steadicam first)")
    a = ap.parse_args()
    anim = json.load(open(a.anim))
    anim = (direct_still(anim) if a.style == "still" else direct(anim, a.seed) if a.style == "music"
            else direct_fingers(anim, a.seed))
    json.dump(anim, open(a.out, "w"))
    from collections import Counter
    print(len(anim["shots"]), "shots:", dict(Counter(s["shot"] for s in anim["shots"])))

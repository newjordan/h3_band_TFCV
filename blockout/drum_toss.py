"""Stick toss and catch for the drum blockout (blockout/drums.py animate(tosses=...)).

A toss {"hand": "R", "t0": release s, "t1": catch s, "spins": 1} takes one hand off the kit in a gap between its
strokes (animate() gives the other hand any stroke that falls in or near the window). The hand moves to a spot
in front of the chest, flicks up and lets go; the stick flies as a rigid body, its centre on a ballistic arc and
turning end over end at a steady rate; it lands back in the hand in exactly the pose the hand holds it in at the
catch, the hand gives a little, then goes back to the strokes after it, which are untouched.

In flight a frame's tip and butt are the stick's, grip and wrist the hand's, and "hand_dir" is the line the empty
hand holds (drum_hands.py and the hitboxes read it instead of tip - butt).
"""
import math

import numpy as np

from . import drum_hands, performer

G = 9.81
MOVE = 0.25         # s the hand takes to reach the toss spot, and to get back to its strokes after the catch
FLICK = 0.12        # s the hand rises with the stick before it lets go
FLICK_UP = 0.06     # m it rises
ABSORB = 0.025      # m the hand gives as the stick lands
ABSORB_S = 0.15     # s it takes
CLEAR = 0.15        # s between a toss's hand motion and the hand's strokes either side
SPOT = (-0.02, 0.32, -0.10)    # toss spot (grip) from the shoulder; x is toward the body's centre
SPOT_DIR = (-0.30, 1.0, 0.20)  # stick direction there (x toward the centre)


def parse(text):
    """'R:4.85:5.55[:spins],L:...' -> [toss]."""
    out = []
    for item in filter(None, (s.strip() for s in (text or "").split(","))):
        f = [x.strip() for x in item.split(":")]
        try:
            if len(f) not in (3, 4) or f[0].upper() not in ("R", "L"):
                raise ValueError
            ts = {"hand": f[0].upper(), "t0": float(f[1]), "t1": float(f[2]), "spins": int(f[3]) if len(f) == 4 else 1}
        except ValueError:
            raise ValueError(f"toss {item!r}: want HAND:release_s:catch_s[:spins], HAND R or L, spins a whole "
                             "number") from None
        if ts["t1"] <= ts["t0"] or ts["spins"] < 0:
            raise ValueError(f"toss {item!r}: the catch must come after the release, spins 0 or more")
        out.append(ts)
    return out


def check(tosses):
    """Raises if two tosses by one hand overlap (the hand has to be back on its strokes between them)."""
    for hand in ("R", "L"):
        ws = sorted(window(ts) for ts in tosses if ts["hand"] == hand)
        for (a0, b0), (a1, b1) in zip(ws, ws[1:]):
            if a1 < b0:
                raise ValueError(f"tosses by hand {hand} around {b0:.2f} s overlap; leave "
                                 f"{b0 - a1:.2f} s more between them")


def window(toss):
    """(start, end) s the toss has the hand: from moving to the spot until it is back on its strokes."""
    return toss["t0"] - FLICK - MOVE - CLEAR, toss["t1"] + ABSORB_S + MOVE + CLEAR


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _rot(axis, ang):
    a = _unit(np.asarray(axis, float))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _slerp(a, b, w):
    ang = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    if ang < 1e-6:
        return b
    return _rot(np.cross(a, b) if np.linalg.norm(np.cross(a, b)) > 1e-9 else np.array([1.0, 0, 0]), w * ang) @ a


def apply(frames, toss, fps, start, stick, tip_len):
    """Rewrites the tossing hand in frames (animate()'s output, times relative to `start`). Returns a summary."""
    hand, side = toss["hand"], 1 if toss["hand"] == "R" else -1
    j0, j1 = int(round((toss["t0"] - start) * fps)), int(round((toss["t1"] - start) * fps))
    t0, t1 = j0 / fps, j1 / fps     # release and catch on frames, so the flick ends exactly where flight begins
    T = t1 - t0
    if T < 0.2:
        raise ValueError(f"toss at {toss['t0']}: the stick needs at least 0.2 s in the air")
    ja = max(0, int(math.floor((t0 - FLICK - MOVE) * fps)))
    jb = min(len(frames) - 1, int(math.ceil((t1 + ABSORB_S + MOVE) * fps)))
    if j0 <= 0 or j1 >= len(frames) - 1:
        raise ValueError(f"toss at {toss['t0']}: release and catch must fall inside the animation")
    spins = toss.get("spins", 1)
    # The flick cocks the wrist by `cock` and the flight turns spins*2pi - cock back to the held pose; this omega
    # makes the wrist's turning rate at release equal the stick's in the air.
    omega = 2 * math.pi * spins / (T + FLICK / 2)
    up = np.array([0.0, 0.0, 1.0])
    sd = _unit(np.array([side * SPOT_DIR[0], SPOT_DIR[1], SPOT_DIR[2]]))
    axis = _unit(np.cross(sd, up))      # the stick turns end over end in its own vertical plane
    cock = omega * FLICK / 2            # wrist angle at release

    def held(j):
        """Hand pose at frame j with the stick in it: the stroke pose blended to the toss spot, plus flick/give."""
        hd = frames[j]["hands"][hand]
        t = j / fps
        grip, d = np.array(hd["grip"]), _unit(np.array(hd["tip"]) - np.array(hd["butt"]))
        sh = np.array(hd["shoulder"])
        w = float(_smooth((t - (t0 - FLICK - MOVE)) / MOVE) * (1 - _smooth((t - t1 - ABSORB_S) / MOVE)))
        spot = sh + np.array([side * SPOT[0], SPOT[1], SPOT[2]])
        grip = grip + w * (spot - grip)
        d = _slerp(d, sd, w)
        if j <= j0:     # flick: the hand rises and the wrist cocks the tip up, reaching the stick's spin rate
            u = float(np.clip((t - (t0 - FLICK)) / FLICK, 0.0, 1.0))
            grip = grip + FLICK_UP * float(_smooth(u)) * up
            d = _rot(axis, cock * u * u) @ d
        elif j < j1:    # empty hand: settles back from the flick while the stick flies
            back = 1 - float(_smooth((t - t0) / (0.5 * T)))
            grip = grip + FLICK_UP * back * up
            d = _rot(axis, cock * back) @ d
        else:           # catch: the hand gives down, and the wrist turns on with the stick's spin and back
            give = math.sin(math.pi * min((t - t1) / ABSORB_S, 1.0))
            grip = grip - ABSORB * give * up
            d = _rot(axis, omega * ABSORB_S / math.pi * give) @ d
        return grip, d

    def put(j, grip, d, tip=None, butt=None):
        hd = frames[j]["hands"][hand]
        wrist = drum_hands.wrist(grip, d, hand)
        sh = np.array(hd["shoulder"])
        hd["grip"], hd["wrist"] = grip.tolist(), wrist.tolist()
        hd["tip"] = (grip + tip_len * d if tip is None else tip).tolist()
        hd["butt"] = (grip - (stick - tip_len) * d if butt is None else butt).tolist()
        hd["elbow"] = performer.arm_ik(sh, wrist, side, drum_hands.forward(d, hand)).tolist()
        if tip is not None:
            hd["hand_dir"] = d.tolist()
        else:
            hd.pop("hand_dir", None)

    pose = {j: held(j) for j in range(ja, jb + 1)}
    g0, d0 = pose[j0]
    g1, d1 = pose[j1]
    mid = tip_len - stick / 2      # the stick's centre from the grip, along the stick
    c0, c1 = g0 + mid * d0, g1 + mid * d1
    v0 = (c1 - c0 + 0.5 * G * T * T * up) / T
    turn = np.cross(d0, d1)
    ang = math.acos(float(np.clip(d0 @ d1, -1.0, 1.0)))
    apex = 0.0
    for j in range(ja, jb + 1):
        grip, d = pose[j]
        if j0 < j < j1:
            s = (j - j0) / (j1 - j0)
            tt = s * T
            c = c0 + v0 * tt - 0.5 * G * tt * tt * up
            ds = _rot(axis, 2 * math.pi * spins * s) @ d0
            if ang > 1e-6:
                ds = _rot(turn, s * ang) @ ds
            apex = max(apex, float(c[2] - c0[2]))
            put(j, grip, d, c + (stick / 2) * ds, c - (stick / 2) * ds)
        else:
            put(j, grip, d)
    return {"hand": hand, "t0": toss["t0"], "t1": toss["t1"], "spins": spins, "apex_m": round(apex, 3),
            "flight_s": round(T, 3)}

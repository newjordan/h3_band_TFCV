"""Music-driven camera edit for a performer blockout: shot list, cuts on downbeats, tracking cameras.

    python -m blockout.cameras ANIM.json OUT.json [--seed N]

Adds fr["cam"] = {"pos", "look", "up", "lens", "shot", "hide_head"} to every frame; render with
blender_piano.py --view edit. Rules:
- cuts land on downbeats; shots are long (~6 s soft, ~10 s at the climax): the impressive thing is
  seeing him actually play, so the big moments hold still and wide instead of cutting on the action;
- the shot is drawn (seeded) from weights that also follow energy: soft passages favour the wide, the
  first-person view, slow orbits and gentle hand tracking; loud passages favour views that show both
  hands on the keys (hands, wide, overhead, slow orbit) over the helm and single-hand close-ups;
- a hand that travels far in a shot (a run down from the top of the keyboard) gets a camera that follows it;
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
        elif max((tr := {h: _travel(frames, f0, f1, h) for h in ("R", "L")}).values()) > FOLLOW_MIN and \
                max(tr.values()) > 1.8 * min(tr.values()):
            name = "follow_" + max(tr, key=tr.get)        # one hand on a long run: stay with it
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
        a = a0 + side * 0.35 * u
        centre = np.array([cx, -0.28, 0.28])
        return centre + np.array([1.25 * math.sin(a), 1.25 * math.cos(a), 0.45]), centre, up, 32.0
    if name == "hands":
        return mid + np.array([0.14 * side, 0.55, 0.34]), mid + np.array([0, 0.02, -0.02]), up, 40.0
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


def direct(anim, seed=0):
    frames, fps = anim["frames"], anim["fps"]
    cx = anim["centre_x"]
    shots = plan(frames, anim.get("downbeats", []), fps, seed, anim.get("beats", []))
    for k, (f0, f1, name) in enumerate(shots):
        L = max(f1 - f0, 1)
        raw = [camera(name, frames[i], (i - f0) / L, cx, seed * 1000 + k) for i in range(f0, f1)]
        pos = np.array([r[0] for r in raw]); look = np.array([r[1] for r in raw])
        if name not in ("pov",):                          # an operator: smooth, a touch behind the action
            pos = dynamics.filter_track(pos, fps, 1.2, 0.9, 0.0)
            look = dynamics.filter_track(look, fps, 1.6, 0.85, 0.0)
        for i, (p, l, r) in enumerate(zip(pos, look, raw)):
            frames[f0 + i]["cam"] = {"pos": [round(float(c), 5) for c in p], "look": [round(float(c), 5) for c in l],
                                     "up": [float(c) for c in r[2]], "lens": round(float(r[3]), 2),
                                     "shot": name, "hide_head": name == "pov"}
    anim["shots"] = [{"start": f0 / fps, "end": f1 / fps, "shot": s} for f0, f1, s in shots]
    return anim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("anim"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    anim = direct(json.load(open(a.anim)), a.seed)
    json.dump(anim, open(a.out, "w"))
    from collections import Counter
    print(len(anim["shots"]), "shots:", dict(Counter(s["shot"] for s in anim["shots"])))

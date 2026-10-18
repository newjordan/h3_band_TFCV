"""De-pop pass on a finished take: a finger joint that moves faster than LIMIT (rad/s) while that finger is free gets its
path smoothed over a few frames around the pop; frames where the finger presses a key (or lands on one) stay exactly as
they are. The review metric counts a pop as any finger joint over 15 rad/s; on C8 nearly all the Rachmaninoff ones are
free fingers (the idle thumb's IP joint snapping, the free pinky's spread), not presses.

    python -m blockout.depop IN.json OUT.json [--limit 12]
"""
import json, math, os, sys

import numpy as np

os.environ.setdefault("HAND_SCALE", "1")
from . import piano as P
from .clean import _q_from_angles, _rebuild

LIMIT = 12.0            # rad/s: target ceiling for a free finger's joints (the metric's pop line is 15)
HALF = 4                # frames each side of a pop that the smoothing may move
LAND = 3                # frames before a press that count as the press (the strike path stays)
LAM = 30.0              # Whittaker smoothing strength inside a pop window
ROUNDS = 4


def depop(anim, log=print):
    fps = anim["fps"]
    T = len(anim["frames"])
    lim = LIMIT / fps
    total = 0
    for h in "RL":
        if h not in anim["frames"][0]["hands"]:
            continue
        rig = P._rig(h)
        Q = np.array([_q_from_angles(rig, f["hands"][h]["angles"]) for f in anim["frames"]])
        press = np.zeros((T, 5), bool)
        for j, f in enumerate(anim["frames"]):
            for p, who in f.get("press", {}).items():
                if who[0] == h:
                    press[max(0, j - LAND):j + 2, int(who[1]) - 1] = True
        Q0 = Q.copy()
        for f in range(5):
            a, b = rig.sl[f], rig.sl[f + 1]
            for _r in range(ROUNDS):
                dq = np.abs(np.diff(Q[:, a:b], axis=0)).max(1)
                bad = np.nonzero((dq > lim) & ~press[1:, f] & ~press[:-1, f])[0]
                if not len(bad):
                    break
                touched = np.zeros(T, bool)
                for k in bad:
                    touched[max(0, k - HALF):min(T, k + HALF + 2)] = True
                w = np.where(touched & ~press[:, f], 1.0, 1e6)          # outside the pop windows / on presses: pinned
                Q[:, a:b] = P._whittaker(Q[:, a:b], w, LAM)
        Q = np.clip(Q, rig.lo, rig.hi)
        moved = np.nonzero(np.abs(Q - Q0).max(1) > 1e-4)[0]
        for j in moved:
            _rebuild(anim, h, j, Q[j])
        total += len(moved)
    log(f"depop: {total} hand-frames smoothed")
    return anim


def main(argv=None):
    import argparse
    global LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--limit", type=float, default=LIMIT)
    o = ap.parse_args(argv)
    LIMIT = o.limit
    anim = json.load(open(o.inp))
    depop(anim)
    anim["depopped"] = {"limit": LIMIT, "half": HALF, "lam": LAM}
    json.dump(anim, open(o.out, "w"))
    print("wrote", o.out)


if __name__ == "__main__":
    main()

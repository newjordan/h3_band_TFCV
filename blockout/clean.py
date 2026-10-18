"""Contact cleanup on a finished take: fingers out of the keys they aren't playing, the rest of the motion untouched.

The working piano (blockout/action.py) sounds every key that skin goes a few mm into, and on V3 that is ~10 wrong
keys a second: free fingers riding low in the key tops, a pressing finger or thumb catching the next key, fingers
travelling low over the black keys. Making the solver's collisions stronger fixes some of that and costs presses
(the whole hand gives up its stretch). This pass works after the solve, finger by finger, with the wrist as it is:

1. Every frame, every finger: the skin (rig cross-sections, rebuilt from the take's joint angles) against the key
   tops with the working piano's contact model (piano.key_pen_round_v; pressed keys lowered by the take's depths).
   The finger's own keys (the note it is holding, coming down to or leaving, from the take's fingering) don't
   count. A finger with no note near also keeps over the black-key tops anywhere in the black-key zone (a leap can
   skip a whole black key between two frames). Skin deeper than TOL is a violation.
2. Each violating finger is re-solved alone (rig.solve_fingers, only=[f]): the same targets as the solve (a pressing
   pad stays on its key, slid toward the key's centre line), strong rounded-key collisions, and a strong pull to its
   own pose, so it moves only as much as it must.
3. The corrections (joint angle deltas) are smoothed in time per finger (Whittaker: pinned at the violating frames,
   eased to zero around them), so a fix eases in and out instead of popping, then the take's frames are rebuilt
   (joint points, pads, angles, key depths). Steps 1-3 repeat (PASSES).

    python -m blockout.clean IN.json NOTES.json OUT.json [--jobs N] [--t0 S --t1 S]
"""
import json, math, os, sys

import numpy as np

os.environ.setdefault("HAND_SCALE", "1")
from . import piano as P

TOL = 0.0003            # m of skin allowed inside a key the finger isn't playing
CLEAR = 0.0010          # m the re-solve aims for past that
GAIN_FREE, GAIN_PRESS = 400.0, 40.0     # key-collision weight (x RIG_W env) in the re-solve: free / pressing finger
PRIOR = 2e-3            # how hard a re-solved pressing finger holds its own pose (finger_residuals "smooth" weight)
PRIOR_FREE = 5e-4       # ... and a free one
CENTRE = 0.0            # share of the way a pressing pad's target slides to its key's centre line (0: stays put; >0 pushed thumbs off)
PRESS_W = (100.0, 300.0, 1000.0)          # a pressing pad's target weights (x across, y along, z depth): it stays where the solve put it
OWN_PRE, OWN_POST = 0.30, 0.15          # s: a finger owns a key this long before its onset and after its release
BLACK_ZONE = 0.005      # m in front of the black keys from where a free finger keeps over their tops
NEAR_OWN = 0.020        # m: a finger this close over a key it is about to play keeps its pad path like a pressing one
SWEEP_U = (0.25, 0.5)   # shares of the way to the neighbouring frames at which the skin is also tested (None: off)
SMOOTH_LAM, W_HI, W_LO = 200.0, 50.0, 1.0  # temporal blending of the corrections
MAX_DQ = 0.3            # rad: the most a correction turns one joint (a bigger one flips the finger's pose and pops)
W_PRESS = 1e4            # ... a frame where the finger presses (or lands) is held to its own correction: no lift bleeds in
FIX_PRESSING = False    # also apply a pressing finger's own re-solve (False: pressing fingers stay exactly as solved:
                        # their re-solves pass through unsmoothed and jitter, Rach pops 20 -> 49/min)
CROSS_GUARD = True      # undo a finger's fix when it crosses index..pinky tips that weren't crossed (the cleanup alone
                        # took crossed tips 0.97 -> 1.47% on the T1 suite)
TAPER = 3.0             # frames over which a free finger's correction fades in after a release / out before a landing
PASSES = 4
ITERS = 20


def _q_from_angles(rig, A):
    q = np.zeros(int(rig.sl[-1]))
    for ch, a in zip(rig.chains, rig.sl[:-1]):
        for i, (b, n, *_r) in enumerate(ch.dofs):
            q[a + i] = A[b][n]
    return q


def _own_keys(anim, notes, h):
    """Per frame, per finger: the set of pitches the finger is on, coming down to or leaving."""
    end = {(round(n[0], 3), n[2]): n[1] for n in notes}
    ev = {f: [] for f in range(5)}
    for s in anim["fingering"].get(h, []):
        for on, p, f in s:
            ev[f].append((on, end.get((round(on, 3), p), on + 0.3), p))
    st, fps, T = anim["start"], anim["fps"], len(anim["frames"])
    own = [[set() for _ in range(5)] for _ in range(T)]
    for f, E in ev.items():
        E.sort()
        for i, (on, off, p) in enumerate(E):
            if i + 1 < len(E):
                off = min(off, E[i + 1][0])
            j0 = max(0, int(math.floor((on - OWN_PRE - st) * fps)))
            j1 = min(T - 1, int(math.ceil((off + OWN_POST - st) * fps)))
            for j in range(j0, j1 + 1):
                own[j][f].add(p)
    for j, fr in enumerate(anim["frames"]):        # and whatever the take itself has the finger pressing
        for p, who in fr.get("press", {}).items():
            if who[0] == h:
                own[j][int(who[1]) - 1].add(int(p))
    return own


def _pen(S_, DW, DB, own, sweep=None):
    """Per sample: how far the skin is inside a key top it doesn't own (m), and the black-zone floor for a free
    finger (own empty). sweep = [(xy, low) of the same samples at the frame before, and after]: the skin is also
    tested half-way along its path to them (its height interpolated): a fast move can cross a key between frames."""
    C = np.array([s[2] for s in S_]); low = np.array([s[3] for s in S_]); W_ = np.array([s[4] for s in S_])
    if sweep:
        out = _pen(S_, DW, DB, own)
        for xy, lo_nb in sweep:
            for u in SWEEP_U:
                Sv = [(None, None, np.array([C[k, 0] + u * (xy[k][0] - C[k, 0]), C[k, 1] + u * (xy[k][1] - C[k, 1]), 0.0]),
                       low[k] + u * (lo_nb[k] - low[k]), W_[k]) for k in range(len(S_))]
                out = np.maximum(out, _pen(Sv, DW, DB, own))
        return out
    DW2, DB2 = DW.copy(), DB.copy()                # the finger's own keys sunk out of reach
    for p in own:
        if p in P._WHITE_IDX:
            DW2[P._WHITE_IDX[p]] = 1e3
        elif p in P._BLACK_IDX:
            DB2[P._BLACK_IDX[p]] = 1e3
    pen = P.key_pen_round_v(C[:, 0], C[:, 1], low, W_, DW2, DB2, np.full(len(S_), -1))
    if not own:
        bz = C[:, 1] >= P.WHITE_L - P.BLACK_L - BLACK_ZONE
        pen = np.where(bz, np.maximum(pen, P.BLACK_H - low), pen)
    return pen


def _frame_fix(args):
    """Re-solve the violating fingers of one hand at one frame. Returns (j, h, q_new or None, worst before)."""
    j, h, root_rot, root_pos, q, keys, own, pads, press_f, sweep = args
    rig = P._rig(h)
    DW, DB = P._depth_arrays({int(k): v for k, v in keys.items()})
    x = np.concatenate([root_pos, np.zeros(3), q])
    rr, rp, pts = rig.points(x, root_rot)
    rots = rig._rots
    bad, worst = [], 0.0
    for f, (ch, P_, R_) in enumerate(zip(rig.chains, pts, rots)):
        m = float(_pen(ch.samples(P_, R_), DW, DB, own[f], sweep[f]).max())
        worst = max(worst, m)
        if m > TOL:
            bad.append(f)
    if not bad:
        return j, h, None, worst

    def env_for(f_own):
        def env(samples, root_rot_, root_pos_):
            out = []
            for f in range(5):
                if not samples[f]:
                    continue
                g = GAIN_PRESS if f_own[f] else GAIN_FREE
                out.extend(math.sqrt(P.RIG_W["env"] * g) * np.maximum(_pen(samples[f], DW, DB, f_own[f], sweep[f]) + CLEAR, 0.0))
            return np.array(out)
        return env

    targets, weights = [], []
    for f in range(5):
        pd = np.array(pads[f])
        if press_f[f] is not None:                 # keep the press: depth and place along the key pinned, only
            pd[0] += CENTRE * (P.key_x(press_f[f]) - pd[0])     # slid toward the key's centre line (a pressing
            targets.append(pd)                                   # finger has no spare joints: anything else lifts it)
            weights.append(np.array(PRESS_W))
        else:
            targets.append(pd)
            weights.append(P.RIG_W["free"] * (P.THUMB_REST if f == 0 else 1.0))
    W = {k: v for k, v in P.RIG_W.items() if k not in ("press", "free")}
    sgn = 1.0 if h == "R" else -1.0

    def crossed(xx):                                # index..pinky tips out of order across the keys by > 4 mm
        _rr, _rp, pts_ = rig.points(xx, root_rot)
        X = np.array([pts_[f][3][0] for f in range(1, 5)]) * sgn
        return set(f + 1 for f in range(3) if X[f + 1] - X[f] < -0.004) | set(f + 2 for f in range(3) if X[f + 1] - X[f] < -0.004)
    cross0 = crossed(x) if CROSS_GUARD else set()
    xn = x
    for grp, prior in (([f for f in bad if press_f[f] is not None], PRIOR),
                       ([f for f in bad if press_f[f] is None], PRIOR_FREE)):
        if grp:                                    # a free finger may move further than a pressing one
            xn = rig.solve_fingers(root_rot, xn, targets, weights, env=env_for(own), W={**W, "smooth": prior}, last=x,
                                   iters=ITERS, only=grp)
    if CROSS_GUARD:                                 # a fix that crosses fingertips that weren't crossed is undone
        new = crossed(xn) - cross0
        for f in new & set(bad):
            a, b = rig.sl[f], rig.sl[f + 1]
            xn[6 + a:6 + b] = x[6 + a:6 + b]
    return j, h, xn[6:], worst


def _skin_xy(anim, h, j):
    """Per finger: (xy, lowest z) of its skin samples at frame j."""
    rig = P._rig(h)
    H = anim["frames"][j]["hands"][h]
    x = np.concatenate([np.array(H["root"]["pos"]), np.zeros(3), _q_from_angles(rig, H["angles"])])
    rr, rp, pts = rig.points(x, np.array(H["root"]["rot"]))
    out = []
    for ch, P_, R_ in zip(rig.chains, pts, rig._rots):
        S_ = ch.samples(P_, R_)
        out.append((np.array([s[2][:2] for s in S_]), np.array([s[3] for s in S_])))
    return out


def _rebuild(anim, h, j, q):
    """Write hand h's frame j from its root and joint angles q (points, pads, angles, its pressed keys' depths)."""
    rig = P._rig(h)
    fr = anim["frames"][j]
    H = fr["hands"][h]
    x = np.concatenate([np.array(H["root"]["pos"]), np.zeros(3), q])
    rr, rp, pts = rig.points(x, np.array(H["root"]["rot"]))
    pads = rig.effectors(pts, rig._rots)
    H["fingers"] = [[[round(float(c), 5) for c in p] for p in Pt] for Pt in pts]
    H["pads"] = [[round(float(c), 5) for c in p] for p in pads]
    H["angles"] = {b: {k: round(v, 5) for k, v in d.items()} for b, d in rig.angles(x).items()}
    for p, who in list(fr.get("press", {}).items()):
        if who[0] != h:
            continue
        f, pi = int(who[1]) - 1, int(p)
        pad = pads[f]
        top = P.BLACK_H if P.is_black(pi) else 0.0
        y0 = P.WHITE_L - P.BLACK_L if P.is_black(pi) else 0.0
        on = abs(pad[0] - P.key_x(pi)) < (P.BLACK_W if P.is_black(pi) else P.WHITE_W) / 2 + 0.002 and y0 - 0.002 <= pad[1] <= P.WHITE_L
        d = float(np.clip((top - pad[2]) / (P.KEY_TRAVEL * P.key_lever(pi, pad[1])), 0, 1)) if on else 0.0
        fr["keys"][p] = round(d, 3)


def clean(anim, notes, jobs=1, t0=None, t1=None, log=print):
    st, fps = anim["start"], anim["fps"]
    T = len(anim["frames"])
    j0 = 0 if t0 is None else max(0, int((t0 - st) * fps))
    j1 = T if t1 is None else min(T, int(math.ceil((t1 - st) * fps)) + 1)
    hands = [h for h in "RL" if h in anim["frames"][0]["hands"]]
    own = {h: _own_keys(anim, notes, h) for h in hands}
    pads0 = {(j, h): [list(p) for p in anim["frames"][j]["hands"][h]["pads"]] for j in range(j0, j1) for h in hands}
    for rep in range(PASSES):                       # (pressing pads are pinned where the take had them, every pass)
        tasks, pinned = [], {}
        skin = {h: [_skin_xy(anim, h, j) for j in range(T)] if SWEEP_U else None for h in hands}
        for j in range(j0, j1):
            fr = anim["frames"][j]
            for h in hands:
                H = fr["hands"][h]
                press_f = [None] * 5
                for p, who in fr.get("press", {}).items():
                    if who[0] == h:
                        press_f[int(who[1]) - 1] = int(p)
                for f in range(5):                  # coming down onto a key it owns: its pad path is kept too, so
                    if press_f[f] is None and own[h][j][f]:     # the strike keeps its timing
                        pd = pads0[(j, h)][f]
                        for p in own[h][j][f]:
                            top = P.BLACK_H if P.is_black(p) else 0.0
                            if abs(pd[0] - P.key_x(p)) < 0.02 and pd[2] - top < NEAR_OWN:
                                press_f[f] = p
                pinned[(j, h)] = press_f
                sweep = [[skin[h][k][f] for k in (j - 1, j + 1) if 0 <= k < T] if SWEEP_U and press_f[f] is None
                         else None for f in range(5)]     # (a pressing / landing finger's path is its strike)
                tasks.append((j, h, np.array(H["root"]["rot"]), np.array(H["root"]["pos"]),
                              _q_from_angles(P._rig(h), H["angles"]), fr["keys"], own[h][j], pads0[(j, h)], press_f, sweep))
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(jobs) as pool:
                res = pool.map(_frame_fix, tasks, chunksize=32)
        else:
            res = [_frame_fix(t) for t in tasks]
        fixed = {(j, h): q for j, h, q, w in res if q is not None}
        log(f"clean pass {rep}: {len(fixed)} hand-frames with skin in a key they aren't playing "
            f"(of {len(res)}; worst {1000 * max(w for *_r, w in res):.1f} mm)")
        if not fixed:
            break
        for h in hands:                             # temporal blend of the corrections, per finger
            rig = P._rig(h)
            Q = np.array([_q_from_angles(rig, anim["frames"][j]["hands"][h]["angles"]) for j in range(j0, j1)])
            Dl = np.zeros_like(Q)
            for (j, hh), q in fixed.items():
                if hh == h:                         # a fix is a nudge: a joint never swings more than MAX_DQ
                    Dl[j - j0] = np.clip(q - Q[j - j0], -MAX_DQ, MAX_DQ)
            Ds = np.zeros_like(Q)
            for f in range(5):
                a, b = rig.sl[f], rig.sl[f + 1]
                moved = np.abs(Dl[:, a:b]).max(1) > 1e-9
                if not moved.any():
                    continue
                # pin the corrected frames, and the frames where this finger presses (a correction may not bleed
                # into a strike or a hold); other frames are pulled to no change, the second difference smoothed
                press = np.array([pinned[(j, h)][f] is not None for j in range(j0, j1)])
                # next to a strike or a release a free finger is on its strike path: its correction tapers to nothing
                # over TAPER frames (no lift right before a landing, so nothing has to snap down onto a pinned strike)
                dist = np.full(len(press), 99)
                idx = np.nonzero(press)[0]
                if len(idx):
                    k_ = np.searchsorted(idx, np.arange(len(press)))
                    nxt = np.where(k_ < len(idx), idx[np.minimum(k_, len(idx) - 1)] - np.arange(len(press)), 99)
                    prv = np.where(k_ > 0, np.arange(len(press)) - idx[np.maximum(k_ - 1, 0)], 99)
                    dist = np.minimum(np.abs(nxt), np.abs(prv))
                taper = np.clip((dist - 1) / TAPER, 0.0, 1.0)
                Dl[:, a:b] *= np.where(press, 1.0 if FIX_PRESSING else 0.0, taper)[:, None]   # a pressing finger stays put
                w = np.where(press, W_PRESS, np.where(moved, W_HI, W_LO))
                Ds[:, a:b] = P._whittaker(Dl[:, a:b], w, SMOOTH_LAM)
            Ds[np.abs(Ds) < 1e-6] = 0.0
            Qn = np.clip(Q + Ds, rig.lo, rig.hi)
            for k in np.nonzero(np.abs(Ds).max(1) > 0)[0]:
                _rebuild(anim, h, j0 + k, Qn[k])
    return anim


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("notes"); ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--t0", type=float); ap.add_argument("--t1", type=float)
    o = ap.parse_args(argv)
    anim = json.load(open(o.inp))
    assert anim.get("thumb_axis", "rest") == P.THUMB_AXIS if hasattr(P, "THUMB_AXIS") else True
    notes = json.load(open(o.notes))
    clean(anim, notes, o.jobs, o.t0, o.t1)
    anim.setdefault("cleaned", {"tol": TOL, "gain_free": GAIN_FREE, "gain_press": GAIN_PRESS, "prior": PRIOR,
                                "centre": CENTRE, "passes": PASSES})
    json.dump(anim, open(o.out, "w"))
    print("wrote", o.out)


if __name__ == "__main__":
    main()

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

MESH = False            # violations and the re-solve's collisions on the rendered skin (the MPFB mesh posed by linear blend
                        # skinning, as blockout/action.py's judge); False: the rig's skin sections (the C3 cleanup)
MESH_REACH = 0.015      # m over the black key tops: a finger's mesh points higher than this at the frame are left out
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
PAD_GUARD = (0.0005, 0.002)    # m (up, across): with FIX_PRESSING "soft", a press / landing frame whose pad the blended
                        # correction lifts or slides more than this is held to no change there (and re-blended)
PAD_GUARD_DIR = False   # True: only a slide *away* from the key's centre line (or off the key) counts (for CENTRE > 0;
                        # tested S6 / M5 2026-10-18: centring didn't help on the mesh judge)
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


class _MeshSkin:
    """One hand's rendered skin, per finger: mesh vertices and triangle-edge midpoints (as the judge's), skinned from the
    rig's posed bones. points(x, root_rot) -> per finger (n, 3); chain_points(f, P, Rs, const) -> finger f's points
    with its own chain moved and every other bone held (const: their share, from const_part)."""

    def __init__(self, h):
        from .action import MESH_JSON, MESH_R
        D = json.load(open(MESH_JSON))["hands"][h]
        self.rig = rig = P._rig(h)
        sk = rig.sk
        self.r = MESH_R
        V, T = np.array(D["verts"]), np.array(D["tris"])
        E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
        bones = sorted({b for w in D["weights"] for b in w})
        Wv = np.zeros((len(V), len(bones)))
        for i, w in enumerate(D["weights"]):
            for b, x in w.items():
                Wv[i, bones.index(b)] = x
        P0 = np.concatenate([V, V[E].mean(1)])
        W0 = np.concatenate([Wv, Wv[E].mean(1)])
        dom = [bones[k] for k in W0.argmax(1)]
        self.bones = bones
        self.M0 = {b: sk.R0[b].T for b in bones}
        self.h0 = {b: sk.h0[b] for b in bones}
        self.fp = []
        for f, ch in enumerate(rig.chains):
            idx = np.array([i for i, b in enumerate(dom) if b in ch.bones])
            Wf = W0[idx]
            cols = [k for k in range(len(bones)) if Wf[:, k].max() > 1e-4]
            self.fp.append((P0[idx], {bones[k]: Wf[:, k] for k in cols}, list(ch.bones)))

    def frames(self, x, ref_rot):
        rig = self.rig
        root_rot = _rotvec(x[3:6]) @ ref_rot
        root_pos = x[:3]
        F = {"wrist": (root_rot, root_pos)}
        for ch, (bR, bH) in zip(rig.chains, rig.bases(root_rot, root_pos)):
            F[ch.base_bone] = (bR, bH)
        _rr, _rp, pts = rig.points(x, ref_rot)
        for f, ch in enumerate(rig.chains):
            for i, b in enumerate(ch.bones):
                F[b] = (rig._rots[f][i], pts[f][i])
        return F

    def _part(self, f, F, which):
        P0, Wb, chain = self.fp[f]
        acc = np.zeros_like(P0)
        for b, w in Wb.items():
            if (b in chain) != which:
                continue
            Wd, Hd = F[b]
            acc += w[:, None] * ((P0 - self.h0[b]) @ (Wd @ self.M0[b]).T + Hd)
        return acc

    def points(self, x, ref_rot):
        F = self.frames(x, ref_rot)
        return [self._part(f, F, True) + self._part(f, F, False) for f in range(5)]

    def const_part(self, x, ref_rot):
        F = self.frames(x, ref_rot)
        return [self._part(f, F, False) for f in range(5)]

    def chain_points(self, f, Pc, Rs, const):
        F = {b: (Rs[i], Pc[i]) for i, b in enumerate(self.fp[f][2])}
        return const[f] + self._part(f, F, True)


def _rotvec(v):
    from .rig.hand import rotvec
    return rotvec(v)


_SKIN = {}


def _skin(h):
    if h not in _SKIN:
        _SKIN[h] = _MeshSkin(h)
    return _SKIN[h]


def _pen_pts(Pt, r, DW, DB, own, sweep=None):
    """Mesh points (n, 3) of one finger: how far each is inside a key top it doesn't own (m), with the black-zone floor
    for a free finger (own empty); sweep = [points (n, 3) at the frame before, after]: also tested part way along."""
    if sweep:
        out = _pen_pts(Pt, r, DW, DB, own)
        for Q in sweep:
            for u in SWEEP_U:
                out = np.maximum(out, _pen_pts(Pt + u * (Q - Pt), r, DW, DB, own))
        return out
    DW2, DB2 = DW.copy(), DB.copy()
    for p in own:
        if p in P._WHITE_IDX:
            DW2[P._WHITE_IDX[p]] = 1e3
        elif p in P._BLACK_IDX:
            DB2[P._BLACK_IDX[p]] = 1e3
    low = Pt[:, 2] - r
    pen = P.key_pen_round_v(Pt[:, 0], Pt[:, 1], low, np.full(len(Pt), r), DW2, DB2, np.full(len(Pt), -1))
    if not own:
        bz = Pt[:, 1] >= P.WHITE_L - P.BLACK_L - BLACK_ZONE
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
    if MESH:
        skin = _skin(h)
        Pts = skin.points(x, root_rot)
        keep = [Pt[:, 2] < P.BLACK_H + MESH_REACH for Pt in Pts]
        for f in range(5):
            if not keep[f].any():
                continue
            sw = [Q[keep[f]] for Q in sweep[f]] if sweep[f] else None
            m = float(_pen_pts(Pts[f][keep[f]], skin.r, DW, DB, own[f], sw).max())
            worst = max(worst, m)
            if m > TOL:
                bad.append(f)
    else:
        for f, (ch, P_, R_) in enumerate(zip(rig.chains, pts, rots)):
            m = float(_pen(ch.samples(P_, R_), DW, DB, own[f], sweep[f]).max())
            worst = max(worst, m)
            if m > TOL:
                bad.append(f)
    if not bad:
        return j, h, None, worst

    def env_for(f_own):
        if MESH:
            const = skin.const_part(xn, root_rot)            # every bone but the solving finger's chain, held

            def menv(f, Pc, Rs):
                Pt = skin.chain_points(f, Pc, Rs, const)[keep[f]]
                sw = [Q[keep[f]] for Q in sweep[f]] if sweep[f] else None
                g = GAIN_PRESS if f_own[f] else GAIN_FREE
                return math.sqrt(P.RIG_W["env"] * g) * np.maximum(_pen_pts(Pt, skin.r, DW, DB, f_own[f], sw) + CLEAR, 0.0)
            menv.chain_env = True
            return menv

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


def _mesh_pts(anim, h, j):
    """Per finger: its mesh points (float32 (n, 3)) at frame j."""
    rig = P._rig(h)
    H = anim["frames"][j]["hands"][h]
    x = np.concatenate([np.array(H["root"]["pos"]), np.zeros(3), _q_from_angles(rig, H["angles"])])
    return [p.astype(np.float32) for p in _skin(h).points(x, np.array(H["root"]["rot"]))]


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
        if MESH:                                    # every finger's mesh points at every frame (the swept test)
            skin = {h: [_mesh_pts(anim, h, j) if j0 - 1 <= j <= j1 else None for j in range(T)] if SWEEP_U else None
                    for h in hands}
        else:
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
                sweep = [[skin[h][k][f] for k in (j - 1, j + 1) if 0 <= k < T and skin[h][k] is not None]
                         if SWEEP_U and press_f[f] is None
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
            hold0 = np.zeros((len(Q), 5), bool)     # (finger, frame) pinned to no change (crossing guard)
            soft = FIX_PRESSING == "soft"
            for _cg in range((2 if CROSS_GUARD else 1) + (2 if soft else 0)):
                Ds = _blend(rig, h, Q, Dl, pinned, j0, j1, hold0)
                if not CROSS_GUARD and not soft:
                    break
                bad_ = _new_crossings(rig, anim, h, j0, Q, Ds) if CROSS_GUARD else np.zeros((len(Q), 5), bool)
                if soft:
                    bad_ |= _pad_moved(rig, anim, h, j0, Q, Ds, {(j, h): pinned[(j, h)] for j in range(j0, j1)})
                if not bad_.any():
                    break
                for k in np.nonzero(bad_.any(1))[0]:          # undo both crossing fingers' fixes around there
                    for f in np.nonzero(bad_[k])[0]:
                        hold0[max(0, k - 2):k + 3, f] = True
            Ds[np.abs(Ds) < 1e-6] = 0.0
            Qn = np.clip(Q + Ds, rig.lo, rig.hi)
            for k in np.nonzero(np.abs(Ds).max(1) > 0)[0]:
                _rebuild(anim, h, j0 + k, Qn[k])
    return anim


def _blend(rig, h, Q, Dl, pinned, j0, j1, hold0):
    """Per finger, the corrections eased in time (Whittaker): corrected frames pinned to their fix, press / landing
    frames and crossing-guard holds pinned to no change, a free finger's fix tapered around strikes."""
    Dl = Dl.copy(); Ds = np.zeros_like(Q)
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
        if FIX_PRESSING == "soft":                  # a pressing / landing finger's own fix (pad pinned in its re-solve:
            Dl[:, a:b] *= np.where(press, 1.0, taper)[:, None]     # it curls / rolls off the neighbour) is eased in
            Dl[hold0[:, f], a:b] = 0.0                              # time like a free one; the pad guard holds any
            w = np.where(hold0[:, f], W_PRESS, np.where(moved, W_HI, W_LO))   # frame where that loses the press
            Ds[:, a:b] = P._whittaker(Dl[:, a:b], w, SMOOTH_LAM)
            continue
        Dl[:, a:b] *= np.where(press, 1.0 if FIX_PRESSING else 0.0, taper)[:, None]   # a pressing finger stays put
        Dl[hold0[:, f], a:b] = 0.0
        w = np.where(press | hold0[:, f], W_PRESS, np.where(moved, W_HI, W_LO))
        Ds[:, a:b] = P._whittaker(Dl[:, a:b], w, SMOOTH_LAM)
    return Ds


def _pad_moved(rig, anim, h, j0, Q, Ds, pinned):
    """(frames, 5) bool: press / landing frames whose pad the blended correction lifts > PAD_GUARD[0] or slides across
    the key > PAD_GUARD[1]."""
    out = np.zeros((len(Q), 5), bool)
    for k in np.nonzero(np.abs(Ds).max(1) > 1e-6)[0]:
        pf = pinned[(j0 + k, h)]
        if all(p is None for p in pf):
            continue
        H = anim["frames"][j0 + k]["hands"][h]
        R0, p0 = np.array(H["root"]["rot"]), np.array(H["root"]["pos"])
        pads = []
        for q in (Q[k], np.clip(Q[k] + Ds[k], rig.lo, rig.hi)):
            _rr, _rp, pts = rig.points(np.concatenate([p0, np.zeros(3), q]), R0)
            pads.append(np.array(rig.effectors(pts, rig._rots)))
        for f in range(5):                          # (sliding toward its key's centre line is allowed)
            if pf[f] is None:
                continue
            kx = P.key_x(pf[f])
            if PAD_GUARD_DIR:
                away = abs(pads[1][f][0] - kx) - abs(pads[0][f][0] - kx)
                off = abs(pads[1][f][0] - kx) > (P.BLACK_W if P.is_black(pf[f]) else P.WHITE_W) / 2 - 0.002
            else:
                away, off = abs(pads[1][f][0] - pads[0][f][0]), False
            if pads[1][f][2] - pads[0][f][2] > PAD_GUARD[0] or away > PAD_GUARD[1] or off:
                out[k, f] = True
    return out


def _new_crossings(rig, anim, h, j0, Q, Ds):
    """(frames, 5) bool: fingers whose tips cross (index..pinky, > 4 mm out of order) after the blend and didn't before."""
    sgn = 1.0 if h == "R" else -1.0
    out = np.zeros((len(Q), 5), bool)
    for k in np.nonzero(np.abs(Ds).max(1) > 1e-6)[0]:
        H = anim["frames"][j0 + k]["hands"][h]
        R0, p0 = np.array(H["root"]["rot"]), np.array(H["root"]["pos"])
        xs = []
        for q in (Q[k], np.clip(Q[k] + Ds[k], rig.lo, rig.hi)):
            _rr, _rp, pts = rig.points(np.concatenate([p0, np.zeros(3), q]), R0)
            xs.append(np.array([pts[f][3][0] for f in range(1, 5)]) * sgn)
        for f in range(3):
            if xs[1][f + 1] - xs[1][f] < -0.004 and not xs[0][f + 1] - xs[0][f] < -0.004:
                out[k, f + 1] = out[k, f + 2] = True
    return out


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

"""Whole-hand pose solver: the hand's root (wrist bone) pose and every finger joint, solved together.

    rig = HandRig("R")
    pose = rig.solve(ref_rot, ref_pos, targets, weights, prev=pose_prev, env=env, forearm=(elbow, ...))

Unknowns: root translation (3), root rotation (3, a rotation vector applied to ref_rot) and each chain's joint
angles. Residuals (all squared and summed, Levenberg-Marquardt, angles projected onto their limits):
  * fingertip -> target for every finger that has one, times its weight (pressing ~1, a free finger's float target
    ~0.2: it goes there if nothing else matters),
  * joint comfort (distance from the relaxed rest pose, per range) and the DIP-PIP tendon coupling,
  * the root's distance from the motion plan's reference pose (a soft prior: the hand stays on its planned path
    unless a finger needs it elsewhere),
  * the wrist joint against the forearm (flexion/extension and deviation beyond a comfortable band),
  * env(points, radii) -> penalty: collisions with the instrument and the other hand, supplied by the caller.
Warm-started from the previous frame, so the pose family carries from frame to frame.
"""
import json, math, os

import numpy as np

from .ik import Chain
from .skeleton import Skeleton, _rot, D


def _furelise_cal():
    """blockout/rig/furelise_cal.json (blockout.furelise.calibrate), unless RIG_NO_FURELISE=1."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "furelise_cal.json")
    if os.environ.get("RIG_NO_FURELISE") == "1" or not os.path.exists(p):
        return None
    return json.load(open(p))


def rotvec(v):
    a = float(np.linalg.norm(v))
    return np.eye(3) if a < 1e-12 else _rot(v / a, a)


class HandRig:
    def __init__(self, side):
        self.side = side
        self.sk = Skeleton(side)
        self.chains = [Chain(self.sk, f) for f in range(1, 6)]
        self.nq = [len(c.lo) for c in self.chains]
        self.lo = np.concatenate([c.lo for c in self.chains]); self.hi = np.concatenate([c.hi for c in self.chains])
        self.rng = self.hi - self.lo
        self.q_rest = np.concatenate([c.q_rest for c in self.chains])
        self.sl = np.cumsum([0] + self.nq)
        sk = self.sk
        # the hand's rest frame relative to the root bone, to turn a planned hand frame into a root pose
        self.M_rest = sk.rest_frame                      # rows: across, forward, up (world, at rest)
        self.c_rest = np.mean([sk.h0[f"finger{f}-1"] for f in range(2, 6)], 0)
        self._centre_abduction()
        self.lo = np.concatenate([c.lo for c in self.chains]); self.hi = np.concatenate([c.hi for c in self.chains])
        self.rng = self.hi - self.lo
        self.q_rest = np.concatenate([c.q_rest for c in self.chains])
        cpath = os.environ.get("RIG_COMFORT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfort.json")
        if os.path.exists(cpath):                     # relaxed pose calibrated to measured pianists (calibrate.py)
            q = np.clip(np.array(json.load(open(cpath))[side]), self.lo, self.hi)
            if len(q) == len(self.q_rest):
                self.q_rest = q
                for ch, a, b in zip(self.chains, self.sl[:-1], self.sl[1:]):
                    ch.q_rest = q[a:b].copy()
                    if ch.k_dip and q[a + ch.i_pip] > 0.05:   # the tendon's DIP:PIP ratio, per finger, from people
                        ch.k_dip = float(np.clip(q[a + ch.i_dip] / q[a + ch.i_pip], 0.4, 1.0))
        cal = _furelise_cal()
        if cal and "k_dip" in cal:                 # measured on concert pianists (FürElise): the end joint stays
            for ch, k in zip(self.chains[1:], cal["k_dip"]):    # nearly straight while the PIP bends
                ch.k_dip = max(float(k), 1e-3)
        # comfort is measured in units of how much people actually vary each joint while playing (human IQR / 1.35)
        self.sigma = self.rng.copy()
        try:
            ref = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "human_ref.json")))
            pf, ms = ref["per_finger"], ref["measures"]
            iqr = lambda d: math.radians(max(4.0, (d["p75"] - d["p25"]) / 1.35))
            names = ["index", "middle", "ring", "pinky"]
            for ci, (ch, a) in enumerate(zip(self.chains, self.sl[:-1])):
                for i, (b, n, *_r) in enumerate(ch.dofs):
                    kind = self.sk.kind[b]
                    if ci == 0:
                        key = {"cmc": "thumb_abd", "tmcp": "thumb_mcp_flex", "tip": "thumb_ip_flex"}[kind]
                        self.sigma[a + i] = iqr(ms[key]["pooled"])
                    elif n == "abd":
                        self.sigma[a + i] = iqr(ms[("abd_im", "abd_mr", "abd_mr", "abd_rp")[ci - 1]]["pooled"])
                    else:
                        meas = {"mcp": "mcp_flex", "mcp_edge": "mcp_flex", "pip": "pip_flex", "dip": "dip_flex"}[kind]
                        self.sigma[a + i] = iqr(pf[f"{meas}.{names[ci - 1]}"])
        except (OSError, KeyError):
            pass

    def anatomical_abduction(self, root_rot, root_pos, pts, f):
        """Finger f's (2-5) abduction in degrees as anatomy measures it: the proximal phalanx against its
        metacarpal (wrist -> MCP), both seen in the palm plane."""
        up = root_rot @ (self.sk.R0["wrist"].T @ self.M_rest[2])
        n = lambda v: v / np.linalg.norm(v)
        mc = n(pts[f - 1][0] - root_pos); s0 = n(pts[f - 1][1] - pts[f - 1][0])
        a = n(mc - up * (mc @ up)); b = n(s0 - up * (s0 @ up))
        return math.degrees(math.atan2(float(np.cross(a, b) @ up), float(a @ b)))

    def _centre_abduction(self):
        """MakeHuman's rest fingers are already a little spread; put each MCP's abduction range around the
        anatomical zero (finger in line with its metacarpal) instead of around the rest pose."""
        Rr, pr = self.sk.rest_root()
        q = np.concatenate([c.q_rest for c in self.chains])
        sl = np.cumsum([0] + [len(c.lo) for c in self.chains])
        x = np.concatenate([pr, np.zeros(3), q])
        self.lo = np.concatenate([c.lo for c in self.chains]); self.hi = np.concatenate([c.hi for c in self.chains])
        self.sl = sl
        for fi, ch in enumerate(self.chains[1:], start=1):
            i = [k for k, d in enumerate(ch.dofs) if d[1] == "abd"][0]
            _, _, pts = self.points(x, Rr)
            m0 = self.anatomical_abduction(Rr, pr, pts, fi + 1)
            x2 = x.copy(); x2[6 + sl[fi] + i] += 0.05
            _, _, pts2 = self.points(x2, Rr)
            gain = (self.anatomical_abduction(Rr, pr, pts2, fi + 1) - m0) / math.degrees(0.05)
            lim = math.degrees(ch.hi[i])                    # symmetric anatomical limit
            lo, hi = sorted(((-lim - m0) / gain, (lim - m0) / gain))
            ch.lo[i], ch.hi[i] = math.radians(lo), math.radians(hi)
            ch.rng = ch.hi - ch.lo
            ch.q_rest[i] = float(np.clip(math.radians(-m0 / gain), ch.lo[i], ch.hi[i]))   # comfort: in line

    def root_from_hand_frame(self, M, c):
        """Root (wrist bone) rotation and head for a hand frame M (rows across/forward/up) at knuckle centre c."""
        R = M.T @ self.M_rest
        return R @ self.sk.R0["wrist"], c + R @ (self.sk.h0["wrist"] - self.c_rest)

    def bases(self, root_rot, root_pos):
        """Posed frame and head of each chain's base bone (metacarpal or wrist) for a root pose."""
        sk = self.sk
        out = []
        for ch in self.chains:
            b = ch.base_bone
            if b == "wrist":
                out.append((root_rot, root_pos))
            else:
                rel = sk.R0["wrist"].T @ sk.R0[b]
                out.append((root_rot @ rel, root_pos + root_rot @ (sk.R0["wrist"].T @ (sk.h0[b] - sk.h0["wrist"]))))
        return out

    def points(self, x, ref_rot):
        root_rot = rotvec(x[3:6]) @ ref_rot
        root_pos = x[:3]
        q = x[6:]
        pts, rots = [], []
        for ch, (bR, bH), a, b in zip(self.chains, self.bases(root_rot, root_pos), self.sl[:-1], self.sl[1:]):
            P, Rs = ch.fk(bR, bH, q[a:b])
            pts.append(P); rots.append(Rs)
        self._rots = rots
        return root_rot, root_pos, pts

    def effectors(self, pts, rots):
        return [ch.effector(P, R) for ch, P, R in zip(self.chains, pts, rots)]

    def residuals(self, x, ref_rot, ref_pos, targets, weights, env, forearm, W, x_last=None):
        root_rot, root_pos, pts = self.points(x, ref_rot)
        rots = self._rots
        eff = self.effectors(pts, rots)
        q = x[6:]
        r = []
        for f in range(5):
            if targets[f] is not None:
                r.append(np.asarray(weights[f], float) * (eff[f] - targets[f]))   # scalar or per-axis weights
        cw = np.ones(len(q))                                  # a pressing finger may leave its relaxed pose to reach
        for f, (a, b) in enumerate(zip(self.sl[:-1], self.sl[1:])):
            if targets[f] is not None and float(np.max(weights[f])) >= 0.99:
                cw[a:b] = math.sqrt(W.get("comfort_press", 1.0))
        scale = self.sigma if W.get("comfort_sigma") else self.rng      # human-spread scaling: experimental, off
        q_rest = self.q_rest
        if W.get("rest_override"):
            q_rest = q_rest.copy()
            for f, v in W["rest_override"].items():
                q_rest[self.sl[f]:self.sl[f + 1]] = v
        r.append(math.sqrt(W["comfort"]) * cw * (q - q_rest) / scale)
        for ch, a in zip(self.chains, self.sl[:-1]):
            if ch.k_dip:
                r.append([math.sqrt(W["couple"]) * (q[a + ch.i_dip] - ch.k_dip * q[a + ch.i_pip])])
        if x_last is not None and W.get("smooth", 0) > 0:   # joints and the hand can't teleport between frames
            r.append(math.sqrt(W["smooth"]) * (x[6:] - x_last[6:]) / self.sigma)
            r.append(math.sqrt(W["smooth"]) * 20.0 * (x[:3] - x_last[:3]))
        r.append(math.sqrt(W["root_pos"]) * (root_pos - ref_pos))
        r.append(math.sqrt(W["root_rot"]) * x[3:6])
        if forearm is not None:                          # wrist joint: the hand's forward axis against the forearm
            el, band = forearm
            fa = root_pos - np.asarray(el, float); fa /= np.linalg.norm(fa)
            Rh = root_rot @ self.sk.R0["wrist"].T
            hu, ha = Rh @ self.M_rest[2], Rh @ self.M_rest[0]
            flex = math.asin(float(np.clip(fa @ hu, -1, 1)))        # + wrist flexed (hand below the forearm line)
            dev = math.asin(float(np.clip(fa @ ha, -1, 1)))         # radial/ulnar deviation
            r.append([math.sqrt(W["wrist"]) * max(0.0, abs(flex) - band[0]),
                      math.sqrt(W["wrist"]) * max(0.0, abs(dev) - band[1])])
        if env is not None:
            pen = env([ch.samples(P, R) for ch, P, R in zip(self.chains, pts, rots)], root_rot, root_pos)
            if np.ndim(pen):                             # per-sample penetrations: smooth least-squares residuals
                r.append(math.sqrt(W["env"]) * np.asarray(pen, float))
            else:
                r.append([math.sqrt(W["env"] * max(pen, 0.0))])
        return np.concatenate([np.ravel(v) for v in r]), (root_rot, root_pos, pts, eff)

    def solve(self, ref_rot, ref_pos, targets, weights, prev=None, env=None, forearm=None, iters=12, W=None,
              last=None, fix_root=False):
        """targets: 5 world points or None; weights: 5 floats. prev: x from the previous frame (warm start).
        fix_root: the wrist is given (prev[:6]) and only the finger joints move -- the hand hangs off its wrist.
        Returns x, (root_rot, root_pos, points per finger), misses per finger (m, None if no target)."""
        W = {**dict(comfort=4e-5, couple=5e-5, root_pos=0.5, root_rot=2e-4, wrist=1e-3, env=1.0), **(W or {})}
        iters = int(W.pop("iters", iters))
        ref_rot = np.asarray(ref_rot, float); ref_pos = np.asarray(ref_pos, float)
        targets = [None if t is None else np.asarray(t, float) for t in targets]
        if prev is not None:
            x = np.array(prev, float)
            if not fix_root:
                x[:3] = 0.5 * x[:3] + 0.5 * ref_pos
                x[3:6] *= 0.5
        else:
            x = np.concatenate([ref_pos, np.zeros(3), self.q_rest])
        lo = np.concatenate([np.full(6, -np.inf), self.lo]); hi = np.concatenate([np.full(6, np.inf), self.hi])
        x = np.clip(x, lo, hi)
        args = (ref_rot, ref_pos, targets, weights, env, forearm, W, None if last is None else np.asarray(last, float))
        r, pose = self.residuals(x, *args)
        c = float(r @ r); lam = 1e-3
        free = np.arange(6 if fix_root else 0, len(x))
        n = len(free)
        for _ in range(iters):
            J = np.empty((len(r), n))
            for k, i in enumerate(free):
                dx = x.copy(); h = 1e-5 if dx[i] + 1e-5 <= hi[i] else -1e-5
                dx[i] += h
                J[:, k] = (self.residuals(dx, *args)[0] - r) / h
            A = J.T @ J; g = J.T @ r
            ok = False
            for _t in range(6):
                step = np.zeros(len(x))
                step[free] = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
                xn = np.clip(x + step, lo, hi)
                rn, pn = self.residuals(xn, *args)
                cn = float(rn @ rn)
                if cn < c:
                    x, r, c, pose, ok = xn, rn, cn, pn, True
                    lam = max(lam / 3, 1e-7)
                    break
                lam *= 4
            if not ok or c < 1e-12:
                break
        misses = [None if targets[f] is None else float(np.linalg.norm(pose[3][f] - targets[f])) for f in range(5)]
        return x, pose, misses

    def finger_residuals(self, f, qf, x, ref_rot, target, weight, env, W, x_last):
        """Finger f alone, its base fixed by the root in x: pad target, comfort, coupling, smoothness, collisions."""
        a, b = self.sl[f], self.sl[f + 1]
        ch = self.chains[f]
        root_rot = rotvec(x[3:6]) @ ref_rot
        bR, bH = self.bases(root_rot, x[:3])[f]
        P, Rs = ch.fk(bR, bH, qf)
        r = []
        pressing = target is not None and float(np.max(weight)) >= 0.99
        if target is not None:
            r.append(np.asarray(weight, float) * (ch.effector(P, Rs) - target))
        cw = math.sqrt(W.get("comfort_press", 1.0)) if pressing else 1.0
        scale = (self.sigma if W.get("comfort_sigma") else self.rng)[a:b]
        rest = W.get("rest_override", {}).get(f, self.q_rest[a:b])
        if f in W.get("rest_override", {}):
            cw *= math.sqrt(W.get("rest_override_w", 30.0))   # an overridden pose (the idle thumb's tuck) is held
        r.append(math.sqrt(W["comfort"]) * cw * (qf - rest) / scale)
        if ch.k_dip:
            r.append([math.sqrt(W["couple"]) * (qf[ch.i_dip] - ch.k_dip * qf[ch.i_pip])])
        if x_last is not None and W.get("smooth", 0) > 0:
            r.append(math.sqrt(W["smooth"]) * (qf - x_last[6 + a:6 + b]) / self.sigma[a:b])
        if env is not None:
            smp = [[] for _ in range(5)]
            smp[f] = ch.samples(P, Rs)
            pen = env(smp, root_rot, x[:3])
            r.append(math.sqrt(W["env"]) * np.asarray(pen, float) if np.ndim(pen) else [math.sqrt(W["env"] * max(pen, 0.0))])
        return np.concatenate([np.ravel(v) for v in r])

    def solve_fingers(self, ref_rot, x0, targets, weights, env=None, W=None, last=None, iters=20):
        """The root is set (x0[:6]): each finger is its own small solve (one finger's trouble can't freeze the
        others), from the previous pose and, if that stalls, from the relaxed pose too; the better one wins."""
        W = {**dict(comfort=4e-5, couple=5e-5, env=1.0), **(W or {})}
        x = np.array(x0, float)
        last = None if last is None else np.asarray(last, float)
        for f in range(5):
            a, b = self.sl[f], self.sl[f + 1]
            lo, hi = self.lo[a:b], self.hi[a:b]
            args = (x, ref_rot, targets[f], weights[f], env, W, last)
            best = None
            for seed in (x[6 + a:6 + b], self.q_rest[a:b]):
                q = np.clip(np.array(seed, float), lo, hi)
                r = self.finger_residuals(f, q, *args); c = float(r @ r); lam = 1e-3; moved = False
                for _ in range(iters):
                    J = np.empty((len(r), b - a))
                    for i in range(b - a):
                        dq = q.copy(); h = 1e-5 if dq[i] + 1e-5 <= hi[i] else -1e-5
                        dq[i] += h
                        J[:, i] = (self.finger_residuals(f, dq, *args) - r) / h
                    A = J.T @ J; g = J.T @ r
                    ok = False
                    for _t in range(8):
                        qn = np.clip(q + np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g), lo, hi)
                        rn = self.finger_residuals(f, qn, *args); cn = float(rn @ rn)
                        if cn < c:
                            q, r, c, ok, moved = qn, rn, cn, True, True
                            lam = max(lam / 3, 1e-7)
                            break
                        lam *= 4
                    if not ok or c < 1e-12:
                        break
                if best is None or c < best[0] - 1e-12:
                    best = (c, q)
                if moved:                                # the warm start went somewhere: no restart needed
                    break
            x[6 + a:6 + b] = best[1]
        return x

    def angles(self, x):
        out = {}
        q = x[6:]
        for ch, a, b in zip(self.chains, self.sl[:-1], self.sl[1:]):
            out.update(ch.angles(q[a:b]))
        return out

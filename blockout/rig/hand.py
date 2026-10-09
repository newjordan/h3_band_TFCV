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
import math

import numpy as np

from .ik import Chain
from .skeleton import Skeleton, _rot, D


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
        pts = []
        for ch, (bR, bH), a, b in zip(self.chains, self.bases(root_rot, root_pos), self.sl[:-1], self.sl[1:]):
            pts.append(ch.fk(bR, bH, q[a:b])[0])
        return root_rot, root_pos, pts

    def residuals(self, x, ref_rot, ref_pos, targets, weights, env, forearm, W):
        root_rot, root_pos, pts = self.points(x, ref_rot)
        q = x[6:]
        r = []
        for f in range(5):
            if targets[f] is not None:
                r.append(weights[f] * (pts[f][-1] - targets[f]))
        r.append(math.sqrt(W["comfort"]) * (q - self.q_rest) / self.rng)
        for ch, a in zip(self.chains, self.sl[:-1]):
            if ch.k_dip:
                r.append([math.sqrt(W["couple"]) * (q[a + ch.i_dip] - ch.k_dip * q[a + ch.i_pip])])
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
            pen = env(pts, [ch.radii for ch in self.chains], root_rot, root_pos)
            r.append([math.sqrt(W["env"] * max(pen, 0.0))])
        return np.concatenate([np.ravel(v) for v in r]), (root_rot, root_pos, pts)

    def solve(self, ref_rot, ref_pos, targets, weights, prev=None, env=None, forearm=None, iters=12, W=None):
        """targets: 5 world points or None; weights: 5 floats. prev: x from the previous frame (warm start).
        Returns x, (root_rot, root_pos, points per finger), misses per finger (m, None if no target)."""
        W = {**dict(comfort=4e-5, couple=5e-5, root_pos=0.5, root_rot=2e-4, wrist=1e-3, env=1.0), **(W or {})}
        ref_rot = np.asarray(ref_rot, float); ref_pos = np.asarray(ref_pos, float)
        targets = [None if t is None else np.asarray(t, float) for t in targets]
        if prev is not None:
            x = np.array(prev, float)
            x[:3] = 0.5 * x[:3] + 0.5 * ref_pos
            x[3:6] *= 0.5
        else:
            x = np.concatenate([ref_pos, np.zeros(3), self.q_rest])
        lo = np.concatenate([np.full(6, -np.inf), self.lo]); hi = np.concatenate([np.full(6, np.inf), self.hi])
        x = np.clip(x, lo, hi)
        args = (ref_rot, ref_pos, targets, weights, env, forearm, W)
        r, pose = self.residuals(x, *args)
        c = float(r @ r); lam = 1e-3
        n = len(x)
        for _ in range(iters):
            J = np.empty((len(r), n))
            for i in range(n):
                dx = x.copy(); h = 1e-5 if dx[i] + 1e-5 <= hi[i] else -1e-5
                dx[i] += h
                J[:, i] = (self.residuals(dx, *args)[0] - r) / h
            A = J.T @ J; g = J.T @ r
            ok = False
            for _t in range(6):
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
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
        misses = [None if targets[f] is None else float(np.linalg.norm(pose[2][f][-1] - targets[f])) for f in range(5)]
        return x, pose, misses

    def angles(self, x):
        out = {}
        q = x[6:]
        for ch, a, b in zip(self.chains, self.sl[:-1], self.sl[1:]):
            out.update(ch.angles(q[a:b]))
        return out

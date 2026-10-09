"""Generic chain IK in joint-angle space for the hand skeleton.

A Chain is a run of bones from a fixed base (its parent's posed frame) to a fingertip. Its DOFs are the joints'
(flex, abd) from skeleton.LIMITS; the solver minimises

    |tip - target|^2  +  w_comfort * sum((q - q_rest)/range)^2  +  w_couple * (DIP - k*PIP)^2  +  w_coll * penalty(points)

with box limits on every DOF, by damped least squares (projected onto the limits) from several seeds: the previous
frame's angles (temporal coherence: the same pose family frame to frame), the relaxed pose, and a curled pose.
The best seed by cost wins; the miss (m) is reported, never hidden by folding or stretching.
"""
import math

import numpy as np

from .skeleton import FINGERS, METACARPAL, Skeleton, _rot, D


class Chain:
    def __init__(self, sk: Skeleton, finger):
        self.sk, self.f = sk, finger
        self.bones = FINGERS[finger - 1]
        self.dofs = []                                  # (bone, dof name, axis(local), lo, hi)
        for b in self.bones:
            lim = sk.limits(b)
            for name in ("abd", "flex"):
                if name in lim and name in sk.axes[b]:
                    lo, hi = lim[name]
                    self.dofs.append((b, name, sk.axes[b][name], lo, hi))
        self.lo = np.array([d[3] for d in self.dofs]); self.hi = np.array([d[4] for d in self.dofs])
        self.rng = self.hi - self.lo
        # comfort centre: a relaxed, curved hand (fingers: MCP 20, PIP 40, DIP ~2/3 PIP; the thumb at its rest)
        neutral = {("mcp", "flex"): 20, ("mcp_edge", "flex"): 20, ("pip", "flex"): 40, ("dip", "flex"): 27}
        self.q_rest = np.clip(np.array([D(neutral.get((sk.kind[b], n), 0.0)) for b, n, *_r in self.dofs]),
                              self.lo, self.hi)
        self.base_bone = sk.parent[self.bones[0]]
        self.rel = [sk.R0[sk.parent[b]].T @ sk.R0[b] for b in self.bones]
        self.off = [sk.R0[sk.parent[b]].T @ (sk.h0[b] - sk.h0[sk.parent[b]]) for b in self.bones]
        self.L = [sk.L[b] for b in self.bones]
        self.radii = [[r if r else 0.008 for r in sk.radii[b]] for b in self.bones]
        # skin cross-sections: the bone runs near the back of the finger, the pad sticks out on the palm side
        up = sk.rest_frame[2]
        self.palm_sign = [1.0 if sk.R0[b][:, 2] @ -up > 0 else -1.0 for b in self.bones]
        self.sect = []
        for b in self.bones:
            sec = sk.section.get(b) or [[r, r, r] for r in self.radii[self.bones.index(b)]]
            sec = [x if x else sec[1] for x in sec]
            self.sect.append(np.array(sec, float))
        # effector: the fingertip pad that meets the key (the thumb: the corner of its tip facing the fingers)
        w, zp, zm = self._sect_at(2, 0.8)
        palm = zp if self.palm_sign[2] > 0 else zm
        side = 0.0
        if finger == 1:
            ac = sk.rest_frame[0]
            side = (0.45 * w) * (1.0 if sk.R0[self.bones[2]][:, 0] @ ac > 0 else -1.0)
            palm *= 0.6
        self.eff_local = np.array([side, 0.8 * sk.L[self.bones[2]], self.palm_sign[2] * 0.85 * palm])
        flex = [i for i, d in enumerate(self.dofs) if d[1] == "flex"]
        self.i_pip, self.i_dip = flex[1], flex[2]
        self.k_dip = 0.0 if finger == 1 else 0.67          # tendon coupling (fingers 2-5 only)

    def _sect_at(self, i, u):
        S = self.sect[i]
        if u <= 0.5:
            t = max(0.0, (u - 0.1) / 0.4); return S[0] + (S[1] - S[0]) * t
        t = min(1.0, (u - 0.5) / 0.4); return S[1] + (S[2] - S[1]) * t

    def support(self, pts, rots, i, u):
        """Lowest skin point of bone i's cross-section at u (an off-centre ellipse in the bone's x-z plane)."""
        R = rots[i]
        w, zp, zm = self._sect_at(i, min(u, 0.95))
        c = pts[i] + R[:, 1] * self.L[i] * u + R[:, 2] * (zp - zm) / 2
        h = (zp + zm) / 2
        bx, bz = R[2, 0], R[2, 2]
        s = math.sqrt((w * bx) ** 2 + (h * bz) ** 2) + 1e-12
        return c - R[:, 0] * (w * w * bx / s) - R[:, 2] * (h * h * bz / s)

    def effector(self, pts, rots):
        """The fingertip pad that meets the key: the lowest skin of the end of the finger (soft minimum over the
        last part of the end bone, so it slides smoothly along the pad as the finger tilts)."""
        P = [self.support(pts, rots, 2, u) for u in (0.55, 0.7, 0.85, 1.0)]
        z = np.array([p[2] for p in P])
        w = np.exp(-(z - z.min()) / 0.0015)
        return (w[:, None] * np.array(P)).sum(0) / w.sum()

    def samples(self, pts, rots, us=(0.2, 0.5, 0.8, 1.0)):
        """Skin cross-sections along every bone: (bone index, u, centre, lowest point z, half-width)."""
        out = []
        for i in range(3):
            R = rots[i]
            for u in us:
                w, zp, zm = self._sect_at(i, min(u, 0.95))
                p = pts[i] + R[:, 1] * self.L[i] * u
                c = p + R[:, 2] * (zp - zm) / 2               # the section spans -zm .. +zp along the bone's z
                h = (zp + zm) / 2
                low = c[2] - math.sqrt((w * R[2, 0]) ** 2 + (h * R[2, 2]) ** 2)   # == support(...)[2]
                out.append((i, u, c, low, w))
        return out

    def angles(self, q):
        out = {b: {} for b in self.bones}
        for v, (b, name, *_r) in zip(q, self.dofs):
            out[b][name] = float(v)
        return out

    def fk(self, base_rot, base_head, q):
        """Joint points [base, j1, j2, tip] and per-bone world rotations, from the base bone's posed frame."""
        W, Hd = base_rot, base_head
        pts, rots = [], []
        k = 0
        for i, b in enumerate(self.bones):
            Hd = Hd + W @ self.off[i]
            R = self.rel[i]
            nd = sum(1 for d in self.dofs if d[0] == b)
            for j in range(nd):
                _b, _n, ax, *_r = self.dofs[k + j]
                R = R @ _rot(ax, q[k + j])
            k += nd
            W = W @ R
            pts.append(Hd); rots.append(W)
            if i == len(self.bones) - 1:
                pts.append(Hd + W[:, 1] * self.L[i])
            else:
                Hd = Hd                                      # next bone's offset is applied from this frame
        return pts, rots

    def residuals(self, q, base_rot, base_head, target, w_comfort, w_couple, coll):
        pts, rots = self.fk(base_rot, base_head, q)
        r = [self.effector(pts, rots) - target, math.sqrt(w_comfort) * (q - self.q_rest) / self.rng]
        if self.k_dip:
            r.append([math.sqrt(w_couple) * (q[self.i_dip] - self.k_dip * q[self.i_pip])])
        if coll is not None:
            r.append([math.sqrt(max(coll(pts, self.radii), 0.0))])
        return np.concatenate([np.ravel(x) for x in r]), pts

    def solve(self, base_rot, base_head, target, q_prev=None, w_comfort=2e-6, w_couple=4e-6, coll=None, iters=30):
        """Levenberg-Marquardt over the residual vector, projected onto the joint limits, from several seeds."""
        target = np.asarray(target, float)
        seeds = [self.q_rest.copy(), np.clip(self.lo + 0.45 * self.rng, self.lo, self.hi)]
        if q_prev is not None:
            seeds.insert(0, np.asarray(q_prev, float))
        best = None
        n = len(self.lo)
        for q in seeds:
            q = np.clip(q, self.lo, self.hi)
            r, pts = self.residuals(q, base_rot, base_head, target, w_comfort, w_couple, coll)
            c = float(r @ r); lam = 1e-3
            for _ in range(iters):
                J = np.empty((len(r), n))
                for i in range(n):
                    dq = q.copy(); h = 1e-5 if q[i] + 1e-5 <= self.hi[i] else -1e-5
                    dq[i] += h
                    J[:, i] = (self.residuals(dq, base_rot, base_head, target, w_comfort, w_couple, coll)[0] - r) / h
                A = J.T @ J; g = J.T @ r
                ok = False
                for _t in range(6):
                    step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
                    qn = np.clip(q + step, self.lo, self.hi)
                    rn, pn = self.residuals(qn, base_rot, base_head, target, w_comfort, w_couple, coll)
                    cn = float(rn @ rn)
                    if cn < c:
                        q, r, c, pts, ok = qn, rn, cn, pn, True
                        lam = max(lam / 3, 1e-7)
                        break
                    lam *= 4
                if not ok or c < 1e-12:
                    break
            if best is None or c < best[0]:
                best = (c, q, pts)
        c, q, pts = best
        return q, pts, float(np.linalg.norm(pts[-1] - target))

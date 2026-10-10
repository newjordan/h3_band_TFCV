"""Hands for the drum blockout: the rigged MPFB2 hand (blockout/rig, the piano's hand) holding a stick.

A drummer's grip doesn't change from stroke to stroke, so the hand is one pose in the stick's frame, solved once
per side and carried by the stick. In the hand's own frame (x across from the thumb side to the pinky side, y from
the wrist to the knuckles, z the back of the hand) the stick runs under the palm from the heel on the pinky side
(HEEL) to under the index finger's first phalanx (INDEX_U along it), GAP under the palm's skin. Fingers 2-5 close
on it like a grasp, from straight: MCP, PIP and DIP bend together; a joint stops once a bone beyond it touches the
stick, the joints past that bone carry on, so the finger wraps the stick (the DIP no further than its tendon
share of the PIP plus DIP_SLACK).
Contact is the rig's skin cross-sections against the stick.
The fulcrum (the frame's "grip") is where the index finger's middle phalanx crosses the stick; the thumb pad
presses on the stick there from the thumb side, kept out of it.

In the kit the stick's frame is d along the stick toward the tip, n the back of the hand (the world up square to
the stick, turned GRIP_ROLL toward the pinky side, between German and American grip) and a across it, away from
the body's centre. The arm reaches for this hand's wrist (wrist()), the wrist bone's head.

openness 0 is the grip, 1 an open hand (OPEN of the way to the rig's relaxed pose): a stick toss opens the hand
after the release and closes it on the catch. add() gives each frame's hand "root" (the wrist bone's world rotation and
head), "angles" (every finger joint, as blockout/rig poses them; blender_drums.py drives the skinned hand with
them) and "fingers" (5 chains of 4 points, thumb first, for checks and the capsule fallback).
"""
import math

import numpy as np

from .rig.hand import HandRig

STICK_R = 0.0075
HEEL = (0.040, -0.065)          # m, (x, y) in the hand's frame where the stick leaves the hand on the pinky side
INDEX_U = 0.0                   # ... and where along the index finger's first phalanx it passes under it
DIP_SLACK = math.radians(12)    # how far a DIP may curl past its tendon share of the PIP
GAP = 0.001                     # m between the stick and the palm's skin
GRIP_ROLL = math.radians(25)
WRIST_DEV = math.radians(30)    # ulnar deviation: the hand bent toward the pinky side off the forearm's line
DEV_OK = (math.radians(-35), math.radians(15))      # wrist deviation that costs nothing (- ulnar, + radial) ...
FLEX_OK = (math.radians(-40), math.radians(40))     # ... and flexion (+ the hand bent toward the palm)
THUMB_AT = math.radians(35)     # where the thumb pad meets the stick: from the thumb side up toward the back
OPEN = 0.6                      # an open hand: this far from the grip to the rig's relaxed pose
STEP = math.radians(0.5)
OPEN_S, CLOSE_S = 0.06, 0.10    # s to open after a release, to close before a catch
PALM = (("wrist", (0.1, 0.5, 0.9)), *((f"metacarpal{m}", (0.5, 0.7, 0.9)) for m in range(1, 5)))


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _sink(c, R, w, h, p0, s):
    """How far a skin cross-section (centre c, bone frame R, half-width w along x, half-height h along z) sinks
    into the stick (axis through p0 along s), m; <= 0 when clear."""
    v = c - p0
    v = v - (v @ s) * s
    dist = float(np.linalg.norm(v))
    u = -v / max(dist, 1e-12)
    return STICK_R + math.hypot(w * float(u @ R[:, 0]), h * float(u @ R[:, 2])) - dist


def _section(sec, u):
    """Rig cross-section [half-width, +z, -z] at u along the bone (entries at u = 0.1, 0.5, 0.9)."""
    S = np.array([x if x else sec[1] for x in sec], float)
    if u <= 0.5:
        t = max(0.0, (u - 0.1) / 0.4)
        return S[0] + (S[1] - S[0]) * t
    return S[1] + (S[2] - S[1]) * min(1.0, (u - 0.5) / 0.4)


def _bone_sink(head, R, L, sec_at, us, p0, s):
    worst = -1.0
    for u in us:
        w, zp, zm = sec_at(u)
        c = head + R[:, 1] * L * u + R[:, 2] * (zp - zm) / 2
        worst = max(worst, _sink(c, R, w, (zp + zm) / 2, p0, s))
    return worst


class Grip:
    """One side's grip, in the rig's rest space (world coordinates of blockout/hand_model/skeleton.json)."""

    def __init__(self, side):
        self.side = side
        rig = self.rig = HandRig(side)
        sk = rig.sk
        Mh, c0 = rig.M_rest, rig.c_rest                  # rows: hand x, y, z as rest-space vectors
        Rr, pr = sk.rest_root()
        b = "finger2-1"
        ix = Mh @ (sk.h0[b] + sk.R0[b][:, 1] * sk.L[b] * INDEX_U - c0)
        s = Mh.T @ _unit([ix[0] - HEEL[0], ix[1] - HEEL[1], 0.0])
        # the stick as high under the palm as its skin allows
        palm = lambda p0: max(_bone_sink(sk.h0[b], sk.R0[b], sk.L[b], lambda u, b=b: _section(sk.section[b], u),
                                         us, p0, s) for b, us in PALM)
        z = 0.0
        while palm(c0 + Mh.T @ np.array([HEEL[0], HEEL[1], z])) > -GAP:
            z -= 0.0005
        p0 = c0 + Mh.T @ np.array([HEEL[0], HEEL[1], z])
        # the palm's skin, for a finger curled past the stick: the plane the stick's top touches
        self.palm = (Mh, c0, z + STICK_R + GAP)
        bases = rig.bases(Rr, pr)
        q = rig.q_rest.copy()
        for ci in range(1, 5):
            ch, a = rig.chains[ci], rig.sl[ci]
            mcp_y = float((Mh @ (sk.h0[f"finger{ci + 1}-1"] - c0))[1])
            qc = self._close(ch, bases[ci], q[a:rig.sl[ci + 1]].copy(), p0, s, mcp_y)
            q[a:rig.sl[ci + 1]] = qc
        # fulcrum: where the index finger's middle phalanx crosses the stick
        ch = rig.chains[1]
        pts, rots = ch.fk(*bases[1], q[rig.sl[1]:rig.sl[2]])
        mid = pts[1] + rots[1][:, 1] * ch.L[1] * 0.5
        F = p0 + ((mid - p0) @ s) * s
        n = Mh[2] - (Mh[2] @ s) * s
        n = _unit(n)
        a_out = _unit(np.cross(s, n))
        if a_out @ Mh[0] < 0:                            # across, toward the pinky side (away from the body)
            a_out = -a_out
        thumb_dir = -math.cos(THUMB_AT) * a_out + math.sin(THUMB_AT) * n
        ch = rig.chains[0]
        q[0:rig.sl[1]] = self._thumb(ch, bases[0], q[0:rig.sl[1]].copy(), F + STICK_R * thumb_dir, F, s)
        self.q_grip = q
        self.q_open = q + OPEN * (rig.q_rest - q)
        self.Lr = np.stack([a_out, s, n], 1)            # stick frame (a, d, n) in rest space
        self.F, self.Rr, self.pr = F, Rr, pr
        self.arm_r = math.cos(WRIST_DEV) * Mh[1] - math.sin(WRIST_DEV) * Mh[0]
        mcp = [sk.h0[f"finger{f}-1"] for f in (2, 5)]
        self.box = [(pr, c0, 0.030, 0.027), (mcp[0] - 0.014 * Mh[2], mcp[1] - 0.014 * Mh[2], 0.024, 0.022)]
        self.sink = self._report(bases, p0, s)

    @staticmethod
    def _chain_sink(ch, base, qc, first, p0, s):
        pts, rots = ch.fk(*base, qc)
        return max(_bone_sink(pts[i], rots[i], ch.L[i], lambda u, i=i: ch._sect_at(i, min(u, 0.95)),
                              (0.1, 0.3, 0.5, 0.7, 0.9, 1.0), p0, s) for i in range(first, 3))

    def _in_palm(self, ch, base, qc, i, mcp_y):
        """How far bone i's skin pushes into the palm (m, <= 0 clear), where the bone has curled back under it."""
        Mh, c0, zp = self.palm
        pts, rots = ch.fk(*base, qc)
        worst = -1.0
        for u in (0.5, 0.8, 1.0):
            w, a, b = ch._sect_at(i, min(u, 0.95))
            p = Mh @ (pts[i] + rots[i][:, 1] * ch.L[i] * u + rots[i][:, 2] * (a - b) / 2 - c0)
            if p[1] < mcp_y - 0.01:
                worst = max(worst, p[2] + (a + b) / 2 - zp)
        return worst

    def _close(self, ch, base, qc, p0, s, mcp_y):
        flex = [k for k, d in enumerate(ch.dofs) if d[1] == "flex"]      # MCP, PIP, DIP
        rate = [1.0, 1.0, ch.k_dip or 1.0]
        for k in flex:
            qc[k] = max(ch.lo[k], 0.0)
        active = [True, True, True]
        while any(active):
            trial = qc.copy()
            for i, k in enumerate(flex):
                if active[i]:
                    trial[k] = min(ch.hi[k], trial[k] + STEP * rate[i])
            if ch.k_dip:        # the shared tendon: the DIP can't curl far past its share of the PIP
                trial[flex[2]] = min(trial[flex[2]], max(qc[flex[2]], ch.k_dip * trial[flex[1]] + DIP_SLACK))
            hit = [self._bone_only(ch, base, trial, i, p0, s) > 0 or self._in_palm(ch, base, trial, i, mcp_y) > 0
                   for i in range(3)]
            if any(hit):
                last = max(i for i in range(3) if hit[i])
                for i in range(last + 1):
                    active[i] = False
                continue
            moved = any(trial[k] != qc[k] for k in flex)
            qc = trial
            for i, k in enumerate(flex):
                if qc[k] >= ch.hi[k]:
                    active[i] = False
            if not moved:
                break
        return qc

    @staticmethod
    def _bone_only(ch, base, qc, i, p0, s):
        pts, rots = ch.fk(*base, qc)
        return _bone_sink(pts[i], rots[i], ch.L[i], lambda u: ch._sect_at(i, min(u, 0.95)),
                          (0.1, 0.3, 0.5, 0.7, 0.9, 1.0), p0, s)

    def _thumb(self, ch, base, q, target, p0, s, iters=40):
        """Thumb pad to target, skin out of the stick, near the relaxed thumb: damped least squares."""
        q_rest = q.copy()

        def res(q):
            pts, rots = ch.fk(*base, q)
            e = ch.effector(pts, rots) - target
            pen = [max(0.0, _bone_sink(pts[i], rots[i], ch.L[i], lambda u, i=i: ch._sect_at(i, min(u, 0.95)),
                                       (0.3, 0.6, 0.9), p0, s)) for i in range(3)]
            return np.concatenate([e, 10.0 * np.array(pen), 2e-3 * (q - q_rest) / ch.rng])

        r = res(q)
        c, lam = float(r @ r), 1e-3
        for _ in range(iters):
            J = np.empty((len(r), len(q)))
            for i in range(len(q)):
                dq = q.copy()
                h = 1e-5 if dq[i] + 1e-5 <= ch.hi[i] else -1e-5
                dq[i] += h
                J[:, i] = (res(dq) - r) / h
            A, g = J.T @ J, J.T @ r
            ok = False
            for _t in range(6):
                qn = np.clip(q + np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g), ch.lo, ch.hi)
                rn = res(qn)
                if float(rn @ rn) < c:
                    q, r, c, ok = qn, rn, float(rn @ rn), True
                    lam = max(lam / 3, 1e-7)
                    break
                lam *= 4
            if not ok:
                break
        return q

    def _report(self, bases, p0, s):
        """Deepest skin of each finger in the stick at the grip, m (negative: clear)."""
        out = []
        for ci, ch in enumerate(self.rig.chains):
            qc = self.q_grip[self.rig.sl[ci]:self.rig.sl[ci + 1]]
            out.append(round(self._chain_sink(ch, bases[ci], qc, 0, p0, s), 4))
        return out

    def place(self, d):
        """Rotation from rest space to the kit, for a stick pointing along d."""
        a, n = frame(d, 1 if self.side == "R" else -1)
        return np.stack([a, d, n], 1) @ self.Lr.T

    def q(self, openness):
        o = float(np.clip(openness, 0.0, 1.0))
        return self.q_grip + o * (self.q_open - self.q_grip)


_GRIPS = {}


def grip_of(side):
    if side not in _GRIPS:
        _GRIPS[side] = Grip(side)
    return _GRIPS[side]


def frame(d, side):
    """(a, n) across the stick for a hand on side +1 (R) / -1 (L) holding it along d."""
    up = np.array([0.0, 0.0, 1.0])
    n = up - (up @ d) * d
    if np.linalg.norm(n) < 1e-6:        # stick straight up or down: the back of the hand faces the drummer
        n = np.array([0.0, -1.0, 0.0]) + d[1] * d
    n = n / np.linalg.norm(n)
    a = side * np.cross(d, n)
    a /= np.linalg.norm(a)
    c, s = math.cos(GRIP_ROLL), math.sin(GRIP_ROLL)     # the back of the hand turns toward the pinky side
    return c * a - s * n, c * n + s * a


def _side(side):
    return "R" if side in (1, "R") else "L"


def wrist(grip, d, side):
    """Where the arm meets this hand (the wrist bone's head), for a stick held at grip pointing along d."""
    g = grip_of(_side(side))
    d = _unit(d)
    return np.asarray(grip, float) + g.place(d) @ (g.pr - g.F)


def forward(d, side):
    """The forearm's line (elbow -> wrist) for performer.arm_ik's hand_fwd: the hand's axis turned WRIST_DEV back
    toward its thumb side. A stick lies across the palm well off the hand's axis; with the wrist straight the
    forearm would point back across the chest, so drummers hold the hand bent toward the pinky side."""
    g = grip_of(_side(side))
    d = _unit(d)
    return g.place(d) @ g.arm_r


def _rows(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _scalar(x):
    return float(x) if np.ndim(x) == 0 else x


def bend(d, side, elbow, wrist):
    """(flexion, deviation) of the wrist, rad, for a stick along d and the forearm elbow -> wrist: flexion +
    toward the palm, deviation + radial (toward the thumb), - ulnar. elbow may be (K, 3) candidates: K of each."""
    g = grip_of(_side(side))
    T = g.place(_unit(d))
    fa = _rows(np.asarray(wrist, float) - np.asarray(elbow, float))
    Mh = g.rig.M_rest
    return tuple(_scalar(np.arcsin(np.clip(fa @ (T @ Mh[i]), -1, 1))) for i in (2, 0))


def _past(f, v):
    """How far (rad) flexion f and deviation v are past FLEX_OK and DEV_OK."""
    return (np.maximum(FLEX_OK[0] - f, 0) + np.maximum(f - FLEX_OK[1], 0)
            + np.maximum(DEV_OK[0] - v, 0) + np.maximum(v - DEV_OK[1], 0))


def strain(d, side, elbow, wrist):
    """How far (rad) the wrist bends past DEV_OK and FLEX_OK."""
    return _scalar(_past(*bend(d, side, elbow, wrist)))


ELBOW_HANG = (0.35, -0.3, -1.0)     # performer.arm_ik's pole (x toward the hand's side): down, out and back


def arm_cost(d, side, shoulder, elbow, wrist):
    """How awkward an elbow is for a stick along d: the wrist bent past its comfortable range (2 per rad), a
    little toward its neutral (0.3 per rad off flat and WRIST_DEV ulnar), and the elbow off its natural hang
    about the shoulder-wrist axis (0 .. 2). elbow may be (K, 3) candidates."""
    sg = 1 if _side(side) == "R" else -1
    f, v = bend(d, side, elbow, wrist)
    sh = np.asarray(shoulder, float)
    u = _unit(np.asarray(wrist, float) - sh)
    pole = np.array([sg * ELBOW_HANG[0], ELBOW_HANG[1], ELBOW_HANG[2]])
    pole = _unit(pole - (pole @ u) * u)
    off = np.asarray(elbow, float) - sh
    off = _rows(off - np.asarray(off @ u)[..., None] * u)
    return _scalar(2 * _past(f, v) + 0.3 * (np.abs(f) + np.abs(v + WRIST_DEV)) + (1 - off @ pole))


def hitbox(grip, d, side):
    """[(a, b, r0, r1)] capsules covering the hand: the palm, and the knuckles with the fingers curled under."""
    g = grip_of(_side(side))
    d = _unit(d)
    T, grip = g.place(d), np.asarray(grip, float)
    return [(grip + T @ (p - g.F), grip + T @ (q - g.F), r0, r1) for p, q, r0, r1 in g.box]


def pose(grip, d, side, openness=0.0):
    """{"root": {"rot", "pos"}, "angles", "fingers"} in the kit, rounded for the JSON."""
    g = grip_of(_side(side))
    d = _unit(d)
    T, grip = g.place(d), np.asarray(grip, float)
    root_rot = T @ g.Rr
    root_pos = grip + T @ (g.pr - g.F)
    x = np.concatenate([root_pos, np.zeros(3), g.q(openness)])
    _, _, pts = g.rig.points(x, root_rot)
    ang = {b: {k: round(v, 4) for k, v in dofs.items()} for b, dofs in g.rig.angles(x).items()}
    return {"root": {"rot": root_rot.round(5).tolist(), "pos": root_pos.round(4).tolist()}, "angles": ang,
            "fingers": [[np.asarray(p).round(4).tolist() for p in ch] for ch in pts]}


def openness(t, tosses, hand):
    """0 (grip) .. 1 (open) at time t for one hand: open from just after each toss's release to its catch."""
    o = 0.0
    for ts in tosses:
        if ts["hand"] != hand:
            continue
        u = float(np.clip((t - ts["t0"]) / OPEN_S, 0, 1))
        v = float(np.clip((t - (ts["t1"] - CLOSE_S)) / CLOSE_S, 0, 1))
        w = u * (1 - v)
        o = max(o, w * w * (3 - 2 * w))
    return o


def add(frames, tosses=()):
    """Writes the hand pose into every hand of every frame (drums.py's output, times relative to the same start
    as the tosses' t0 / t1)."""
    for fr in frames:
        for hand, hd in fr["hands"].items():
            d = np.array(hd["hand_dir"]) if "hand_dir" in hd else np.array(hd["tip"]) - np.array(hd["butt"])
            hd.update(pose(hd["grip"], d, hand, openness(fr["t"], tosses, hand)))


if __name__ == "__main__":
    for side in "RL":
        g = grip_of(side)
        deg = lambda q: np.round(np.degrees(q), 1).tolist()
        print(side, "sink per finger (m):", g.sink)
        for ci, ch in enumerate(g.rig.chains):
            print("  ", ch.f, [d[:2] for d in ch.dofs], deg(g.q_grip[g.rig.sl[ci]:g.rig.sl[ci + 1]]))
        print("  wrist from fulcrum (stick frame a, d, n):", np.round(g.Lr.T @ (g.pr - g.F), 4).tolist())

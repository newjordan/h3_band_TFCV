"""Hand skeleton: joints with their own frames, degrees of freedom and anatomical limits, and forward kinematics.

The hand is the MPFB2 rig (blockout/hand_model/skeleton.json, exported by export_skeleton.py). A pose is a dict of
joint angles (radians) per bone plus the root (wrist bone) world transform. Every bone rotates about axes fixed in
its own rest frame, so FK here and the Blender pose (matrix_basis = the same local rotation) agree exactly.

Joint table (degrees; flexion positive toward the palm, abduction positive away from the middle finger/out to the
thumb side for the thumb). Ranges are the usual active ranges of motion of an adult hand (AAOS / clinical ROM
tables): finger MCP flexion 90, hyperextension ~20, abduction ~20 (index, pinky ~25); PIP flexion ~100; DIP flexion
~80, hyperextension ~5 with the DIP following the PIP through the shared tendon (DIP ~ 2/3 PIP); thumb CMC about
+-30 deg around its rest attitude on two axes; thumb MCP -10..55, IP -10..80; wrist flexion/extension +-70,
radial/ulnar deviation -20..30. Instruments set targets; nothing here knows about pianos.
"""
import json, math, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKELETON_JSON = os.path.join(os.path.dirname(HERE), "hand_model", "skeleton.json")
FINGERS = [[f"finger{f}-{s}" for s in (1, 2, 3)] for f in range(1, 6)]   # thumb .. pinky
METACARPAL = {2: "metacarpal1", 3: "metacarpal2", 4: "metacarpal3", 5: "metacarpal4"}
D = math.radians

# joint -> list of (dof name, (lo, hi) degrees). Axes are resolved per bone from the rest pose (see Skeleton).
LIMITS = {
    "mcp": {"flex": (-20, 90), "abd": (-20, 20)},
    "mcp_edge": {"flex": (-20, 90), "abd": (-25, 25)},          # index and pinky spread further
    "pip": {"flex": (0, 103)},
    "dip": {"flex": (-5, 85)},
    "cmc": {"flex": (-30, 30), "abd": (-30, 30)},
    "tmcp": {"flex": (-10, 55)},
    "tip": {"flex": (-20, 80)},                                  # thumb IP: hyperextends 10-20 deg normally
    "wrist": {"flex": (-70, 70), "abd": (-20, 30)},
}
if __import__("os").environ.get("RIG_LIMITS"):       # experiment hook: {"mcp_edge": {"abd": [-40, 40]}, ...}
    for _k, _v in json.loads(__import__("os").environ["RIG_LIMITS"]).items():
        LIMITS.setdefault(_k, {}).update({a: tuple(b) for a, b in _v.items()})


def joint_kind(bone):
    if bone.startswith("finger1-"):
        return {"1": "cmc", "2": "tmcp", "3": "tip"}[bone[-1]]
    if bone.startswith("finger"):
        f = int(bone[6])
        return {"1": "mcp_edge" if f in (2, 5) else "mcp", "2": "pip", "3": "dip"}[bone[-1]]
    if bone == "wrist":
        return "wrist"
    return None


def _rot(axis, a):
    axis = np.asarray(axis, float)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def hand_scale():
    """Uniform hand scale from blockout/rig/furelise_cal.json ("hand_scale"); 1 with RIG_NO_FURELISE=1 or HAND_SCALE=1."""
    import os
    if os.environ.get("HAND_SCALE"):
        return float(os.environ["HAND_SCALE"])
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "furelise_cal.json")
    if os.environ.get("RIG_NO_FURELISE") == "1" or not os.path.exists(p):
        return 1.0
    return float(json.load(open(p)).get("hand_scale", 1.0))


class Skeleton:
    """One hand. Bone rest data in world space; per joint the local axes of its DOFs and the sign that makes
    positive flexion curl toward the palm."""

    def __init__(self, side, path=SKELETON_JSON):
        self.side = side
        B = json.load(open(path))["hands"][side]
        self.names = [n for n in self._order(B) if n not in ("lowerarm01", "lowerarm02")]
        self.parent = {n: B[n]["parent"] for n in self.names}
        self.R0 = {n: np.array(B[n]["rest"]) for n in B}               # rest frame: columns = local x, y, z
        self.h0 = {n: np.array(B[n]["head"]) for n in B}
        self.L = {n: B[n]["length"] for n in B}
        self.radii = {n: B[n]["radii"] for n in B}
        self.section = {n: B[n].get("section") for n in B}     # [[half-width, +z extent, -z extent] x (u .1 .5 .9)]
        self.scale = hand_scale()                   # the hand (not the forearm) scaled about the wrist to a pianist's
        if self.scale != 1.0:
            hw = self.h0["wrist"].copy()
            for n in self.names:
                self.h0[n] = hw + self.scale * (self.h0[n] - hw)
                self.L[n] = self.L[n] * self.scale
                self.radii[n] = [r * self.scale if r else r for r in self.radii[n]]
                if self.section[n]:
                    self.section[n] = [[v * self.scale if v else v for v in row] if row else row for row in self.section[n]]
        # hand frame at rest: across (thumb side -> pinky side), forward (wrist -> knuckles), up (back of hand)
        mcp = np.array([self.h0[f"finger{f}-1"] for f in range(2, 6)])
        c = mcp.mean(0)
        fwd = c - self.h0["wrist"]; fwd /= np.linalg.norm(fwd)
        ac = mcp[3] - mcp[0]; ac -= fwd * (ac @ fwd); ac /= np.linalg.norm(ac)
        up = np.cross(ac, fwd) * (1 if side == "R" else -1)
        self.rest_frame = np.stack([ac, fwd, up])
        self.axes = {}
        for n in self.names:
            k = joint_kind(n)
            if k is None:
                continue
            R = self.R0[n]
            if k in ("cmc", "tmcp", "tip"):
                # thumb hinge: the axis of its rest bend (metacarpal -> proximal), expressed in the bone frame
                d1 = self._dir("finger1-1"); d2 = self._dir("finger1-2")
                hinge = np.cross(d1, d2); hinge /= np.linalg.norm(hinge)
                flex_w = hinge
                abd_w = np.cross(self._dir(n), flex_w); abd_w /= np.linalg.norm(abd_w)
            else:
                # a true hinge: square to this bone and parallel to the palm (fingers fan, so each bone's own axis)
                y = self._dir(n)
                flex_w = np.cross(y, up); flex_w /= np.linalg.norm(flex_w)
                abd_w = up - y * (up @ y); abd_w /= np.linalg.norm(abd_w)
            # ... with the sign that sends the bone's tail toward the palm (-up) / toward the thumb (-across)
            fl = R.T @ flex_w
            fl = self._snap(fl)
            y = self._dir(n)
            if (np.cross(R @ fl, y) @ -up) < 0 and k not in ("cmc", "tmcp", "tip"):
                fl = -fl
            dofs = {"flex": fl}
            if "abd" in LIMITS[k]:
                ab = self._snap(R.T @ abd_w)
                dofs["abd"] = ab
            self.axes[n] = dofs
        self.kind = {n: joint_kind(n) for n in self.names}

    @staticmethod
    def _snap(v):
        """Bone-local axis: keep it exact (unit), but it is not forced onto x/z -- MPFB bone rolls vary."""
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    def _dir(self, n):
        return self.R0[n][:, 1]

    @staticmethod
    def _order(B):
        out, seen = [], set()

        def visit(n):
            if n in seen:
                return
            p = B[n]["parent"]
            if p is not None and p in B:
                visit(p)
            seen.add(n); out.append(n)
        for n in B:
            visit(n)
        return out

    def limits(self, bone):
        return {k: (D(lo), D(hi)) for k, (lo, hi) in LIMITS[self.kind[bone]].items()}

    def local_rot(self, bone, ang):
        """Rotation (bone-local) for joint angles {'flex': a, 'abd': b}: abduction first, then flexion."""
        R = np.eye(3)
        ax = self.axes.get(bone, {})
        if "abd" in ax and ang.get("abd"):
            R = R @ _rot(ax["abd"], ang["abd"])
        if "flex" in ax and ang.get("flex"):
            R = R @ _rot(ax["flex"], ang["flex"])
        return R

    def fk(self, root_rot, root_pos, angles):
        """World rotation and head/tail of every bone. root_rot/root_pos: the wrist bone's world frame and head
        (root_rot already includes any wrist joint angles). angles: {bone: {'flex':, 'abd':}}."""
        W, H, T = {}, {}, {}
        for n in self.names:
            p = self.parent[n]
            if n == "wrist":
                W[n] = np.asarray(root_rot, float)
                H[n] = np.asarray(root_pos, float)
            else:
                rel = self.R0[p].T @ self.R0[n]
                W[n] = W[p] @ rel @ self.local_rot(n, angles.get(n, {}))
                H[n] = H[p] + W[p] @ (self.R0[p].T @ (self.h0[n] - self.h0[p]))
            T[n] = H[n] + W[n][:, 1] * self.L[n]
        return W, H, T

    def rest_root(self):
        return self.R0["wrist"].copy(), self.h0["wrist"].copy()

    def chain_points(self, H, T, f):
        """Finger f (1 = thumb .. 5 = pinky): [base, j1, j2, tip] world points."""
        b = FINGERS[f - 1]
        return [H[b[0]], H[b[1]], H[b[2]], T[b[2]]]


def _selftest():
    for side in "RL":
        S = Skeleton(side)
        Rr, pr = S.rest_root()
        W, H, T = S.fk(Rr, pr, {})
        err = max(np.linalg.norm(H[n] - S.h0[n]) for n in S.names)
        print(side, "rest FK error (m):", f"{err:.2e}")
        # flexing the middle finger's PIP should move its tip toward the palm
        up = S.rest_frame[2]
        for b in ("finger3-2", "finger3-1", "finger1-2"):
            W2, H2, T2 = S.fk(Rr, pr, {b: {"flex": D(40)}})
            tip0 = T["finger3-3" if b.startswith("finger3") else "finger1-3"]
            tip1 = T2["finger3-3" if b.startswith("finger3") else "finger1-3"]
            print("  ", b, "+40 flex moves tip toward palm by", round(float((tip0 - tip1) @ up) * 1000, 1), "mm")


if __name__ == "__main__":
    _selftest()

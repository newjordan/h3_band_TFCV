"""Stitch overlapping suite parts: frames crossfaded over each 1 s overlap (root, joint angles, points; keys from the
later part past the middle)."""
import json, glob, numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from blockout.rig.hand import rotvec
from scipy.spatial.transform import Rotation as Rot
O = sys.argv[1]
parts = sorted(glob.glob(f"{O}/part_*.json"), key=lambda p: int(p.split("_")[-1][:-5]))
A = [json.load(open(p)) for p in parts]
fps = A[0]["fps"]; T0 = 39.9
def absf(a):
    return {int(round((a["start"] + f["t"] - T0) * fps)): f for f in a["frames"]}
out = {}
def lerp(x, y, w):
    if isinstance(x, list): return [lerp(a, b, w) for a, b in zip(x, y)]
    if isinstance(x, dict): return {k: lerp(x[k], y[k], w) for k in x}
    return x * (1 - w) + y * w
for a in A:
    F = absf(a)
    keys = sorted(F)
    ov = [j for j in keys if j in out]
    for j in keys:
        f = F[j]
        if j in out and ov:
            w = (j - ov[0] + 1) / (len(ov) + 1)
            g = out[j]; h = {}
            for name in f["hands"]:
                p, q = g["hands"][name], f["hands"][name]
                r = dict(q)
                for k in ("wrist", "fingers", "pads", "elbow", "angles"):
                    r[k] = lerp(p[k], q[k], w)
                R = Rot.from_matrix([p["root"]["rot"], q["root"]["rot"]])
                rv = (R[0].inv() * R[1]).as_rotvec() * w
                r["root"] = {"rot": (R[0] * Rot.from_rotvec(rv)).as_matrix().tolist(),
                             "pos": lerp(p["root"]["pos"], q["root"]["pos"], w), "plan": q["root"]["plan"]}
                h[name] = r
            f = dict(f) if w >= 0.5 else dict(g)
            f["hands"] = h
            f["body"] = dict(f["body"]); f["body"]["elbows"] = {n: h[n]["elbow"] for n in h}
        out[j] = f
idx = sorted(out)
frames = []
for j in idx:
    f = dict(out[j]); f["t"] = round(j / fps, 4); frames.append(f)
base = dict(A[0]); base["start"] = T0; base["frames"] = frames
bt = sorted({round(b + a["start"] - T0, 4) for a in A for b in a["beats"]}); db = sorted({round(b + a["start"] - T0, 4) for a in A for b in a["downbeats"]})
base["beats"], base["downbeats"] = bt, db
base["qa_parts"] = [a["qa"] for a in A]
json.dump(base, open(f"{O}/suite.json", "w"))
print(len(frames), "frames")

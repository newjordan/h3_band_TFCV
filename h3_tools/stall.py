"""Solver stalls: a finger whose joints are bit-identical to the previous frame while its target moved, and
pops: a joint changing > 12 deg in one frame."""
import json, sys, numpy as np
for path in sys.argv[1:]:
    a = json.load(open(path)); fr = a["frames"]; out = []
    for h in "LR":
        names = sorted({(b, k) for b, d in fr[0]["hands"][h]["angles"].items() for k in d})
        Q = np.degrees(np.array([[f["hands"][h]["angles"][b][k] for b, k in names] for f in fr]))
        fing = np.array([int(b[6]) - 1 for b, k in names])
        st = pop = 0
        for f in range(5):
            q = Q[:, fing == f]; d = np.abs(np.diff(q, axis=0))
            st += int((d.max(1) < 1e-4).sum()); pop += int((d.max(1) > 12).sum())
        out.append(f"{h}: stalled finger-frames {st}, pops>12deg {pop}")
    print(path.split("/")[-1], " | ".join(out))

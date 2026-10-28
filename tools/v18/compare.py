"""Compare two solved suites frame by frame (e.g. a fresh re-solve against the saved v18 take).
Usage: python tools/v18/compare.py A.json B.json"""
import json, sys, numpy as np
A, B = (json.load(open(p)) for p in sys.argv[1:3])
fa, fb = A["frames"], B["frames"]
print("frames", len(fa), len(fb))
n = min(len(fa), len(fb)); dang = []; dwr = []; dkey = 0
for i in range(n):
    for h in "RL":
        a, b = fa[i]["hands"][h], fb[i]["hands"][h]
        dang.append(max(abs(a["angles"][j][k] - b["angles"][j][k]) for j in a["angles"] for k in a["angles"][j]))
        dwr.append(np.abs(np.subtract(a["wrist"], b["wrist"])).max())
    ka, kb = fa[i]["keys"], fb[i]["keys"]
    dkey += any(abs(ka.get(p, 0) - kb.get(p, 0)) > 1e-6 for p in set(ka) | set(kb))
dang, dwr = np.degrees(dang), np.array(dwr) * 1000
print(f"joint angle diff deg: max {dang.max():.6f}  p99 {np.percentile(dang, 99):.6f}  frames>0.01deg {(dang > .01).sum()}")
print(f"wrist diff mm: max {dwr.max():.6f}  p99 {np.percentile(dwr, 99):.6f}")
print(f"frames with any key depth difference: {dkey}/{n}")

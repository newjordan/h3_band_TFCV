import sys, numpy as np, math
sys.path.insert(0, "/home/frosty40/h3b_wrist")
from blockout import midi, feeling, piano as P
notes = midi.read("/home/frosty40/h3b_wrist/examples/piano/knights_suite.mid")
bts, dbs = midi.beats("/home/frosty40/h3b_wrist/examples/piano/knights_suite.mid")
ch = feeling.Chart("/home/frosty40/h3b_wrist/examples/piano/knights_suite.feeling.json", dbs)
notes, tm = feeling.perform(notes, ch, 0)
sched = [(float(tm(t)), h) for t, h in ch.heads]
efn = lambda t: float(np.clip(1.4 * ch.at("strength", np.interp(t, tm.tp, tm.ts)), 0, 1))
bts2, dbs2 = [float(tm(b)) for b in bts], [float(tm(b)) for b in dbs]
anim = P.animate(notes, 24, 250.0, 2.5, 1.0, bts2, dbs2, sched, efn, 0)
fr = anim["frames"]
def rdiff(a, b):
    R = np.asarray(a).T @ np.asarray(b)
    return math.degrees(np.arccos(np.clip((np.trace(R[:3,:3]) - 1) / 2, -1, 1)))
mx = {"L": 0, "R": 0}
for j in range(1, len(fr)):
    t = 250 - 2 + j / 24
    for s in ("L", "R"):
        r0 = fr[j-1]["hands"][s]["root"]["rot"]; r1 = fr[j]["hands"][s]["root"]["rot"]
        p0 = np.array(fr[j-1]["hands"][s]["root"]["pos"]); p1 = np.array(fr[j]["hands"][s]["root"]["pos"])
        d = rdiff(r0, r1); mv = np.linalg.norm(p1 - p0)
        mx[s] = max(mx[s], d)
        if d > 25 or mv > 0.05:
            print(f"t={t:6.2f} {s}: rot {d:5.1f} deg, wrist {mv*1000:4.0f} mm", flush=True)
            print("  rot prev:", np.round(r0, 3).tolist(), flush=True)
            print("  rot now :", np.round(r1, 3).tolist(), flush=True)
print("max per-frame rot (L/R):", mx, flush=True)
print("PROBE DONE", flush=True)

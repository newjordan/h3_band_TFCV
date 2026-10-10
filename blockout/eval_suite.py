"""Key-sync evaluation of the rig over a suite window, note by note, for parameter sweeps.

    python -m blockout.eval_suite window MID FEELING START DUR OUT.json     # one window (notes counted in [START, START+DUR))
    python -m blockout.eval_suite aggregate OUT_DIR                          # all windows of a sweep -> summary

A note is synced when its key goes at least 50% down within 0.25 s of its onset (as everywhere in this project).
Each missed note is classified: unfingered (no finger was assigned), sideways (pad off the key across it),
short (pad short of the key along it: in front of a black key, or off the back), shallow (on the key but not
pressed 50% down), and tagged with its finger. The wrist's sideways bend against the forearm is summarised too.
"""
import json, math, os, sys

import numpy as np

PAD = 0.6          # s of extra animation after the window so its last notes have time to sound


def window(mid, feeling_path, start, dur, out):
    from . import midi, feeling, piano
    notes = midi.read(mid)
    bts, dbs = midi.beats(mid)
    ch = feeling.Chart(feeling_path, dbs)
    notes, tm = feeling.perform(notes, ch, 0)
    sched = [(float(tm(t)), h) for t, h in ch.heads]
    efn = lambda t, ch=ch, tm=tm: float(np.clip(1.4 * ch.at("strength", np.interp(t, tm.tp, tm.ts)), 0, 1))
    bts, dbs = [float(tm(b)) for b in bts], [float(tm(b)) for b in dbs]
    anim = piano.animate(notes, 24, start, dur + PAD, 1.0, bts, dbs, sched, efn, 0)
    fr = anim["frames"]; fps = anim["fps"]
    fmap = {}
    for side, sl in anim["fingering"].items():
        for s in sl:
            for t, p, f in s:
                fmap[(round(t, 3), p)] = (side, f)
    res = {"start": start, "dur": dur, "notes": 0, "ok": 0, "miss": [], "dev": []}
    for n in notes:
        if not (start <= n.start < start + dur):
            continue
        t = n.start - start
        j0, j1 = int(t * fps), min(len(fr) - 1, int((min(n.end, n.start + 0.25) - start) * fps) + 1)
        d = max(fr[j]["keys"].get(str(n.pitch), 0.0) for j in range(j0, j1 + 1))
        res["notes"] += 1
        if d >= 0.5:
            res["ok"] += 1
            continue
        side, f = fmap.get((round(n.start, 3), n.pitch), (None, None))
        m = {"t": round(n.start, 3), "pitch": n.pitch, "black": piano.is_black(n.pitch), "depth": round(d, 2)}
        if side is None:
            m["kind"] = "unfingered"
        else:
            jj = min(range(j0, j1 + 1), key=lambda j: fr[j]["hands"][side]["pads"][f][2])
            pad = fr[jj]["hands"][side]["pads"][f]
            kx = piano.key_x(n.pitch)
            hw = (piano.BLACK_W if piano.is_black(n.pitch) else piano.WHITE_W) / 2
            y0 = piano.WHITE_L - piano.BLACK_L if piano.is_black(n.pitch) else 0.0
            m.update(side=side, finger=f + 1, dx=round((pad[0] - kx) * 1000), y=round(pad[1] * 1000), z=round(pad[2] * 1000))
            if abs(pad[0] - kx) > hw + 0.002:
                m["kind"] = "sideways"
            elif pad[1] < y0 - 0.002 or pad[1] > piano.WHITE_L:
                m["kind"] = "short"
            else:
                m["kind"] = "shallow"
        res["miss"].append(m)
    # motion quality: wrist shake (high-pass of the wrist path), finger tremor (joints reversing > 2 deg on
    # consecutive frames), joints out of their anatomical range
    from scipy.signal import savgol_filter
    for side in ("L", "R"):
        W = np.array([f_["hands"][side]["wrist"] for f_ in fr])
        if len(W) > 9:
            res.setdefault("jitter_mm", []).append(float(np.sqrt(((W - savgol_filter(W, 9, 2, axis=0)) ** 2).sum(1).mean()) * 1000))
        names = sorted({(b, k) for b, dd in fr[0]["hands"][side]["angles"].items() for k in dd})
        Q = np.degrees(np.array([[f_["hands"][side]["angles"][b][k] for b, k in names] for f_ in fr]))
        v = np.diff(Q, axis=0)
        z = (v[1:] * v[:-1] < 0) & (np.abs(v[1:]) > 2) & (np.abs(v[:-1]) > 2)
        res.setdefault("tremor", []).append(float(z.sum() / z.size * fps))
    res["anatomy_out"] = int(sum(anim["qa"]["anatomy"]["out_of_range"].values()))
    res["frames"] = len(fr)
    for f in fr[::3]:
        for side, h in f["hands"].items():
            wr, el = np.array(h["wrist"]), np.array(h["elbow"])
            kn = np.mean([h["fingers"][k][0] for k in range(1, 5)], 0)
            fa = (wr - el) / np.linalg.norm(wr - el); hf = (kn - wr) / np.linalg.norm(kn - wr)
            res["dev"].append(round(math.degrees(math.acos(float(np.clip(fa @ hf, -1, 1)))), 1))
    json.dump(res, open(out, "w"))


def aggregate(d, sections=(("debussy", 39.9), ("rach", 78.8), ("chopin", 166.9))):
    import glob
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "w_*.json")))]
    n = sum(r["notes"] for r in R); ok = sum(r["ok"] for r in R)
    miss = [m for r in R for m in r["miss"]]
    dev = np.array([v for r in R for v in r["dev"]])
    sec = {}
    for r in R:
        name = [s for s, t0 in sections if r["start"] >= t0 - 1e-6][-1]
        a = sec.setdefault(name, [0, 0]); a[0] += r["ok"]; a[1] += r["notes"]
    from collections import Counter
    kinds = Counter(m["kind"] for m in miss)
    fing = Counter(f"{m.get('side', '?')}{m.get('finger', '?')}" for m in miss)
    out = {"windows": len(R), "notes": n, "ok": ok, "keysync": round(100 * ok / max(n, 1), 2),
           "sections": {k: f"{v[0]}/{v[1]} = {100 * v[0] / max(v[1], 1):.1f}%" for k, v in sec.items()},
           "miss_kinds": dict(kinds), "miss_fingers": dict(fing.most_common()),
           "jitter_mm": round(float(np.mean([v for r in R for v in r.get("jitter_mm", [])])), 2),
           "tremor_per_joint_s": round(float(np.mean([v for r in R for v in r.get("tremor", [])])), 3),
           "anatomy_out_per_min": round(sum(r.get("anatomy_out", 0) for r in R) / max(sum(r.get("frames", 0) for r in R) / 24 / 60, 1e-9), 1),
           "wrist_dev_median": round(float(np.median(dev)), 1) if len(dev) else None,
           "wrist_dev_p95": round(float(np.percentile(dev, 95)), 1) if len(dev) else None}
    json.dump(out | {"misses": miss}, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "window":
        window(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    else:
        aggregate(sys.argv[2])

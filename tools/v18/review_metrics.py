"""Score a solved suite the same way the 2026-10-16 rebase review scored every take, so a change can be compared
with v18 directly.  Usage: python tools/v18/review_metrics.py SUITE.json [NOTES.json] [--win 40-80,80-127,...]

keys in time: key at least half down within -1.5..+3 frames of each onset (from the take's own key depths)
finger speed: mean joint speed of index..pinky (deg/s); frozen: share of joints moving < 2 deg over a 1 s window
broken: knuckle bent back (PIP < -5, DIP < -15, MCP < -30 deg), tip hooked > 30 deg past PIP, or past range
pops/min: frames where any finger joint moves faster than 15 rad/s; crossed: index..pinky tips out of order > 4 mm"""
import json, sys, numpy as np
D = np.pi / 180
SECTIONS = {"Debussy": (39.9, 79.9), "Rach": (79.9, 127.0), "Chopin": (127.0, 390.5), "suite": (39.9, 390.5)}


def score(d, notes, t0, t1):
    fps, st = d["fps"], d["start"]
    fr = [f for f in d["frames"] if t0 - 1e-6 <= st + f["t"] < t1 - 1e-6]
    if len(fr) < fps:
        return None
    q = np.array([[[f["hands"][h]["angles"][f"finger{i}-{j}"]["flex"] for i in range(1, 6) for j in (1, 2, 3)]
                   for f in fr] for h in "RL"])
    ab = np.array([[[f["hands"][h]["angles"][f"finger{i}-1"]["abd"] for i in range(1, 6)] for f in fr] for h in "RL"])
    T = q.shape[1]; flex = q.reshape(2, T, 5, 3); lg = flex[:, :, 1:, :]
    mcp, pip, dip = lg[..., 0], lg[..., 1], lg[..., 2]
    dq = np.diff(flex, axis=1) * fps
    fz = [((lg[:, k:k + fps].max(1) - lg[:, k:k + fps].min(1)) < 2 * D).mean() for k in range(0, T - fps + 1, fps // 2)]
    br = (pip < -5 * D) | (dip < -15 * D) | (mcp < -30 * D) | (dip > pip + 30 * D) | (pip > 110 * D) | (dip > 90 * D) \
        | (np.abs(ab[:, :, 1:]) > 30 * D)
    cross = 0
    for h in "RL":
        X = np.array([[f["hands"][h]["fingers"][i][3][0] for i in range(1, 5)] for f in fr])
        s = np.sign(np.median(np.diff(X, axis=1)))
        cross += (np.diff(X, axis=1) * s < -0.004).any(axis=1).sum()
    ft = np.array([st + f["t"] for f in fr]); keys = [f["keys"] for f in fr]
    on = [n for n in notes if t0 + 0.1 <= n[0] < t1 - 0.15]; hit = 0
    for n in on:
        i0, i1 = np.searchsorted(ft, n[0] - 1.5 / fps), np.searchsorted(ft, n[0] + 3 / fps)
        hit += max([keys[i].get(str(n[2]), 0) for i in range(i0, min(i1, len(keys)))] + [0]) >= 0.5
    return {"keys in time %": 100 * hit / max(1, len(on)), "onsets": len(on),
            "finger speed deg/s": np.degrees(np.abs(dq[:, :, 1:]).mean()), "frozen %": 100 * float(np.mean(fz)),
            "broken %": 100 * br.mean(), "pops/min": float((np.abs(dq) > 15).any(axis=(2, 3)).sum() / (T / fps) * 60 / 2),
            "crossed %": 100 * cross / (2 * T), "curl PIP deg": np.degrees(pip.mean())}


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    d = json.load(open(a[0])); notes = json.load(open(a[1] if len(a) > 1 else a[0].rsplit("/", 1)[0] + "/notes.json"))
    wins = dict(SECTIONS)
    if "--win" in sys.argv:
        wins = {w: tuple(map(float, w.split("-"))) for w in sys.argv[sys.argv.index("--win") + 1].split(",")}
    rows = {k: score(d, notes, *w) for k, w in wins.items()}
    cols = next(r for r in rows.values() if r).keys()
    print(f"{'window':10s}" + "".join(f"{c:>20s}" for c in cols))
    for k, r in rows.items():
        if r:
            print(f"{k:10s}" + "".join(f"{r[c]:20.2f}" for c in cols))

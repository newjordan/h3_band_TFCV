#!/usr/bin/env python3
"""Does an H3 take move the way its blockout moves? Ranks takes against the grey blockout they were made from.

  drum_follow_score.py GREY.mp4 TAKE.mp4... --hits HITS.json --start START [--boot N] [--json]

The grey blockout is on the beat by construction, so a take whose arm and stick motion matches the grey's around each
hand hit has its sticks on the hits. This sidesteps drum_sync_score.py, which can't separate takes on a steady groove
(the grey blockout itself only clears its controls by about 0.1 there).
Motion: mean |frame diff| per cell of a 4x4 grid over the arm and stick zone (640x368 grey).
  hit    median over cells of the correlation with the grey in +-4-frame windows round every hand hit (kick and
         hi-hat pedal left out), at lag 0. Takes are ranked by this.
  lag    the frame offset that best matches the grey in those windows (0 = on time, + = the take is late)
  follow the same over the whole clip, at lag 0
  ctrl   'hit' against the grey shifted by 1 s: what a steady groove scores without following the grey
  drift  correlation of the bottom third (the kit) between the first and last half-second: 1 = the kit holds still
  flick  mean |frame diff| in the top corners (backdrop flicker; lower is better)
  sharp  Laplacian variance in the arm zone (detail; report only)
--boot N resamples the hits N times and reports how often each take ranks first, to show whether the ranking is noise.
"""
import argparse, json, subprocess

import numpy as np

W, H, FPS = 640, 368, 24
Y0, Y1, X0, X1, N = 60, 260, 120, 520, 4
HALF = 4


def frames(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def cells(f):
    d = np.abs(np.diff(f, axis=0))[:, Y0:Y1, X0:X1]
    ys, xs = np.array_split(np.arange(Y1 - Y0), N), np.array_split(np.arange(X1 - X0), N)
    return np.stack([d[:, a[0]:a[-1] + 1, b[0]:b[-1] + 1].mean(axis=(1, 2)) for a in ys for b in xs], 1)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def windowed(ref, take, idx, lag=0):
    """median over cells of take vs ref in the +-HALF windows round idx, take read lag frames later"""
    n = min(len(ref), len(take))
    rows = np.array([i for c in idx for i in range(c - HALF, c + HALF + 1) if 0 <= i < n and 0 <= i + lag < n])
    return float(np.median([corr(ref[rows, c], take[rows + lag, c]) for c in range(ref.shape[1])]))


def laplace_var(f):
    z = f[:, Y0:Y1, X0:X1]
    lap = 4 * z[:, 1:-1, 1:-1] - z[:, :-2, 1:-1] - z[:, 2:, 1:-1] - z[:, 1:-1, :-2] - z[:, 1:-1, 2:]
    return float(lap.var(axis=(1, 2)).mean())


def hit_rows(hits_path, start, nframes):
    """motion row i is the change from frame i to i + 1, so a hit on frame f shows as the stick arriving in row f - 1"""
    hits = [h for h in json.load(open(hits_path))["hits"] if h["piece"] not in ("kick", "hihat_pedal")]
    return sorted({round((h["t"] - start) * FPS) - 1 for h in hits if 0 <= (h["t"] - start) * FPS < nframes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grey"); ap.add_argument("takes", nargs="+")
    ap.add_argument("--hits", required=True); ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    gf = frames(a.grey)
    ref = cells(gf)
    idx = hit_rows(a.hits, a.start, len(gf))
    out, motion = [], []
    for p in a.takes:
        tf = frames(p)
        t = cells(tf)
        motion.append(t)
        n = min(len(ref), len(t))
        lags = {L: windowed(ref, t, idx, L) for L in range(-3, 4)}
        k = slice(H * 2 // 3, H)
        corners = np.concatenate([tf[:, :40, :100], tf[:, :40, -100:]], axis=2)
        out.append(dict(take=p.replace("\\", "/").split("/")[-1], hit=round(lags[0], 3), lag=max(lags, key=lags.get),
                        follow=round(float(np.median([corr(ref[:n, c], t[:n, c]) for c in range(ref.shape[1])])), 3),
                        ctrl=round(windowed(np.roll(ref, FPS, axis=0), t, idx), 3),
                        drift=round(corr(tf[:12, k].mean(0).ravel(), tf[-12:, k].mean(0).ravel()), 3),
                        flick=round(float(np.abs(np.diff(corners, axis=0)).mean()), 2), sharp=round(laplace_var(tf), 1)))
    if a.boot:
        rng, wins = np.random.default_rng(0), np.zeros(len(motion), int)
        for _ in range(a.boot):
            pick = rng.choice(idx, len(idx), replace=True)
            wins[int(np.argmax([windowed(ref, t, pick) for t in motion]))] += 1
        for r, w in zip(out, wins):
            r["first"] = round(float(w) / a.boot, 3)
    out.sort(key=lambda r: -r["hit"])
    if a.json:
        print(json.dumps(out))
        return
    print(f"{len(idx)} hand-hit windows, ranked by 'hit'")
    print(f"{'take':26s} {'hit':>6s} {'lag':>4s} {'follow':>7s} {'ctrl':>6s} {'drift':>6s} {'flick':>6s} {'sharp':>7s}"
          + ("  first" if a.boot else ""))
    for r in out:
        print(f"{r['take']:26s} {r['hit']:+6.3f} {r['lag']:+4d} {r['follow']:+7.3f} {r['ctrl']:+6.3f} {r['drift']:6.3f} "
              f"{r['flick']:6.2f} {r['sharp']:7.1f}" + (f"  {r['first']:5.1%}" if a.boot else ""))


if __name__ == "__main__":
    main()

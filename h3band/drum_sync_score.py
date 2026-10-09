#!/usr/bin/env python3
"""Stick-sync scorer for drum takes: stick motion vs known hit times, with controls a steady groove can't fool.

  drum_sync_score.py VIDEO HITS.json START [--box x0,y0,x1,y1] [--json]

HITS.json: {"hits": [{"t", "piece"}]} (blockout.drums --hits-out, or drum_events.py); kick and pedal hits
are dropped (feet are mostly out of frame). Frame 0 of VIDEO is START seconds into the hits, 24 fps.
Motion: strum_score2's sub-box signal (median-removed optical flow, the 3x3-cell window with the most
> 3 Hz vertical motion, high-passed); speed = |vertical velocity|.
Score: peak correlation of speed with the hit train within +-2 frames (H3's video tokens are 4 frames).
Controls: the hit train shifted by half its median gap (between hits instead of on them) and by 1/2/3 s,
plus a permutation p-value over 200 circular shifts of the motion. sync_margin = score - best control.
strum_score2's +-4-frame search let shifted controls realign onto an 8th-note groove (6.4 frames a hit at
112 bpm): the blockout itself only beat them by 0.17 there.
"""
import argparse, json

import numpy as np

from strum_score2 import FPS, read_frames, subbox_signal, train

LAG = 2


def xcorr(a, b, maxlag=LAG):
    best = -1.0
    for lag in range(-maxlag, maxlag + 1):
        u, v = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        if u.std() > 0 and v.std() > 0:
            best = max(best, float(np.corrcoef(u, v)[0, 1]))
    return best


def analyze(video, hits_path, start, box=None):
    frames = read_frames(video, box)
    n = len(frames)
    speed = np.abs(subbox_signal(frames)[0])
    hits = [h for h in json.load(open(hits_path))["hits"] if h["piece"] not in ("kick", "hihat_pedal")]
    on = np.array(sorted({round(h["t"] - start, 4) for h in hits if 0 <= (h["t"] - start) * FPS < n}))
    real = xcorr(speed, train(on, n))
    half = float(np.median(np.diff(on))) / 2 if len(on) > 1 else 0.13
    ctrl = {"between_hits": xcorr(speed, train(on + half, n))}
    for s in (1.0, 2.0, 3.0):
        ctrl[f"shift_{s:g}s"] = xcorr(speed, np.roll(train(on, n), int(round(s * FPS))))
    rng = np.random.default_rng(0)
    ot = train(on, n)
    null = [xcorr(np.roll(speed, int(rng.integers(6, n - 6))), ot) for _ in range(200)]
    p = float((np.sum(np.array(null) >= real) + 1) / 201)
    return {"video": video, "frames": n, "hits": int(len(on)), "score": round(real, 3),
            "controls": {k: round(v, 3) for k, v in ctrl.items()}, "p": round(p, 3),
            "sync_margin": round(real - max(ctrl.values()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("hits"); ap.add_argument("start", type=float)
    ap.add_argument("--box"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    box = [int(float(v)) for v in a.box.split(",")] if a.box else None
    r = analyze(a.video, a.hits, a.start, box)
    print(json.dumps(r) if a.json else "\n".join(f"{k:12s} {v}" for k, v in r.items()))


if __name__ == "__main__":
    main()

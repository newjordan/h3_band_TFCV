#!/usr/bin/env python3
"""CPU strum-sync scorer: motion inside the guitarist box vs the guitar onset train.

  strum_score.py VIDEO.mp4 EVENTS.json SONG_START [--box x0,y0,x1,y1] [--json]

Motion = mean optical-flow magnitude (Farneback) inside the box, frame t vs t-1, detrended by a
0.5 s moving average so only the rhythmic part remains. Onsets = guitar_events*.json events inside
[SONG_START, SONG_START + video length), as a per-frame train smoothed by a 1-frame Gaussian.
Peak normalized cross-correlation over +-8 frames, plus controls with the onset train circularly
shifted by 1, 2, 3 s. The guitar counterpart of sync_score.py.
"""
import argparse
import json

import cv2
import numpy as np

MAX_LAG = 8
SHIFTS_S = (1.0, 2.0, 3.0)


def motion_track(video, box):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    prev, out = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if box:
            x0, y0, x1, y1 = box
            frame = frame[y0:y1, x0:x1]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 320.0 / max(g.shape)
        if scale < 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if prev is None:
            out.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            out.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev = g
    cap.release()
    return np.array(out), fps


def onset_train(events_path, t0, n, fps):
    ev = json.load(open(events_path))["events"]
    train = np.zeros(n)
    for e in ev:
        f = (e["t"] - t0) * fps
        if 0 <= f < n:
            train[int(round(f)) if int(round(f)) < n else n - 1] += 1.0
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    return np.convolve(train, k / k.sum(), "same"), sum(1 for e in ev if 0 <= (e["t"] - t0) * fps < n)


def detrend(x, fps):
    w = max(3, int(round(fps / 2)))
    return x - np.convolve(x, np.ones(w) / w, "same")


def xcorr(a, b):
    best = (-2.0, 0)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            u, v = a[lag:], b[:len(b) - lag]
        else:
            u, v = a[:lag], b[-lag:]
        if len(u) < 8 or u.std() == 0 or v.std() == 0:
            continue
        c = float(np.corrcoef(u, v)[0, 1])
        if c > best[0]:
            best = (c, lag)
    return best


def analyze(video, events, t0, box=None):
    mot, fps = motion_track(video, box)
    m = detrend(mot, fps)
    train, n_on = onset_train(events, t0, len(m), fps)
    peak, lag = xcorr(m, train)
    zero = float(np.corrcoef(m, train)[0, 1]) if m.std() and train.std() else float("nan")
    ctrl = [xcorr(m, np.roll(train, int(s * fps)))[0] for s in SHIFTS_S]
    return {"video": video, "frames": len(m), "onsets": n_on, "box": box,
            "peak_corr": round(peak, 3), "peak_lag_frames": lag, "corr_lag0": round(zero, 3),
            "control_shifted_peak": [round(c, 3) for c in ctrl],
            "margin_vs_control": round(peak - max(ctrl), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video"); p.add_argument("events"); p.add_argument("song_start", type=float)
    p.add_argument("--box"); p.add_argument("--json", action="store_true")
    a = p.parse_args()
    box = [int(float(v)) for v in a.box.split(",")] if a.box else None
    r = analyze(a.video, a.events, a.song_start, box)
    if a.json:
        print(json.dumps(r))
    else:
        for k, v in r.items():
            print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()

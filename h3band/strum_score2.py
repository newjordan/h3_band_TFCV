#!/usr/bin/env python3
"""Strum-sync scorer v2: picking-stroke motion vs the guitar onsets, robust to body bounce.

  strum_score2.py VIDEO.mp4 EVENTS.json SONG_START [--box x0,y0,x1,y1] [--beats BEATS.json]
                  [--grid-lag SECONDS] [--json]

ALWAYS pass --box tight around the picking hand (and check the reported `subbox`). Given a whole-performer
zone, the sub-box search below picks whatever moves fastest at > 3 Hz, and on a real take that was
drifting stage smoke next to the guitarist: a false "in sync" result.

Signal (hand landmarks first, sub-box fallback):
  1. MediaPipe HandLandmarker ($H3B_HAND_MODEL, default models/hand_landmarker.task, from
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
     on the zone; if a hand is found in >= 50% of frames, the stroke signal is the vertical velocity of the
     lowest hand's wrist + index MCP.
  2. Otherwise (gloved / dark hands): dense optical flow on the zone with the zone's median flow subtracted
     (removes rigid body bounce and camera push), cut into a 24x24 cell grid; the 3x3-cell window with the
     most > 3 Hz vertical-velocity energy is the picking sub-box (a hand is ~1-2 cells; a 25%-of-zone pool
     diluted a synthetic stroke below the controls); the stroke signal is its mean vertical velocity,
     high-passed above 3 Hz. Strokes = peaks of |velocity|.

Metrics vs the onsets (frame 0 = SONG_START, 24 fps):
  a  peak xcorr of the stroke speed |v| with the onset train, +-4 frames
  b  phase locking of stroke peaks on the 8th-note grid: R*cos(mean stroke phase - mean onset phase)
     (Rayleigh R of stroke phases, signed by agreement with where the guitar actually strikes); informational
  c  share of stroke-speed power at the 8th/16th rates (3.9 / 7.8 Hz bands) within 1-10 Hz
Strokes = the N strongest |velocity| peaks, N = onsets in the clip.
Controls on the same video: onsets circularly shifted 1/2/3 s, onsets from another song section (+60 s),
and (for b) the grid phase-shifted by a quarter beat. sync_margin_X = real - max(control), except that
section_+60s is excluded from b's max (grid-locked music has the same phase in every section);
p_a / p_b = permutation p-values from 200 circular shifts of the motion.

sync_margin = sync_margin_a. b is reported but NOT used: validation showed it cannot separate a synthetic
onset-locked stroke from the static plate (both ~0.13). At 24 fps a 16th note is ~3 frames, so frame-quantized
stroke times carry +-1/3 cycle of phase error on a 16th grid, and on the 8th grid 16th-strummed onsets are
bimodal (phase 0 and 0.5). Beat-phase locking needs >= 60 fps footage or sub-frame stroke timing.
sync_margin = mean of the a and b margins. c has no audio control (it is a property of the motion).
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, find_peaks

FPS = 24.0
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402

GRID_LAG = 0.0       # set with --grid-lag: beat-tracker lag (guitar_events.py reports it as grid_lag.offset_s)
BEAT_P = 0.5         # replaced by the median beat period of the beats file in analyze()
HAND_MODEL = config.HAND_MODEL
GRID = 24
WIN = 3   # picking sub-box = the 3x3-cell window (1/64 of the zone) with the most > 3 Hz motion


def read_frames(video, box):
    cap = cv2.VideoCapture(video)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if box:
            x0, y0, x1, y1 = box
            f = f[y0:y1, x0:x1]
        out.append(f)
    cap.release()
    return out


def hand_signal(frames):
    if not HAND_MODEL.exists():
        return None
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opt = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), num_hands=2,
                                       min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2)
    lm = vision.HandLandmarker.create_from_options(opt)
    ys = []
    for f in frames:
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if r.hand_landmarks:
            h = max(r.hand_landmarks, key=lambda L: L[0].y)  # lowest hand = picking hand near the guitar body
            ys.append((h[0].y + h[5].y) / 2 * f.shape[0])
        else:
            ys.append(np.nan)
    ys = np.array(ys)
    if np.isnan(ys).mean() > 0.5:
        return None
    idx = np.arange(len(ys))
    ys = np.interp(idx, idx[~np.isnan(ys)], ys[~np.isnan(ys)])
    return np.diff(ys, prepend=ys[0])


def subbox_signal(frames):
    gray = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        s = 360.0 / max(g.shape)
        gray.append(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g)
    h, w = gray[0].shape
    vy = np.zeros((len(gray), GRID, GRID))
    for t in range(1, len(gray)):
        fl = cv2.calcOpticalFlowFarneback(gray[t - 1], gray[t], None, 0.5, 3, 11, 3, 5, 1.1, 0)
        fy = fl[..., 1] - np.median(fl[..., 1])          # remove rigid zone motion (bounce, push-in)
        cells = fy[:h // GRID * GRID, :w // GRID * GRID].reshape(GRID, h // GRID, GRID, w // GRID)
        vy[t] = cells.mean(axis=(1, 3))
    hp = sosfiltfilt(butter(3, 3.0, btype="high", fs=FPS, output="sos"), vy, axis=0)
    energy = (hp ** 2).sum(axis=0)
    best, at = -1.0, (0, 0)
    for r in range(GRID - WIN + 1):
        for c in range(GRID - WIN + 1):
            e = energy[r:r + WIN, c:c + WIN].sum()
            if e > best:
                best, at = e, (r, c)
    r, c = at
    win = hp[:, r:r + WIN, c:c + WIN].reshape(len(gray), -1)
    sig = win.mean(axis=1)
    cover = {"grid": GRID, "window_rows": [r, r + WIN], "window_cols": [c, c + WIN],
             "energy_share": round(float(best / energy.sum()), 3)}
    return sig, cover


def onsets_in(events, t0, n):
    return np.array([e["t"] - t0 for e in events if 0 <= (e["t"] - t0) * FPS < n])


def train(times, n):
    x = np.zeros(n)
    for t in times:
        f = int(round(t * FPS))
        if 0 <= f < n:
            x[f] += 1
    k = np.exp(-0.5 * (np.arange(-2, 3) / 0.7) ** 2)
    return np.convolve(x, k / k.sum(), "same")


def xcorr(a, b, maxlag=4):
    best = -2.0
    for lag in range(-maxlag, maxlag + 1):
        u, v = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        if u.std() > 0 and v.std() > 0:
            best = max(best, float(np.corrcoef(u, v)[0, 1]))
    return best


SUB = 2  # 8th-note grid (informational: see the b note in the docstring)


def phase8(times_song, beats, grid_shift=0.0):
    b = np.asarray(beats) + GRID_LAG + grid_shift
    out = []
    for t in times_song:
        i = np.searchsorted(b, t) - 1
        if 0 <= i < len(b) - 1:
            out.append(2 * np.pi * (((t - b[i]) / ((b[i + 1] - b[i]) / SUB)) % 1.0))
    return np.array(out)


def lock(stroke_ph, onset_ph):
    if len(stroke_ph) < 3 or len(onset_ph) < 3:
        return float("nan")
    zs, zo = np.exp(1j * stroke_ph).mean(), np.exp(1j * onset_ph).mean()
    return float(abs(zs) * np.cos(np.angle(zs) - np.angle(zo)))


def spectral_share(speed):
    f, p = welch(speed - speed.mean(), fs=FPS, nperseg=min(64, len(speed)))
    band = lambda lo, hi: p[(f >= lo) & (f <= hi)].sum()
    tot = band(1.0, 10.0)
    return float((band(3.4, 4.4) + band(7.2, 8.4)) / tot) if tot > 0 else float("nan")


def analyze(video, events_path, t0, box=None, beats_path=None, grid_lag=0.0):
    global GRID_LAG, BEAT_P
    GRID_LAG = grid_lag
    frames = read_frames(video, box)
    n = len(frames)
    sig = hand_signal(frames)
    if sig is not None:
        source, cover = "hand_landmarks", None
        sig = sosfiltfilt(butter(3, 3.0, btype="high", fs=FPS, output="sos"), sig)
    else:
        source = "subbox_flow"
        sig, cover = subbox_signal(frames)
    speed = np.abs(sig)
    ev = json.load(open(events_path))["events"]
    beats = json.load(open(beats_path or config.BEATS))["beats"]
    BEAT_P = float(np.median(np.diff(beats)))
    on = onsets_in(ev, t0, n)
    # strokes = the N strongest speed peaks, N = number of guitar onsets in the clip
    pk, props = find_peaks(speed, distance=2, height=0)
    pk = np.sort(pk[np.argsort(props["peak_heights"])[::-1][:max(3, len(on))]])
    stroke_t = (pk - 0.5) / FPS                              # flow between frames f-1 and f
    on_ph = phase8(on + t0, beats)

    a_real = xcorr(speed, train(on, n))
    b_real = lock(phase8(stroke_t + t0, beats), on_ph)
    ctrl_a, ctrl_b = {}, {}
    for s in (1.0, 2.0, 3.0):
        sh = np.roll(train(on, n), int(round(s * FPS)))
        ctrl_a[f"shift_{s:g}s"] = xcorr(speed, sh)
        shifted_on_t = ((on + s) % (n / FPS))
        ctrl_b[f"shift_{s:g}s"] = lock(phase8(stroke_t + t0, beats), phase8(shifted_on_t + t0, beats))
    other = onsets_in(ev, t0 + 60.0, n)
    ctrl_a["section_+60s"] = xcorr(speed, train(other, n))
    ctrl_b["section_+60s"] = lock(phase8(stroke_t + t0, beats), phase8(other + t0 + 60.0, beats))
    ctrl_b["grid_quarter_beat"] = lock(phase8(stroke_t + t0, beats, BEAT_P / 4), on_ph)
    ma = a_real - max(ctrl_a.values())
    # grid-locked music has the same 8th-note phase in every section, so section_+60s is reported for b but
    # is not a valid b control (a perfectly synced take ties it); b's margin uses the shifts and the grid shift
    mb = b_real - max(v for k, v in ctrl_b.items() if v == v and k != "section_+60s")
    # permutation nulls: circularly shift the motion (>= 6 frames) and re-score, 200 draws
    rng = np.random.default_rng(0)
    ot = train(on, n)
    null_a, null_b = [], []
    for _ in range(200):
        k = int(rng.integers(6, n - 6))
        null_a.append(xcorr(np.roll(speed, k), ot))
        null_b.append(lock(phase8(((stroke_t + k / FPS) % (n / FPS)) + t0, beats), on_ph))
    p_a = float((np.sum(np.array(null_a) >= a_real) + 1) / 201)
    p_b = float((np.sum(np.array(null_b) >= b_real) + 1) / 201)
    zs = np.exp(1j * phase8(stroke_t + t0, beats)).mean()
    return {"video": pathlib.Path(video).name, "signal": source, "subbox": cover, "frames": n,
            "onsets": int(len(on)), "strokes": int(len(pk)),
            "a_xcorr": round(a_real, 3), "a_controls": {k: round(v, 3) for k, v in ctrl_a.items()},
            "b_lock": round(b_real, 3), "b_controls": {k: round(v, 3) for k, v in ctrl_b.items()},
            "b_stroke_R": round(float(abs(zs)), 3),
            "c_rhythm_share": round(spectral_share(speed), 3),
            "p_a": round(p_a, 3), "p_b": round(p_b, 3),
            "sync_margin_a": round(ma, 3), "sync_margin_b": round(mb, 3),
            "sync_margin": round(ma, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video"); p.add_argument("events"); p.add_argument("song_start", type=float)
    p.add_argument("--box"); p.add_argument("--beats"); p.add_argument("--grid-lag", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    box = [int(float(v)) for v in a.box.split(",")] if a.box else None
    r = analyze(a.video, a.events, a.song_start, box, a.beats, a.grid_lag)
    print(json.dumps(r) if a.json else "\n".join(f"{k:16s} {v}" for k, v in r.items()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU lip-sync scorer: mouth aperture (MediaPipe FaceLandmarker) vs vocal energy envelope.

  sync_score.py VIDEO.mp4 AUDIO.wav [--box x0,y0,x1,y1] [--shift SECONDS] [--json]

Aperture = inner-lip gap (landmarks 13-14) / face height (10-152), largest face (or the face whose
center is inside --box). Envelope = 150-4000 Hz band RMS in dB at the video frame rate.
--shift circularly shifts the audio (control).
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import config  # noqa: E402

MODEL = str(config.FACE_MODEL)
MAX_LAG = 8
OPEN_THRESH = 0.035  # inner-lip gap / face height
JAW_OPEN = 0.15      # MediaPipe jawOpen blendshape


def _faces(lm, rgb, x0=0, y0=0, sx=1.0):
    """Landmarks of every face in rgb, mapped back to full-frame pixels."""
    h, w = rgb.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for fi, f in enumerate(res.face_landmarks):
        xs = np.array([p.x * w for p in f]) / sx + x0
        ys = np.array([p.y * h for p in f]) / sx + y0
        bs = {c.category_name: c.score for c in res.face_blendshapes[fi]} if res.face_blendshapes else {}
        out.append((xs, ys, bs.get("jawOpen", np.nan)))
    return out


def _pick(faces, box):
    best, best_area = None, -1.0
    for xs, ys, j in faces:
        if box is not None and not (box[0] <= xs.mean() <= box[2] and box[1] <= ys.mean() <= box[3]):
            continue
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area > best_area:
            best_area, best = area, (xs, ys, j)
    return best


def mouth_track(video, box=None):
    """Full-frame detection, then a tracked crop (3x face height, upscaled to 512) around the last face:
    the short-range face model misses small or mic-occluded faces at full-frame scale."""
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.IMAGE, num_faces=4,
                                        min_face_detection_confidence=0.1, min_face_presence_confidence=0.1,
                                        output_face_blendshapes=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    ap, jaw = np.full(len(frames), np.nan), np.full(len(frames), np.nan)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        seed = None  # (cx, cy, face_h) of the target face
        if box is not None:
            seed = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (box[3] - box[1]) / 1.5)
        # seed pass: full frames until a face is found (also catches the target when no box is given)
        found = {}
        for i, rgb in enumerate(frames):
            hit = _pick(_faces(lm, rgb), box)
            if hit is not None:
                found[i] = hit
        if not found:
            # tiled search (short-range model wants a face to fill the input): every 6th frame until a hit
            for i in range(0, len(frames), 6):
                rgb = frames[i]
                H, W = rgb.shape[:2]
                for side in (H // 4, H // 3, H // 2):
                    for y0 in range(0, max(1, H - side + 1), side // 2):
                        for x0 in range(0, max(1, W - side + 1), side // 2):
                            crop = rgb[y0:y0 + side, x0:x0 + side]
                            sx = 512.0 / side
                            hit = _pick(_faces(lm, cv2.resize(crop, None, fx=sx, fy=sx, interpolation=cv2.INTER_AREA),
                                               x0, y0, sx), box)
                            if hit is not None and (i not in found or
                                                    np.ptp(hit[1]) > np.ptp(found[i][1])):
                                found[i] = hit
                if found:
                    break
        if seed is None and found:
            xs, ys, _ = max(found.values(), key=lambda f: (f[0].max() - f[0].min()) * (f[1].max() - f[1].min()))
            seed = (xs.mean(), ys.mean(), ys.max() - ys.min())
        for i, rgb in enumerate(frames):
            hit = found.get(i)
            if hit is None and seed is not None:
                H, W = rgb.shape[:2]
                side = int(max(3.0 * seed[2], 256))
                x0 = int(np.clip(seed[0] - side / 2, 0, max(0, W - side)))
                y0 = int(np.clip(seed[1] - side / 2, 0, max(0, H - side)))
                crop = rgb[y0:y0 + side, x0:x0 + side]
                sx = 512.0 / max(crop.shape[:2])
                hit = _pick(_faces(lm, cv2.resize(crop, None, fx=sx, fy=sx, interpolation=cv2.INTER_CUBIC),
                                   x0, y0, sx), None)
            if hit is None:
                continue
            xs, ys, j = hit
            seed = (xs.mean(), ys.mean(), ys.max() - ys.min())
            ap[i] = np.hypot(xs[13] - xs[14], ys[13] - ys[14]) / max(np.hypot(xs[10] - xs[152], ys[10] - ys[152]), 1e-6)
            jaw[i] = j
    return ap, jaw, fps


def vocal_envelope(audio, fps, n_frames, shift_s=0.0):
    x, sr = sf.read(str(audio), always_2d=True)
    x = x.mean(axis=1)
    if shift_s:
        x = np.roll(x, int(round(shift_s * sr)))
    x = sosfiltfilt(butter(4, [150, 4000], btype="band", fs=sr, output="sos"), x)
    hop = sr / fps
    env = np.empty(n_frames)
    for k in range(n_frames):
        a, b = int(round(k * hop)), int(round((k + 1) * hop))
        seg = x[a:b]
        env[k] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-8) if seg.size else -160.0
    return env


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def score(sig, env, open_thresh, prefix):
    n = min(len(sig), len(env))
    sig, env = sig[:n], env[:n]
    lags = {}
    for lag in range(-MAX_LAG, MAX_LAG + 1):  # lag > 0: mouth trails the audio
        if lag >= 0:
            lags[lag] = _corr(sig[lag:], env[:n - lag])
        else:
            lags[lag] = _corr(sig[:n + lag], env[-lag:])
    finite = {k: v for k, v in lags.items() if np.isfinite(v)}
    peak_lag = max(finite, key=finite.get) if finite else None
    face = np.isfinite(sig)
    hi, lo = np.percentile(env, 95), np.percentile(env, 5)
    voiced = env > lo + 0.5 * (hi - lo)
    silent = env < lo + 0.25 * (hi - lo)
    open_ = sig > open_thresh
    r = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)
    return {
        f"{prefix}_peak_xcorr": r(finite[peak_lag]) if finite else None,
        f"{prefix}_peak_lag": peak_lag,
        f"{prefix}_xcorr_lag0": r(lags[0]),
        f"{prefix}_closed_in_silence": r((~open_[silent & face]).mean()) if (silent & face).any() else None,
        f"{prefix}_open_in_voice": r(open_[voiced & face].mean()) if (voiced & face).any() else None,
    }


def score_all(ap, jaw, env):
    n = min(len(ap), len(env))
    hi, lo = np.percentile(env[:n], 95), np.percentile(env[:n], 5)
    out = {"frames": int(n), "no_face_frac": round(float(1 - np.isfinite(ap[:n]).mean()), 4),
           "voiced_frac": round(float((env[:n] > lo + 0.5 * (hi - lo)).mean()), 4),
           "env_range_db": round(float(hi - lo), 1)}
    out.update(score(jaw, env, JAW_OPEN, "jaw"))
    out.update(score(ap, env, OPEN_THRESH, "lip"))
    return out


def analyze(video, audio, box=None, shift=0.0):
    ap, jaw, fps = mouth_track(video, box)
    return score_all(ap, jaw, vocal_envelope(audio, fps, len(ap), shift)), (ap, jaw, fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video"); p.add_argument("audio")
    p.add_argument("--box"); p.add_argument("--shift", type=float, default=0.0)
    a = p.parse_args()
    box = [float(v) for v in a.box.split(",")] if a.box else None
    print(json.dumps(analyze(a.video, a.audio, box, a.shift)[0]))


if __name__ == "__main__":
    sys.exit(main())

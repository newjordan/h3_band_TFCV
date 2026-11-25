#!/usr/bin/env python3
"""Stem quality at 24 fps, same envelope as sync_score.py (150-4000 Hz band RMS, dB).

  dyn_range_db  : p95 - p5 of the envelope
  silent_frac   : frames more than 40 dB below p95 (near-silent)
  side_mid      : rms(L-R) / rms(L+R); high = reverb / wide bleed
  corr_vs_dry   : envelope correlation with the dry vocal (bleed check for non-vocal stems)
  win5_*        : medians over 5 s windows from --sung-start to the end, the scale of one
                  H3 shot; whole-song numbers include any instrumental intro

Usage: stem_quality.py OUT.json name=path [name=path ...] [--ref dry] [--sung-start SECONDS]
"""
import argparse
import json

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

FPS = 24
SILENCE_DB = -160.0


def envelope(x, sr):
    mono = np.asarray(x, dtype=np.float64).mean(axis=1)
    if mono.size == 0:
        return np.array([SILENCE_DB])
    try:
        mono = sosfiltfilt(butter(4, [150, 4000], btype="band", fs=sr, output="sos"), mono)
    except ValueError:
        # filtfilt needs more samples than its padding. A preview shorter than that is silence
        # at this bandwidth, not a crash.
        return np.array([SILENCE_DB])
    hop = sr / FPS
    n = int(len(mono) / hop)
    if n <= 0:
        return np.array([SILENCE_DB])
    if float(hop).is_integer():
        take = n * int(hop)
        seg = mono[:take]
        if len(seg) < take:
            return np.array([SILENCE_DB])
        rms = np.sqrt((seg.reshape(n, int(hop)) ** 2).mean(axis=1))
    else:
        rms = np.array([
            np.sqrt(np.mean(mono[int(round(k * hop)):int(round((k + 1) * hop))] ** 2))
            if int(round((k + 1) * hop)) > int(round(k * hop)) else 0.0
            for k in range(n)
        ])
    return 20 * np.log10(rms + 1e-8)


def stats(path, ref_env=None, sung_start=0.0):
    x, sr = sf.read(path, always_2d=True)
    env = envelope(x, sr)
    hi, lo = np.percentile(env, 95), np.percentile(env, 5)
    out = {"dyn_range_db": round(float(hi - lo), 1),
           "silent_frac": round(float((env < hi - 40).mean()), 3),
           "silent_frac_25db": round(float((env < hi - 25).mean()), 3)}
    w = 5 * FPS
    sung = env[int(sung_start * FPS):]
    # Shorter than one 5 s window (a preview, or --sung-start near the end): score
    # the samples we have. An empty list makes np.median warn and return nan.
    span = sung if len(sung) else env
    wins = [span[i:i + w] for i in range(0, len(span) - w + 1, w)] or [span]
    out["win5_dyn_range_db"] = round(float(np.median([np.percentile(v, 95) - np.percentile(v, 5) for v in wins])), 1)
    out["win5_silent_frac"] = round(float(np.median([(v < np.percentile(v, 95) - 25).mean() for v in wins])), 3)
    if x.shape[1] == 2:
        mid, side = x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]
        out["side_mid"] = round(float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-12)), 3)
    if ref_env is not None:
        n = min(len(env), len(ref_env))
        out["corr_vs_dry"] = round(float(np.corrcoef(env[:n], ref_env[:n])[0, 1]), 3)
    return out, env


def main():
    ap = argparse.ArgumentParser(description="Score stem dynamic range, silence and stereo width at 24 fps.")
    ap.add_argument("out", help="JSON path to write")
    ap.add_argument("stems", nargs="+", help="name=path, one per stem")
    ap.add_argument("--ref", help="stem name whose envelope the others are correlated against")
    ap.add_argument("--sung-start", type=float, default=0.0, help="seconds into the file where the 5 s windows begin")
    a = ap.parse_args()
    out_path, sung_start, ref = a.out, a.sung_start, a.ref
    stems = dict(item.split("=", 1) for item in a.stems)
    ref_env = stats(stems[ref], None, sung_start)[1] if ref else None
    res = {k: dict(path=v, **stats(v, None if k == ref else ref_env, sung_start)[0]) for k, v in stems.items()}
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"{'stem':14s} {'dyn_dB':>7s} {'sil40':>6s} {'sil25':>6s} {'w5_dyn':>6s} {'w5_sil':>6s} {'side/mid':>8s} {'corr_dry':>8s}")
    for k, r in res.items():
        print(f"{k:14s} {r['dyn_range_db']:7.1f} {r['silent_frac']:6.3f} {r['silent_frac_25db']:6.3f} "
              f"{r['win5_dyn_range_db']:6.1f} {r['win5_silent_frac']:6.3f} "
              f"{r.get('side_mid', float('nan')):8.3f} {r.get('corr_vs_dry', float('nan')):8.3f}")


if __name__ == "__main__":
    main()

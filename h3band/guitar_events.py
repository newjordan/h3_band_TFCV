#!/usr/bin/env python3
"""Guitar pattern identifier: onsets (spectral flux), stroke direction, beat quantization.

Direction: a downstroke hits low strings first, an upstroke high strings first. Around each onset
(-30..+40 ms) we take the 50%-rise time of the log energy in a low band (70-330 Hz, low-string
fundamentals) and a high band (1-4 kHz, upper-string partials). dt = t_high - t_low;
dt >= +MIN_DT -> down, <= -MIN_DT -> up, otherwise unknown. Every call carries its dt.

Beat grid: beats.json comes from a beat tracker with a constant lag; with --drums the lag is estimated
from the snare (phase of 1.5-6 kHz drum onsets on the local 8th-note grid) and removed before quantizing.

Usage: guitar_events.py GUITAR.wav BEATS.json OUT.json [--drums DRUMS.wav]
"""
import argparse, json

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft

HOP = 64           # 1.33 ms at 48 kHz
NFFT = 1024
MIN_GAP_S = 0.06
MIN_DT_S = 0.006


def band_flux(mag, freqs, lo, hi):
    b = (freqs >= lo) & (freqs < hi)
    lm = np.log1p(100 * mag[b])
    return np.maximum(np.diff(lm, axis=1, prepend=lm[:, :1]), 0).sum(axis=0)


def pick(flux, fps, k=1.5):
    w = int(0.5 * fps) | 1
    med = median_filter(flux, w)
    mad = median_filter(np.abs(flux - med), w) + 1e-9
    thr = med + k * 1.4826 * mad + 0.02 * flux.max()
    local = flux == maximum_filter1d(flux, int(MIN_GAP_S * fps) | 1)
    idx = np.flatnonzero(local & (flux > thr))
    keep, last = [], -1e9
    for i in idx:
        if i - last >= MIN_GAP_S * fps:
            keep.append(i); last = i
    return np.array(keep, dtype=int)


def band_env(mag, freqs, lo, hi):
    b = (freqs >= lo) & (freqs < hi)
    return np.log1p(100 * (mag[b] ** 2).sum(axis=0))


def rise_time(env, s, e, fps):
    seg = env[s:e]
    base, peak = seg[:max(1, int(0.02 * fps))].min(), seg.max()
    return int(np.argmax(seg >= base + 0.5 * (peak - base))) / fps


def spectrum(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    freqs, _, Z = stft(x.mean(axis=1), sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    return np.abs(Z), freqs, sr / HOP, (NFFT / 2) / sr


def local_phase(t, beats):
    j = np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 2)
    return j, (t - beats[j]) / (beats[j + 1] - beats[j])


def grid_lag(drums, beats):
    mag, freqs, fps, t0 = spectrum(drums)
    t = t0 + pick(band_flux(mag, freqs, 1500, 6000), fps, k=4) / fps
    t = t[(t > beats[0]) & (t < beats[-1])]
    _, ph = local_phase(t, beats)
    z = np.exp(2j * np.pi * ph * 2).mean()   # 8th-note grid
    period = float(np.median(np.diff(beats)))
    return float(np.angle(z) / (2 * np.pi * 2) * period), float(abs(z)), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("guitar"); ap.add_argument("beats"); ap.add_argument("out")
    ap.add_argument("--drums")
    a = ap.parse_args()

    beats = np.array(json.load(open(a.beats))["beats"])
    lag = None
    if a.drums:
        off, R, n = grid_lag(a.drums, beats)
        lag = {"offset_s": round(off, 4), "phase_concentration_R": round(R, 3), "snare_onsets": n}
        beats = beats + off
    period = float(np.median(np.diff(beats)))

    mag, freqs, fps, t0 = spectrum(a.guitar)
    full = band_flux(mag, freqs, 70, 6000)
    el, eh = band_env(mag, freqs, 70, 330), band_env(mag, freqs, 1000, 4000)
    onsets = pick(full, fps)

    events = []
    pre, post = int(0.03 * fps), int(0.04 * fps)
    for i in onsets:
        t = t0 + i / fps
        s, e = max(0, i - pre), min(len(el), i + post)
        dt = rise_time(eh, s, e, fps) - rise_time(el, s, e, fps)
        direction = "down" if dt >= MIN_DT_S else "up" if dt <= -MIN_DT_S else "unknown"
        if beats[0] <= t < beats[-1]:
            bi, ph = local_phase(t, beats)
            bi, ph = int(bi), float(ph)
        else:
            bi = 0 if t < beats[0] else len(beats) - 1
            ph = (t - beats[bi]) / period
        sub = int(np.round(ph * 4))
        local = (beats[bi + 1] - beats[bi]) if bi + 1 < len(beats) else period
        grid_t = beats[bi] + sub * local / 4
        if sub >= 4:
            bi, sub = bi + 1, sub - 4
        events.append({"t": round(t, 4), "strength": round(float(full[i]), 2), "direction": direction,
                       "dt_ms": round(dt * 1000, 1), "beat": bi, "sixteenth": sub, "grid_t": round(float(grid_t), 4),
                       "dev_ms": round((t - grid_t) * 1000, 1)})

    known = [ev for ev in events if ev["direction"] != "unknown"]
    on8 = [ev for ev in events if ev["sixteenth"] in (0, 2)]
    off = [ev for ev in events if ev["sixteenth"] in (1, 3)]
    summary = {
        "n_onsets": len(events), "n_known_direction": len(known),
        "known_frac": round(len(known) / max(1, len(events)), 3),
        "down": sum(ev["direction"] == "down" for ev in known), "up": sum(ev["direction"] == "up" for ev in known),
        # sanity, not ground truth: 8th-note strumming puts downs on the 8ths, ups on the 16th offbeats
        "median_dt_ms_on_8ths": round(float(np.median([ev["dt_ms"] for ev in on8])), 1) if on8 else None,
        "median_dt_ms_on_16th_offbeats": round(float(np.median([ev["dt_ms"] for ev in off])), 1) if off else None,
        "down_rate_on_8ths": round(float(np.mean([ev["direction"] == "down" for ev in on8])), 3) if on8 else None,
        "down_rate_on_16th_offbeats": round(float(np.mean([ev["direction"] == "down" for ev in off])), 3) if off else None,
        "on_8ths": len(on8), "on_16th_offbeats": len(off),
        "median_abs_dev_ms": round(float(np.median([abs(ev["dev_ms"]) for ev in events])), 1) if events else None,
        "beat_period_s": round(period, 4), "grid_lag": lag,
    }
    json.dump({"source": f"spectral-flux onsets on {a.guitar}", "method": __doc__.strip(),
               "summary": summary, "events": events}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

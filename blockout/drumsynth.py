"""Bare drum synth for blockout previews: a hit list -> 48 kHz stereo WAV. Numpy only.

Kick (pitch-swept sine and a click), snare (tone plus noise), toms (swept sines), hi-hat (high-passed
noise; open hats are choked by the next closed hat or pedal), crash and ride (long noise tails, the ride
with a ping). Panned by where each piece sits in the kit. It is a timing reference, not a drummer.

    python -m blockout.drumsynth HITS.json OUT.wav [--start S --dur D]
    (HITS.json: {"hits": [...]} from blockout.drums --hits-out, or h3band/drum_events.py)
"""
import json

import numpy as np

from .drums import KIT
from .synth import SR, write_wav


def _hp(x, n=2):
    for _ in range(n):
        x = np.diff(x, prepend=0.0)
    return x


def _sweep(t, f_end, f_start, tau):
    f = f_end + (f_start - f_end) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def voice(piece, vel, rng, open_=False, length=None):
    a = (vel / 127.0) ** 1.5
    if piece == "kick":
        t = np.arange(int(0.45 * SR)) / SR
        s = _sweep(t, 48, 150, 0.035) * np.exp(-t / 0.22)
        s[:150] += 0.5 * rng.standard_normal(150) * np.exp(-np.arange(150) / 40)
        return a * s
    if piece == "snare":
        t = np.arange(int(0.35 * SR)) / SR
        return a * (0.45 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06)
                    + 0.55 * _hp(rng.standard_normal(len(t)), 1) * np.exp(-t / 0.13))
    if piece in ("tom1", "tom2", "floor"):
        f0 = {"tom1": 210, "tom2": 155, "floor": 100}[piece]
        t = np.arange(int(0.7 * SR)) / SR
        return a * _sweep(t, f0, f0 * 1.5, 0.04) * np.exp(-t / (0.25 if piece != "floor" else 0.4))
    if piece in ("hihat", "hihat_pedal"):
        tau = 0.32 if open_ else (0.018 if piece == "hihat_pedal" else 0.035)
        L = int(min(length or 1.0, 1.0) * SR) if open_ else int(0.25 * SR)
        t = np.arange(max(L, 64)) / SR
        g = 0.25 if piece == "hihat_pedal" else 0.45
        s = g * _hp(rng.standard_normal(len(t)), 3) * np.exp(-t / tau)
        if open_ and length:     # choke
            s *= np.exp(-np.maximum(t - length, 0) / 0.01)
        return a * s
    if piece in ("crash", "crash2"):
        t = np.arange(int(2.5 * SR)) / SR
        return a * 0.5 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.9) * np.minimum(1, t / 0.003)
    if piece == "ride":
        t = np.arange(int(1.5 * SR)) / SR
        ping = sum(np.sin(2 * np.pi * f * t) * w for f, w in ((3150, 0.10), (4710, 0.06), (6020, 0.04)))
        return a * (0.18 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.6) + ping * np.exp(-t / 0.35))
    return np.zeros(1)


def render(hits, start=0.0, dur=None, seed=0):
    rng = np.random.default_rng(seed)
    hits = sorted(hits, key=lambda h: h["t"])
    end = dur if dur else max(h["t"] for h in hits) - start + 3.0
    out = np.zeros((int(end * SR) + 1, 2))
    hats = [h for h in hits if h["piece"] in ("hihat", "hihat_pedal")]
    for h in hits:
        t0 = h["t"] - start
        if t0 < -3 or t0 > end:
            continue
        length = None
        if h.get("open"):
            nxt = next((o["t"] for o in hats if o["t"] > h["t"] + 1e-3 and not o.get("open")), None)
            length = (nxt - h["t"]) if nxt else None
        s = voice(h["piece"], h["vel"], rng, bool(h.get("open")), length)
        x = KIT.get(h["piece"] if h["piece"] != "hihat_pedal" else "hihat", {"c": (0, 0, 0)})["c"][0]
        pan = float(np.clip(x / 0.8, -1, 1))
        gains = np.array([np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))])
        i0 = int(round(t0 * SR))
        a, b = max(i0, 0), min(i0 + len(s), len(out))
        if b > a:
            out[a:b] += s[a - i0:b - i0, None] * gains[None, :]
    out /= max(1e-9, np.abs(out).max()) / 0.8
    return out.astype(np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("hits"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--dur", type=float, default=None)
    a = ap.parse_args()
    write_wav(a.out, render(json.load(open(a.hits))["hits"], a.start, a.dur))

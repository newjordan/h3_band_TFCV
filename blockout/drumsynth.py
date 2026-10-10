"""Bare drum synth for blockout previews: a hit list -> 48 kHz stereo WAV. Numpy only.

Kick (pitch-swept sine and a click), snare (tone plus noise), toms (swept sines, pitched by size), hi-hat
(high-passed noise; open hats are choked by the next closed hat or pedal), crash and ride (long noise tails, the
ride with a ping), splash (short and bright), china (ring-modulated noise). Panned by where each piece sits in
the kit. It is a timing reference, not a drummer.
Hits annotated by blockout/drum_teacher.py sound their technique: cross-stick click, rimshot crack, ride bell,
and a choked crash cut off at its choke time.

    python -m blockout.drumsynth HITS.json OUT.wav [--start S --dur D] [--kit big]
    (HITS.json: {"hits": [...]} from blockout.drums --hits-out, or h3band/drum_events.py)
"""
import json

import numpy as np

from . import drum_kit
from .synth import SR, write_wav


def _hp(x, n=2):
    for _ in range(n):
        x = np.diff(x, prepend=0.0)
    return x


def _sweep(t, f_end, f_start, tau):
    f = f_end + (f_start - f_end) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


TOM_PITCH = {"tom1": (0.127, 210), "tom2": (0.152, 155), "floor": (0.20, 100)}   # the standard kit's toms (Hz)


def _ptype(p):
    return p.split("_")[0].rstrip("0123456789")       # drum_kit.piece_type


def _tom_pitch(piece, r):
    """Fundamental (Hz) of a tom of radius r: the standard toms' pitches, about r^-1.7 between and beyond."""
    if piece in TOM_PITCH and (r is None or abs(r - TOM_PITCH[piece][0]) < 1e-6):
        return TOM_PITCH[piece][1]
    return 210 * (0.127 / (r or 0.127)) ** 1.69


def voice(piece, vel, rng, open_=False, length=None, tech=None, r=None):
    """r: the piece's radius (m), which sets a tom's pitch."""
    a = (vel / 127.0) ** 1.5
    if tech == "cross_stick":
        t = np.arange(int(0.12 * SR)) / SR
        return a * (0.6 * np.sin(2 * np.pi * 1250 * t) * np.exp(-t / 0.012)
                    + 0.3 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.006))
    if tech == "rimshot":
        t = np.arange(int(0.35 * SR)) / SR
        return a * (0.4 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06)
                    + 0.9 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.09)
                    + 0.5 * np.sin(2 * np.pi * 920 * t) * np.exp(-t / 0.015))
    if tech == "bell":
        t = np.arange(int(1.5 * SR)) / SR
        ping = sum(np.sin(2 * np.pi * f * t) * w for f, w in ((2380, 0.30), (3570, 0.18), (5010, 0.08)))
        return a * (0.05 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.4) + ping * np.exp(-t / 0.7))
    kind = "hihat_pedal" if piece == "hihat_pedal" else _ptype(piece)
    if kind == "kick":
        t = np.arange(int(0.45 * SR)) / SR
        s = _sweep(t, 48, 150, 0.035) * np.exp(-t / 0.22)
        s[:150] += 0.5 * rng.standard_normal(150) * np.exp(-np.arange(150) / 40)
        return a * s
    if kind == "snare":
        t = np.arange(int(0.35 * SR)) / SR
        return a * (0.45 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06)
                    + 0.55 * _hp(rng.standard_normal(len(t)), 1) * np.exp(-t / 0.13))
    if kind in ("tom", "floor"):
        f0 = _tom_pitch(piece, r)
        t = np.arange(int(0.7 * SR)) / SR
        return a * _sweep(t, f0, f0 * 1.5, 0.04) * np.exp(-t / (0.25 if kind != "floor" else 0.4))
    if kind in ("hihat", "hihat_pedal"):
        tau = 0.32 if open_ else (0.018 if kind == "hihat_pedal" else 0.035)
        L = int(min(length or 1.0, 1.0) * SR) if open_ else int(0.25 * SR)
        t = np.arange(max(L, 64)) / SR
        g = 0.25 if kind == "hihat_pedal" else 0.45
        s = g * _hp(rng.standard_normal(len(t)), 3) * np.exp(-t / tau)
        if open_ and length:     # choke
            s *= np.exp(-np.maximum(t - length, 0) / 0.01)
        return a * s
    if kind == "crash":
        t = np.arange(int(2.5 * SR)) / SR
        return a * 0.5 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.9) * np.minimum(1, t / 0.003)
    if kind == "splash":        # small and bright: gone in about a second
        t = np.arange(int(1.2 * SR)) / SR
        # 0.27: a third difference carries about 1.8x a second difference's level, so it sits with the crash
        return a * 0.27 * _hp(rng.standard_normal(len(t)), 3) * np.exp(-t / 0.3) * np.minimum(1, t / 0.002)
    if kind == "china":         # trashy: noise ring-modulated by a few clashing partials
        t = np.arange(int(2.0 * SR)) / SR
        ring = 0.6 + 0.4 * sum(np.sin(2 * np.pi * f * t) for f in (433, 587, 811)) / 3
        return a * 0.55 * _hp(rng.standard_normal(len(t)), 2) * ring * np.exp(-t / 0.6) * np.minimum(1, t / 0.002)
    if kind == "ride":
        t = np.arange(int(1.5 * SR)) / SR
        ping = sum(np.sin(2 * np.pi * f * t) * w for f, w in ((3150, 0.10), (4710, 0.06), (6020, 0.04)))
        return a * (0.18 * _hp(rng.standard_normal(len(t)), 2) * np.exp(-t / 0.6) + ping * np.exp(-t / 0.35))
    return np.zeros(1)


def render(hits, start=0.0, dur=None, seed=0, kit=None):
    """kit: the kit dict the hits were played on (pans and tunes by it); default the standard kit."""
    kit = kit or drum_kit.STANDARD
    rng = np.random.default_rng(seed)
    hits = sorted(hits, key=lambda h: h["t"])
    end = dur if dur else max(h["t"] for h in hits) - start + 3.0
    out = np.zeros((int(end * SR) + 1, 2))
    hat_of = lambda p: "hihat" if p == "hihat_pedal" else p     # the pedal closes the main hi-hat
    for h in hits:
        t0 = h["t"] - start
        if t0 < -3 or t0 > end:
            continue
        length = None
        if h.get("open"):
            nxt = next((o["t"] for o in hits if o["t"] > h["t"] + 1e-3 and not o.get("open")
                        and hat_of(o["piece"]) == h["piece"]), None)
            length = (nxt - h["t"]) if nxt else None
        s = voice(h["piece"], h["vel"], rng, bool(h.get("open")), length, h.get("tech"),
                  kit.get(h["piece"], {}).get("r"))
        if h.get("choke"):     # blockout/drum_teacher.py: the hand pinches the cymbal at choke["t"]
            ts = np.arange(len(s)) / SR
            s = s * np.exp(-np.maximum(ts - (h["choke"]["t"] - h["t"]), 0) / 0.03)
        x = kit.get(hat_of(h["piece"]), {"c": (0, 0, 0)})["c"][0]
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
    ap.add_argument("--kit", help="drum_kit.py preset or kit spec JSON the hits were played on")
    a = ap.parse_args()
    hits, kit = json.load(open(a.hits))["hits"], None
    if a.kit:       # hits named for the standard kit (drum_events.py) are pointed at this kit's pieces
        kit, notes = drum_kit.load(a.kit)
        for n in notes:
            print(n)
        hits = drum_kit.remap(hits, kit)
    write_wav(a.out, render(hits, a.start, a.dur, kit=kit))

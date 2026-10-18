"""Sampled grand piano for blockout audio: performed notes -> 48 kHz stereo WAV.

Plays an SFZ+FLAC piano (Salamander Grand Piano V3, Yamaha C5, CC-BY 3.0, by Alexander Holm; FreePats
packaging) with no SFZ engine: picks the region by key and velocity, repitches by resampling, applies the
SFZ velocity curve, and damps each note at the first pedal change after its key is released. Pedal is
automatic: it changes when the bass moves (and at least once a bar), the way Debussy is usually pedalled.
A light synthetic room reverb glues it together.

    python -m blockout.sampler NOTES.json OUT.wav --sfz PATH/SalamanderGrandPiano-V3+20200602.sfz
           [--dur S] [--downbeats JSON]

NOTES.json: [[start, end, pitch, velocity, track], ...] (piano.py --notes-out). Needs soundfile + scipy.
"""
import json, pathlib, re

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 48000


def load_sfz(path):
    path = pathlib.Path(path)
    regions, group = [], {}
    for line in path.read_text().splitlines():
        line = line.split("//")[0].strip()
        if line.startswith("<group>"):
            group = dict(re.findall(r"(\w+)=(\S+)", line))
        elif line.startswith("<region>") and "trigger" not in group:
            r = {**group, **dict(re.findall(r"(\w+)=(\S+)", line))}
            if "sample" in r and "pitch_keycenter" in r:
                regions.append({"file": path.parent / r["sample"], "lokey": int(r["lokey"]), "hikey": int(r["hikey"]),
                                "lovel": int(r.get("lovel", 1)), "hivel": int(r.get("hivel", 127)),
                                "key": int(r["pitch_keycenter"]), "veltrack": float(r.get("amp_veltrack", 100)) / 100,
                                "release": float(r.get("ampeg_release", 1.0))})
    return regions


class Piano:
    def __init__(self, sfz):
        self.regions = load_sfz(sfz)
        self.cache = {}

    def region(self, pitch, vel):
        for r in self.regions:
            if r["lokey"] <= pitch <= r["hikey"] and r["lovel"] <= vel <= r["hivel"]:
                return r
        return min(self.regions, key=lambda r: abs(r["key"] - pitch) + abs(r["lovel"] - vel) / 10)

    def sample(self, f):
        if f not in self.cache:
            x, sr = sf.read(str(f), dtype="float32", always_2d=True)
            assert sr == SR, f"{f}: {sr} Hz"
            self.cache[f] = x
        return self.cache[f]

    def note(self, pitch, vel, length):
        """One note, `length` seconds long before its release tail."""
        r = self.region(pitch, vel)
        x = self.sample(r["file"])
        ratio = 2 ** ((pitch - r["key"]) / 12)
        n_out = int(min(len(x) / ratio, (length + r["release"]) * SR))
        src = np.arange(n_out) * ratio
        y = np.stack([np.interp(src, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], axis=1)
        gain = (1 - r["veltrack"]) + r["veltrack"] * (vel / 127) ** 2
        t = np.arange(n_out) / SR
        env = np.where(t < length, 1.0, np.exp(-(t - length) / (r["release"] / 5)))
        return (y * (gain * env)[:, None]).astype(np.float32)


def auto_pedal(notes, downbeats=()):
    """Pedal change times: whenever the bass (lowest new note of the lower track) moves, and every downbeat."""
    lower = max({n[4] for n in notes})
    by_t = {}
    for s, e, p, v, trk in notes:
        if trk == lower:
            by_t.setdefault(round(s, 2), []).append(p)
    ch, last = set(), None
    for t in sorted(by_t):
        b = min(by_t[t])
        if b != last:
            ch.add(t); last = b
    ch |= {round(d, 2) for d in downbeats}
    return np.array(sorted(ch))


def harmony_pedal(notes, delay=0.12, settle=0.10, k=2, win=0.03, gap=0.08):
    """A pianist's pedal for the working piano (blockout/action.py), from the written notes. Two sources of change:
    the bass really moving (the new lowest note of the lower track is at or below every lower-track note still written
    to sound, ignoring ones that end within `settle` s; broken-chord bass notes don't count), changed `delay` s after
    the onset so legato releases fall first (syncopated pedal); and clusters of >= k written note ends within `win` s,
    which in a transcription of a pedalled recording are the player's own pedal lifts. Changes closer than `gap` merge.
    Against auto_pedal on the Knight's Suite it halves the song's blur (sound past the written end, mean 0.48 -> 0.25 s)
    and changes about a third as often. auto_pedal stays the default of render() (the H3 audio paths use it)."""
    low = max({n[4] for n in notes})
    L = [n for n in notes if n[4] == low]
    by = {}
    for n in L:
        by.setdefault(round(n[0], 3), []).append(n[2])
    ts = sorted(by)
    ch = []
    for i, t in enumerate(ts):
        live = [n[2] for n in L if n[0] < t - 1e-3 and n[1] > t + settle]
        if not live or min(by[t]) <= min(live):
            ch.append(min(t + delay, ts[i + 1] - 0.01) if i + 1 < len(ts) else t + delay)
    E = np.sort([n[1] for n in notes])
    i = 0
    while i < len(E):
        j = i
        while j + 1 < len(E) and E[j + 1] - E[i] < win:
            j += 1
        if j - i + 1 >= k:
            ch.append(E[j] + 0.01)
        i = j + 1
    out = []
    for t in sorted(ch):
        if out and t - out[-1] < gap:
            out[-1] = max(out[-1], t)
        else:
            out.append(t)
    return np.array(out)


def room_ir(seconds=2.2, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t / (seconds / 6.9))[:, None]
    k = np.ones(24) / 24                                          # darken the tail
    ir = np.stack([np.convolve(ir[:, c], k, "same") for c in range(2)], 1)
    ir[: int(0.012 * SR)] = 0                                      # pre-delay
    return ir / np.abs(ir).sum(0).max() * 6


def render(notes, sfz, dur=None, downbeats=(), wet=0.16, pedal=None, norm=True):
    """pedal: None = auto_pedal(notes, downbeats); an array of pedal-change times; "off" = no pedal (each damper
    falls at its note's end). norm=False returns the raw mix (to put two renders at one gain)."""
    pno = Piano(sfz)
    if pedal is None:
        pedal = auto_pedal(notes, downbeats)
    elif isinstance(pedal, str):
        pedal = np.zeros(0)
    end = dur or max(n[1] for n in notes) + 4.0
    out = np.zeros((int(end * SR) + SR, 2), np.float32)
    for s, e, p, v, trk in notes:
        if s >= end:
            continue
        k = np.searchsorted(pedal, e)                              # damper falls at the next pedal change
        damp = e if not len(pedal) else pedal[k] + 0.04 if k < len(pedal) else e + 3.0
        y = pno.note(p, max(1, min(127, int(v))), max(damp, e) - s)
        i = int(s * SR)
        n = min(len(y), len(out) - i)
        out[i:i + n] += y[:n]
    out = out[: int(end * SR)]
    if wet:
        rev = np.stack([fftconvolve(out[:, c], room_ir()[:, c])[: len(out)] for c in range(2)], 1)
        out = out + wet * rev
    return out / max(1e-9, np.abs(out).max()) * 0.89 if norm else out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("notes"); ap.add_argument("out")
    ap.add_argument("--sfz", required=True)
    ap.add_argument("--dur", type=float); ap.add_argument("--downbeats")
    a = ap.parse_args()
    notes = json.load(open(a.notes))
    db = json.load(open(a.downbeats)) if a.downbeats else ()
    sf.write(a.out, render(notes, a.sfz, a.dur, db), SR, subtype="PCM_16")
    print("wrote", a.out)

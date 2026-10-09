"""Bare additive piano for blockout previews: MIDI notes -> 48 kHz stereo WAV. Numpy only.

Stiff-string partials (inharmonicity B), per-partial exponential decay, a short hammer click, damper release
at note-off. It is a timing reference, not a performance.

    python3 -m blockout.synth NOTES.mid OUT.wav [--start S --dur D --speed X]
"""
import numpy as np

SR = 48000


def render(notes, start=0.0, dur=None, speed=1.0, sustain=0.0):
    notes = [(n.start / speed - start, n.end / speed - start, n.pitch, n.velocity) for n in notes]
    end = dur if dur else max(e for _, e, _, _ in notes) + 2.0
    out = np.zeros(int(end * SR) + 1)
    rng = np.random.default_rng(0)
    for t0, t1, p, vel in notes:
        if t1 < 0 or t0 > end:
            continue
        f0 = 440.0 * 2 ** ((p - 69) / 12)
        ring = min(6.0, 2.0 + 60.0 / f0) if sustain else 0.0
        L = max(t1 - t0, 0.05) + 0.4 + ring
        t = np.arange(int(L * SR)) / SR
        B = 0.0002 * (f0 / 261.6) ** 0.5
        s = np.zeros_like(t)
        for k in range(1, 16):
            fk = k * f0 * np.sqrt(1 + B * k * k)
            if fk > SR / 2.2:
                break
            amp = (vel / 127) ** (1 + 0.15 * k) / k ** 1.1
            s += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * (0.4 + 0.25 * k) * (f0 / 261.6) ** 0.4)
        s *= np.minimum(1.0, t / 0.002)
        click = rng.standard_normal(int(0.006 * SR)) * np.exp(-np.arange(int(0.006 * SR)) / (0.0015 * SR))
        s[:len(click)] += 0.05 * (vel / 127) * click
        off = max(t1 - t0, 0.05) + ring
        s *= np.where(t < off, 1.0, np.exp(-(t - off) / 0.08))
        i0 = int(t0 * SR)
        a, b = max(i0, 0), min(i0 + len(s), len(out))
        if b > a:
            out[a:b] += s[a - i0:b - i0]
    out /= max(1e-9, np.abs(out).max()) / 0.8
    return np.stack([out, out], axis=1).astype(np.float32)


def write_wav(path, x):
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import argparse
    from . import midi
    ap = argparse.ArgumentParser()
    ap.add_argument("mid"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--sustain", action="store_true")
    a = ap.parse_args()
    write_wav(a.out, render(midi.read(a.mid), a.start, a.dur, a.speed, a.sustain))

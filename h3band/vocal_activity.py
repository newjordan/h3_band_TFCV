#!/usr/bin/env python3
"""Vocal activity on the 24 fps grid and on H3's latent-frame grid.

A frame is voiced if the dry-vocal envelope (150-4000 Hz RMS, dB) is within VOICED_DB of the song's p95,
or an aligned lyric word covers it. H3 latent frame of pixel frame f (counted from the clip's first
frame): clip, r = divmod(f, 17); clip*5 + (0 if r == 0 else 1 + (r-1)//4).

Song-global grid (clip starts at song t=0) is written by default; a shot gets its own grid with
--start SECONDS --frames N (shot-local f), since the latent grid restarts at every render's frame 0.

Usage: vocal_activity.py DRY.wav OUT.json [--align align.json] [--start S --frames N]
"""
import argparse, json

import numpy as np
import soundfile as sf

from stem_quality import envelope

FPS = 24
VOICED_DB = 40.0  # legato sung vowels sit 25-40 dB under the chorus peaks; -25 dB drops them


def latent_index(f):
    clip, r = divmod(f, 17)
    return clip * 5 + (0 if r == 0 else 1 + (r - 1) // 4)


def grid(voiced, words_cov):
    lat = {}
    for f, v in enumerate(voiced):
        lat.setdefault(latent_index(f), []).append(bool(v))
    return [{"latent": k, "frames": len(v), "voiced_frac": round(sum(v) / len(v), 3), "voiced": any(v)}
            for k, v in sorted(lat.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dry"); ap.add_argument("out")
    ap.add_argument("--align"); ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int)
    a = ap.parse_args()

    x, sr = sf.read(a.dry, always_2d=True, dtype="float32")
    env = envelope(x, sr)
    p95 = float(np.percentile(env, 95))
    voiced_e = env > p95 - VOICED_DB
    cov = np.zeros(len(env), dtype=bool)
    if a.align:
        for s, e, *_ in json.load(open(a.align))["words"]:
            cov[int(s * FPS):int(np.ceil(e * FPS))] = True
    voiced = voiced_e | cov if a.align else voiced_e

    f0 = int(round(a.start * FPS))
    n = a.frames if a.frames else len(voiced) - f0
    seg_v, seg_e, seg_c = voiced[f0:f0 + n], voiced_e[f0:f0 + n], cov[f0:f0 + n]
    out = {
        "source": a.dry, "fps": FPS, "start_s": a.start, "frames": int(len(seg_v)),
        "rule": f"energy > p95 - {VOICED_DB:.0f} dB (p95 = {p95:.1f} dB)" + (" OR aligned word" if a.align else ""),
        "voiced_frac": round(float(seg_v.mean()), 3),
        "energy_only_voiced_frac": round(float(seg_e.mean()), 3),
        "word_only_voiced_frac": round(float(seg_c.mean()), 3) if a.align else None,
        "energy_word_agreement": round(float((seg_e == seg_c).mean()), 3) if a.align else None,
        "pixel_frames": "".join("1" if v else "0" for v in seg_v),
        "latent_frames": grid(seg_v, seg_c),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("pixel_frames", "latent_frames")})
    print("latent frames:", len(out["latent_frames"]),
          "voiced:", sum(l["voiced"] for l in out["latent_frames"]))


if __name__ == "__main__":
    main()

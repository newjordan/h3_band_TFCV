"""Drummer emotion: four sliders, named presets built from them, and a timeline that crossfades between presets.

Sliders (0..1):
  force  how hard every stroke is: stroke height and how much of the arm drives it
  range  dynamic range: how far apart a ghost note and an accent look (0 = every stroke alike)
  body   how much the torso leans into big strokes, on top of the head style
  flair  showmanship: how high the stick follows through after a crash

Presets (and the performer.py head style each one uses):
  calm     wrist strokes, small range, still body              head "focused"
  groove   the neutral player                                   head "groove"
  intense  big arm strokes, wide range, leans into accents      head "wild"
  showy    big strokes, widest range, cymbal follow-through     head "crowd"

A timeline is "0:calm,8:intense,16:showy" (seconds). Slider overrides apply on top of every preset.
"""
import numpy as np

SLIDERS = ("force", "range", "body", "flair")
PRESETS = {
    "calm":    ({"force": 0.30, "range": 0.35, "body": 0.15, "flair": 0.00}, "focused"),
    "groove":  ({"force": 0.55, "range": 0.55, "body": 0.45, "flair": 0.25}, "groove"),
    "intense": ({"force": 0.85, "range": 0.75, "body": 0.85, "flair": 0.50}, "wild"),
    "showy":   ({"force": 0.75, "range": 0.90, "body": 0.70, "flair": 1.00}, "crowd"),
}
XFADE = 0.6          # s, like performer.STYLE_XFADE
NEUTRAL = PRESETS["groove"][0]

TIERS = ((0.06, "ghost", "wrist"), (0.16, "tap", "wrist"), (0.32, "accent", "forearm"), (9.0, "full", "full arm"))


def parse(timeline, overrides=None):
    """'0:calm,8:intense' -> [(t, sliders dict, head style)], sorted. overrides: {slider: value} or None."""
    out = []
    for item in timeline.split(","):
        t, name = item.split(":")
        name = name.strip()
        if name not in PRESETS:
            raise ValueError(f"unknown emotion {name!r}; presets: {', '.join(PRESETS)}")
        sl = dict(PRESETS[name][0])
        for k, v in (overrides or {}).items():
            if k not in SLIDERS:
                raise ValueError(f"unknown slider {k!r}; sliders: {', '.join(SLIDERS)}")
            if v is not None and v >= 0:
                sl[k] = float(v)
        out.append((float(t), sl, PRESETS[name][1]))
    return sorted(out, key=lambda x: x[0])


def at(t, sched):
    """Crossfaded slider values at time t."""
    cur = dict(sched[0][1])
    for t0, sl, _ in sched[1:]:
        if t < t0:
            break
        u = min(1.0, (t - t0) / XFADE)
        u = u * u * (3 - 2 * u)
        cur = {k: cur[k] + (sl[k] - cur[k]) * u for k in SLIDERS}
    return cur


def head_schedule(sched):
    return [(t, head) for t, _, head in sched]


def height(vel, s, base):
    """Stroke height (m) for MIDI velocity vel under sliders s; base is drums.stroke_height. The groove preset
    gives base(vel) unchanged."""
    v = float(np.clip(80 + (vel - 80) * s["range"] / NEUTRAL["range"], 1, 127))
    return base(v) * (1.0 + s["force"] - NEUTRAL["force"])


def tier(h):
    for top, name, drive in TIERS:
        if h < top:
            return name, drive
    return TIERS[-1][1:]


def arm_lift(h):
    """Fraction of the tip's height the hand rises: ghost strokes are all wrist, full strokes lift the whole arm."""
    return 0.15 + 0.55 * float(np.clip((h - 0.04) / 0.42, 0.0, 1.0))


def lean_kick(h, s):
    """Forward torso lean (performer lean units) a stroke of height h adds under sliders s."""
    return s["body"] * 0.30 * float(np.clip((h - 0.20) / 0.25, 0.0, 1.0))


def rebound(piece, s):
    """Rebound fraction of the prep height; crashes follow through higher with flair (the ride is played, not
    swung at)."""
    return 0.5 + (0.7 * s["flair"] if piece in ("crash", "crash2") else 0.0)

"""Feeling chart -> performance: turns flat MIDI (one tempo, one velocity) into a played performance.

The chart (examples/piano/*.feeling.json) gives, per bar: strength (pp..ff), touch (caress..strike),
freedom (how much each note varies), tempo (rubato) and head style. Between rows values are interpolated.
On top of the chart, three things a pianist does without being told:
- phrase breathing: every 2-bar phrase swells a little and falls back,
- voicing: the top note of a right-hand chord sings, inner notes recede, the bass is firm,
- variation: each note's timing, velocity, touch and lift wander, scaled by freedom (seeded, repeatable).

Motion and audio are both built from the result, so they stay locked together.
"""
import json, math

import numpy as np

from .midi import Note


class Chart:
    def __init__(self, path, downbeats):
        d = json.load(open(path))
        self.rows = d["rows"]
        self.db = np.asarray(downbeats)
        self.bar_len = float(np.median(np.diff(self.db)))
        bars = np.array([r["bar"] for r in self.rows], float)
        self.t_rows = np.interp(bars - 1, np.arange(len(self.db)), self.db,
                                right=self.db[-1] + (bars.max() - len(self.db)) * self.bar_len)
        self.marks = [(t, r["mark"]) for t, r in zip(self.t_rows, self.rows) if r.get("mark")]
        heads, last = [], None
        for t, r in zip(self.t_rows, self.rows):
            if r.get("head") and r["head"] != last:
                heads.append((t, r["head"])); last = r["head"]
        self.heads = heads

    def at(self, key, t):
        """Value of a field at score time t (seconds in the original MIDI)."""
        v = np.array([r.get(key, np.nan) for r in self.rows], float)
        ok = ~np.isnan(v)
        return np.interp(t, self.t_rows[ok], v[ok])

    def breath(self, t):
        """Phrase swell: +-0.06 over 2-bar phrases, peaking a little past the middle."""
        ph = ((t - self.db[0]) / (2 * self.bar_len)) % 1.0
        return 0.06 * math.sin(math.pi * ph ** 0.8)


class TimeMap:
    """Score time -> performance time, by integrating 1/tempo over the score."""

    def __init__(self, chart, t_end, dt=0.01):
        self.ts = np.arange(0, t_end + 1.0, dt)
        rate = 1.0 / np.maximum(chart.at("tempo", self.ts), 0.2)
        self.tp = np.concatenate([[0], np.cumsum(rate[:-1] * dt)])

    def __call__(self, t):
        return np.interp(t, self.ts, self.tp)


def perform(notes, chart, seed=0):
    """Returns (performed notes with velocity set and a .touch/.lift attribute, time map)."""
    rng = np.random.default_rng(seed)
    t_end = max(n.end for n in notes)
    tm = TimeMap(chart, t_end)
    groups = {}
    for n in notes:
        groups.setdefault((n.track, round(n.start, 3)), []).append(n)
    tracks = sorted({n.track for n in notes})
    rh = tracks[0] if tracks else 1
    out = []
    for (trk, t0), ch in sorted(groups.items(), key=lambda kv: kv[0][1]):
        ch.sort(key=lambda n: n.pitch)
        s, touch, free = chart.at("strength", t0), chart.at("touch", t0), chart.at("freedom", t0)
        s = s + chart.breath(t0)
        # a chord's notes land together (one hand), but the hand as a whole drifts
        dt_hand = rng.normal(0, 0.006 + 0.014 * free)
        for k, n in enumerate(ch):
            voice = 0.0
            if trk == rh:
                voice = 0.10 if k == len(ch) - 1 else (-0.06 if len(ch) > 1 else 0.0)
            else:
                voice = 0.04 if k == 0 else -0.05
            v = s + voice + rng.normal(0, 0.03 + 0.05 * free)
            vel = int(np.clip(round(18 + 100 * v), 8, 120))
            # chord notes spread by a few ms (rolled a hair), more when free
            dt_note = dt_hand + (k * rng.uniform(0, 0.004 + 0.008 * free) if len(ch) > 1 else 0.0)
            start = float(tm(n.start)) + dt_note
            end = float(tm(n.end)) + dt_hand + rng.normal(0, 0.01 * free)
            p = Note(max(0.0, start), max(start + 0.05, end), n.pitch, vel, n.track)
            p.touch = float(np.clip(touch + rng.normal(0, 0.05 * free) + 0.15 * max(0, v - 0.6), 0, 1))
            p.lift = float(np.clip(1 + rng.normal(0, 0.25 * free), 0.4, 1.8))
            out.append(p)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out, tm


def plot(chart, tm, out_png, title):
    """The feeling chart as a picture for the review site (needs matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.linspace(0, chart.db[-1] + chart.bar_len, 2000)
    tp = tm(ts)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "axes.labelcolor": "#ddd",
                         "xtick.color": "#bbb", "ytick.color": "#bbb", "text.color": "#ddd"})
    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True, facecolor="#17191c",
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})
    for ax in axes:
        ax.set_facecolor("#17191c")
        ax.grid(axis="x", color="#2a2d32", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a = axes[0]
    st = np.array([chart.at("strength", t) + chart.breath(t) for t in ts])
    a.fill_between(tp, 0, st, color="#e0b04a", alpha=0.25, lw=0)
    a.plot(tp, st, color="#e0b04a", lw=1.8, label="strength (with phrase breathing)")
    a.plot(tp, chart.at("touch", ts), color="#7cb8e0", lw=1.8, label="touch: 0 caress, 1 strike")
    a.plot(tp, chart.at("freedom", ts), color="#9a9c9f", lw=1.2, ls="--", label="freedom (variation)")
    for y, lab in ((0.12, "pp"), (0.25, "p"), (0.55, "mf"), (0.72, "f"), (0.88, "ff")):
        a.axhline(y, color="#2a2d32", lw=0.6)
        a.text(-2, y, lab, ha="right", va="center", fontsize=9, color="#9a9c9f")
    for t, m in chart.marks:
        a.annotate(m, (float(tm(t)), 1.0), rotation=28, fontsize=8, color="#cfcfcb", ha="left", va="bottom",
                   annotation_clip=False)
        a.axvline(float(tm(t)), color="#3a3d42", lw=0.6)
    a.set_ylim(0, 1.0)
    a.legend(loc="upper right", frameon=False, fontsize=9)
    b = axes[1]
    b.plot(tp, chart.at("tempo", ts), color="#c48ad8", lw=1.6)
    b.axhline(1, color="#2a2d32", lw=0.6)
    b.set_ylabel("tempo x")
    c = axes[2]
    cols = {"expressive": "#e0b04a", "focused": "#7cb8e0", "groove": "#7cc48a", "wild": "#e06a5a", "crowd": "#c48ad8"}
    hs = chart.heads + [(ts[-1], None)]
    for (t0, h), (t1, _) in zip(hs, hs[1:]):
        c.axvspan(float(tm(t0)), float(tm(t1)), color=cols.get(h, "#888"), alpha=0.55, lw=0)
        c.text(float(tm(t0)) + 1, 0.5, h, va="center", fontsize=9, color="#111")
    c.set_yticks([]); c.set_ylabel("head")
    c.set_xlabel("performance time (s)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.9))
    fig.savefig(out_png, dpi=110, facecolor=fig.get_facecolor())

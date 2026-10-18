"""Piano roll of what the hands played on the working piano (blockout/action.py) against the score, one window.
Usage: python tools/v18/action_roll.py OUT_DIR NOTES.json T0 T1 PNG [--title TEXT]

Grey bars: the written notes. Blue: what sounded (blow to damper). Red ring: a written note that never sounded.
Yellow diamond: a blow the score does not have (a wrong key). x is video time (suite time - 39.9 s)."""
import json, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from blockout.action import match, name

SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"
WRITTEN, PLAYED, MISSED, WRONG = "#6f6e69", "#3987e5", "#d03b3b", "#fab219"
V0 = 39.9


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out, notes, t0, t1, png = a[0], json.load(open(a[1])), float(a[2]), float(a[3]), a[4]
    title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else ""
    blows = json.load(open(f"{out}/played.json"))
    m = match(blows, notes, t0 - 1, t1 + 1)
    used = set(m.values())
    W = [i for i, n in enumerate(notes) if n[1] > t0 and n[0] < t1]
    B = [j for j, b in enumerate(blows) if b["t_off"] > t0 and b["t_on"] < t1]
    ps = [notes[i][2] for i in W] + [blows[j]["pitch"] for j in B]
    lo, hi = min(ps) - 1, max(ps) + 1
    fig, ax = plt.subplots(figsize=(16, 6.4), dpi=110)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for i in W:
        s, e, p = notes[i][:3]
        ax.barh(p, e - s, left=s - V0, height=0.78, color=WRITTEN, edgecolor="none")
    for j in B:
        b = blows[j]
        ax.barh(b["pitch"], b["t_off"] - b["t_on"], left=b["t_on"] - V0, height=0.36, color=PLAYED, edgecolor="none")
    miss = [i for i in W if i not in m and t0 <= notes[i][0] < t1]
    ax.scatter([notes[i][0] - V0 for i in miss], [notes[i][2] for i in miss], s=90, facecolors="none",
               edgecolors=MISSED, linewidths=2, zorder=5)
    wrong = [j for j in B if j not in used and t0 <= blows[j]["t_on"] < t1
             and not any(abs(blows[j]["t_on"] - notes[i][0]) < 0.4 and notes[i][2] == blows[j]["pitch"] for i in W)]
    ax.scatter([blows[j]["t_on"] - V0 for j in wrong], [blows[j]["pitch"] for j in wrong], s=70, marker="D",
               color=WRONG, edgecolors=SURF, linewidths=1.5, zorder=6)
    ax.set_xlim(t0 - V0, t1 - V0); ax.set_ylim(lo - 0.5, hi + 0.5)
    yt = [p for p in range(lo, hi + 1) if p % 12 in (0, 5)]
    ax.set_yticks(yt); ax.set_yticklabels([name(p) for p in yt], color=INK2, fontsize=10)
    ax.tick_params(axis="x", colors=INK2, labelsize=10); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("video time (s)", color=INK2, fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    n_in = sum(t0 <= notes[i][0] < t1 for i in W)
    ax.set_title(f"{title}   written {n_in}, never sounded {len(miss)}, wrong keys {len(wrong)}", color=INK,
                 fontsize=13, loc="left", pad=28)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(color=WRITTEN, label="written"), Patch(color=PLAYED, label="sounded (blow to damper)"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=MISSED, mew=2, ms=9, label="written, never sounded"),
          Line2D([], [], marker="D", ls="", color=WRONG, ms=8, label="wrong key")]
    lg = ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=10.5)
    for t in lg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(png, facecolor=SURF)
    print("wrote", png)


if __name__ == "__main__":
    main()

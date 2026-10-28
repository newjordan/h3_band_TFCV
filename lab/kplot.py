"""Leg K plot: hand-region fidelity per H3 run (lines + dots). python lab/kplot.py"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(L, "k_results.csv"))))
x = range(len(rows))
fig, ax = plt.subplots(figsize=(9, 4.8))
for k, c, lab in (("hand_r", "#c05621", "edge structure inside the hand mask (r)"), ("flow_cos", "#2b6cb0", "motion direction agreement (cos)"),
                  ("frame_r", "#718096", "whole-frame structure (r)")):
    ax.plot(list(x), [float(r[k]) for r in rows], "-o", color=c, label=lab, ms=5)
ax.set_xticks(list(x)); ax.set_xticklabels([r["id"] for r in rows], rotation=45, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("agreement with the Blender plate"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("H3 render vs Blender plate, shot kh111 (124 frames)"); fig.tight_layout()
os.makedirs(os.path.join(L, "plots"), exist_ok=True); fig.savefig(os.path.join(L, "plots", "k_handfidelity.png"), dpi=140)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot([float(r["frame_r"]) for r in rows], [float(r["hand_r"]) for r in rows], "o", color="#c05621")
for r in rows: ax.annotate(r["id"], (float(r["frame_r"]), float(r["hand_r"])), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel("whole-frame structure r"); ax.set_ylabel("hand-region structure r"); ax.grid(alpha=0.3)
ax.set_title("The keyboard carries the whole-frame number"); fig.tight_layout(); fig.savefig(os.path.join(L, "plots", "k_frame_vs_hand.png"), dpi=140)

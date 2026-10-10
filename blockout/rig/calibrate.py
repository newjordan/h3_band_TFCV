"""Calibrate the rig's comfort centre (its relaxed playing pose) to measured human pianists.

    python -m blockout.rig.calibrate          # writes blockout/rig/comfort.json

Finds the joint angles whose pose, measured exactly like the human reference (blockout/human_score.py on MediaPipe
landmarks of real pianists, blockout/human_ref.json), matches the human per-finger medians: MCP/PIP/DIP flexion,
the thumb's MCP/IP flexion, tip pitch and spread, and the spacing between fingers. Hand level, palm down.
"""
import json, math, os

import numpy as np
from scipy.optimize import least_squares

from .. import human_score as hs
from .hand import HandRig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "comfort.json")
FN = ["index", "middle", "ring", "pinky"]


def targets(ref):
    pf, m = ref["per_finger"], ref["measures"]
    t = {}
    for k in ("mcp_flex", "pip_flex", "dip_flex"):
        for i, n in enumerate(FN):
            t[(k, i)] = pf[f"{k}.{n}"]["p50"]
    for k in ("thumb_mcp_flex", "thumb_ip_flex", "thumb_tip_pitch", "thumb_abd", "abd_im", "abd_mr", "abd_rp"):
        t[(k, None)] = m[k]["pooled"]["p50"]
    return t


def pose_points(rig, q):
    Rr, pr = rig.root_from_hand_frame(np.eye(3) if rig.side == "R" else np.diag([-1.0, 1, 1]), np.zeros(3))
    x = np.concatenate([pr, np.zeros(3), q])
    _, rp, pts = rig.points(x, Rr)
    return np.asarray([rp] + [p for P in pts for p in P], float)


def calibrate(side="R", ref_path=None):
    ref = json.load(open(ref_path or hs.REF))
    T = targets(ref)
    rig = HandRig(side)

    def resid(q):
        r = hs.measures(pose_points(rig, q), side)
        out = []
        for (k, i), v in T.items():
            got = r[k][i] if i is not None else r[k]
            out.append((got - v) / 5.0)                       # degrees, scaled
        return np.array(out)
    sol = least_squares(resid, np.clip(rig.q_rest, rig.lo + 1e-3, rig.hi - 1e-3), bounds=(rig.lo, rig.hi))
    r = hs.measures(pose_points(rig, sol.x), side)
    report = {f"{k}{'' if i is None else '.' + FN[i]}": (round(r[k][i] if i is not None else r[k], 1), round(v, 1))
              for (k, i), v in T.items()}
    return sol.x, report


if __name__ == "__main__":
    out = {}
    for side in "RL":
        q, rep = calibrate(side)
        out[side] = q.tolist()
        print(side, "ours vs human median:", rep)
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT)

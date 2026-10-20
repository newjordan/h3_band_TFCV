"""recamera.py EDIT.json OUT.json --swap TYPE[,TYPE...] --safe TYPE[,TYPE...] : the same edit (same cuts, same timing),
with every shot whose camera type is in --swap re-shot on a type from --safe (blockout/cameras.py's own shot tracks:
backed off until the hands stay in frame, swung round if it would pass through an arm), never the same camera as the
shot before or after (no jump cuts). Seen from above, curled fingers point at the camera: H3 can't put skeleton fingers
on the keys from the rig's joints there, and a key going down moves straight at the lens, so overhead-style shots swap."""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from blockout import cameras as C


def shots_of(F):
    out, prev = [], None
    for i, fr in enumerate(F):
        s = fr["cam"].get("shot")
        if s != prev:
            out.append([i, i + 1, s]); prev = s
        else:
            out[-1][1] = i + 1
    return out


def kind(shot):
    return (shot.rsplit("_", 1)[0], shot.rsplit("_", 1)[1]) if shot[-2:] in ("_L", "_R") else (shot, None)


def track(name, side, F, f0, f1, fps, seed, cx, margin=0.92):
    """cameras._best without the overhead / both fallbacks."""
    energy = float(np.mean([F[i].get("energy", 0.3) for i in range(f0, f1)]))
    pts = [C._track_pts(F[i], side) for i in range(f0, f1)]
    best = None
    for nm, swing in [(name, 0.0), (name, 18.0), ("steady_low", 0.0), ("steady_low", 20.0)]:
        for dscale in (1.0, 1.12, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3):
            tr = C._shot_track(nm, F, f0, f1, side, fps, seed, dscale, energy, cx, swing)
            worst = max(C._in_frame(tr[0][i], tr[1][i], tr[2][i], tr[3][i], pts[i]) for i in range(0, f1 - f0, 2))
            if worst <= margin:
                break
        clear = C._arm_clear(tr[0], F, f0, f1)
        best = (nm, tr, worst, clear)
        if clear >= C.ARM_CLEAR:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit"); ap.add_argument("out")
    ap.add_argument("--swap", default="overhead,both,descend,rise")
    ap.add_argument("--safe", default="steady_low,across,orbit")
    a = ap.parse_args()
    d = json.load(open(a.edit)); F = d["frames"]; fps = d["fps"]
    swap, safe = a.swap.split(","), a.safe.split(",")
    S = shots_of(F)
    new = [kind(s[2]) for s in S]
    for k, (f0, f1, shot) in enumerate(S):
        nm, sd = new[k]
        if nm not in swap:
            continue
        busy = "R" if C._onsets(F, f0, f1, "R") >= C._onsets(F, f0, f1, "L") else "L"
        side = sd or busy
        nbrs = {new[k - 1] if k else None, new[k + 1] if k + 1 < len(S) else None}
        cands = [(t, s) for s in (side, "L" if side == "R" else "R") for t in safe]
        nm2, sd2 = next(c for c in cands if c not in nbrs)
        got, tr, worst, clear = track(nm2, sd2, F, f0, f1, fps, 7000 + k, d["centre_x"])
        pos, look, up, lens, fst, focus = tr
        for i in range(f1 - f0):
            F[f0 + i]["cam"] = {"pos": [round(float(v), 5) for v in pos[i]], "look": [round(float(v), 5) for v in look[i]],
                                "up": [round(float(v), 5) for v in up[i]], "lens": float(lens[i]),
                                "shot": f"{got}_{sd2}", "hide_head": True, "hide_body": True, "hide_case": got == "across"}
        new[k] = (got, sd2)
        print(f"shot {k:2d} {f0:5d}-{f1:5d} {shot:>14} -> {got}_{sd2}  in-frame {worst:.2f}  arm clear {clear:.3f} m")
    d["recamera"] = {"source": a.edit, "swap": swap, "safe": safe}
    json.dump(d, open(a.out, "w"))
    print("->", a.out)


if __name__ == "__main__":
    main()

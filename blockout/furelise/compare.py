"""Our rig vs a FürElise performance of the same MIDI, note by note and in distribution.

    python -m blockout.furelise.compare DATASET_DIR PIECE_ID REF.json ANIM.json [ANIM.json ...] [--json OUT]

Per note (an onset in their key sensors inside our animated window): their hand and finger (the fingertip on the
key as it goes down) vs ours (our fingering), and the strike depth along the key. Distributions: hand placement,
joint angles of pressing and free fingers, thumb, spread, anticipation, against REF (blockout.furelise.ref).
"""
import argparse, json

import numpy as np

from .. import human_score as hs
from . import ref as fr


def our_frames(paths):
    """Concatenate animated parts (each anim's frame times are relative to its own start)."""
    frames, fing = [], {"L": {}, "R": {}}
    for p in paths:
        a = json.load(open(p))
        for f in a["frames"]:
            f = dict(f); f["T"] = a["start"] + f["t"]; frames.append(f)
        for side, sl in a["fingering"].items():
            for s in sl:
                for t, pitch, f in s:
                    fing[side][(round(t, 3), pitch)] = f
    frames.sort(key=lambda f: f["T"])
    return frames, fing


def band(v):
    return hs._band(v) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("pid", type=int); ap.add_argument("ref"); ap.add_argument("anims", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    R = json.load(open(a.ref))
    frames, fing = our_frames(a.anims)
    T = np.array([f["T"] for f in frames])
    t0, t1 = T[0], T[-1]
    hands, pressed = fr.load_piece(a.ds, a.pid)
    ons = [(f, p) for f, p in fr.onsets(pressed) if t0 + 0.1 <= f / fr.FPS <= t1 - 0.3]
    # our fingering lookup by pitch and nearest onset time
    ours = {}
    for side, d in fing.items():
        for (t, p), f in d.items():
            ours.setdefault(p, []).append((t, side, f))
    agree_hand = agree_finger = n = 0
    conf = np.zeros((5, 5), int)
    depth = {"theirs": {}, "ours": {}}
    for f0, p in ons:
        w = fr.who_pressed(hands, f0, p)
        if w is None:
            continue
        side, fi, tip = w
        t = f0 / fr.FPS
        cand = [c for c in ours.get(p, []) if abs(c[0] - t) < 0.15]
        if not cand:
            continue
        ct, cside, cf = min(cand, key=lambda c: abs(c[0] - t))
        n += 1
        agree_hand += cside == side
        if cside == side:
            agree_finger += cf == fi
            conf[fi, cf] += 1
        key = f"f{fi + 1}.{'black' if fr.piano.is_black(p) else 'white'}"
        depth["theirs"].setdefault(key, []).append(float(tip[1]))
        j = int(np.argmin(np.abs(T - (ct + 0.04))))
        pad = frames[j]["hands"][cside]["pads"][cf]
        okey = f"f{cf + 1}.{'black' if fr.piano.is_black(p) else 'white'}"
        depth["ours"].setdefault(okey, []).append(float(pad[1]))
    out = {"notes": n, "hand_agree": round(agree_hand / max(n, 1), 3), "finger_agree": round(agree_finger / max(agree_hand, 1), 3),
           "confusion_theirs_rows_ours_cols": conf.tolist(),
           "strike_y_p50": {k: {"theirs": round(float(np.median(depth["theirs"].get(k, [np.nan]))), 3),
                                "ours": round(float(np.median(depth["ours"].get(k, [np.nan]))), 3)}
                            for k in sorted(set(depth["theirs"]) | set(depth["ours"]))}}
    # distributions from our frames
    place, rows, press = [], [], {"pressing": {k: [] for k in hs.PER_FINGER}, "free": {k: [] for k in hs.PER_FINGER}}
    for f in frames[::2]:
        held = {(v[0], int(v[1]) - 1) for v in (f.get("press") or {}).values()}
        for side, h in f["hands"].items():
            P = hs.anim_points(h)
            m = hs.measures(P, side)
            rows.append(m)
            busy = any((side, i) in held for i in range(5))
            place.append(fr.placement(P, side) | {"busy": busy})
            for i in range(4):
                for k in hs.PER_FINGER:
                    press["pressing" if (side, i + 1) in held else "free"][k].append(m[k][i])
    cmp = {}
    for k in ("wrist_z", "wrist_y", "knuckle_z", "knuckle_y", "pitch", "yaw", "roll"):
        cmp[f"place.{k}"] = (R["placement"][k]["playing"]["p50"], band([p[k] for p in place if p["busy"]])["p50"])
    for t in ("pressing", "free"):
        for k in ("mcp_flex", "pip_flex", "dip_flex", "tip_pitch"):
            cmp[f"{t}.{k}"] = (R["press_split"][t][k]["p50"], band(press[t][k])["p50"])
    for k in ("thumb_mcp_flex", "thumb_ip_flex", "thumb_abd", "thumb_tip_pitch", "abd_im", "abd_mr", "abd_rp", "span"):
        cmp[k] = (R["measures"][k]["pooled"]["p50"], band([m[k] for m in rows])["p50"])
    out["median_theirs_vs_ours"] = {k: [round(x, 3), round(y, 3)] for k, (x, y) in cmp.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

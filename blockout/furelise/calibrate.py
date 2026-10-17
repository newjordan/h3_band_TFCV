"""Calibrate the rig to FürElise (blockout/furelise/ref.py output) -> blockout/rig/furelise_cal.json + comfort.

    python -m blockout.furelise.calibrate REF.json

Writes, from the pianists' measured playing (held-out pieces excluded by the reference itself):
  * hand placement: knuckle-line height over the white key tops and its depth into the keys, palm pitch;
  * strike depth along the key per finger (white keys, white keys next to a sounding black key, black keys),
    as the hit-pad band p25-p75;
  * each finger's DIP:PIP ratio while pressing (tendon coupling);
  * the finger-pair span table the fingering uses (Parncutt-style MinPrac..MaxPrac in semitones), from the
    chords and successive notes the pianists actually fingered;
and refits the rig's relaxed playing pose (blockout/rig/comfort.json) to the reference's per-finger medians.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "rig", "furelise_cal.json")
FN = ("index", "middle", "ring", "pinky")


def main(ref_path):
    R = json.load(open(ref_path))
    pl = {k: v["playing"]["p50"] for k, v in R["placement"].items()}
    cal = {"source": R["source"], "ref": os.path.basename(ref_path), "pieces": len(R["pieces"]), "held_out": R["held_out"],
           "knuckle_z": pl["knuckle_z"], "knuckle_y": pl["knuckle_y"], "pitch_deg": pl["pitch"],
           "wrist_z": pl["wrist_z"], "wrist_y": pl["wrist_y"]}
    hp = {}
    for kind in ("white", "white_near_black", "black"):
        band = []
        for f in range(1, 6):
            s = R["strike"].get(f"f{f}.{kind}") or R["strike"].get(f"f{f}.white")
            band.append([s["tip_y"]["p25"], s["tip_y"]["p75"]])
        hp[kind] = band
    cal["hitpad"] = hp
    pr = R["per_finger"]
    cal["k_dip"] = [float(np.clip(pr[f"dip_flex.{n}"]["p50"] / max(pr[f"pip_flex.{n}"]["p50"], 1.0), 0.0, 1.0)) for n in FN]
    span = {}
    for k, d in R["span"].items():               # the minimums from successive notes (thumb-under and crossing
        sq, ch = d.get("seq"), d.get("chord")    # moves live there), the comfortable band and the stretch from
        if not sq:                               # chords, where the hand holds the shape
            continue
        if not ch or ch[-1] < 30:
            ch = sq
        span[k] = [int(np.floor(sq[0])), int(np.floor(sq[1])), int(round(ch[2])), int(round(ch[3])),
                   int(np.ceil(ch[4])), int(np.ceil(max(ch[5], ch[4] + 1)))]
    cal["span"] = span
    json.dump(cal, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print(json.dumps({k: v for k, v in cal.items() if k not in ("span",)}, indent=1))
    print("span", span)
    from ..rig import calibrate as rc
    out = {}
    for side in "RL":
        q, rep = rc.calibrate(side, ref_path)
        out[side] = q.tolist()
        print(side, "comfort, ours vs FürElise median:", rep)
    json.dump(out, open(rc.OUT, "w"))
    print("wrote", rc.OUT)


if __name__ == "__main__":
    main(sys.argv[1])

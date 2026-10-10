"""FürElise -> hand reference for the rig (same measures as blockout/human_score.py, from accurate 3D captures).

FürElise (Wang, Xu, Shi, Schumann, Liu; SIGGRAPH Asia 2024; https://for-elise.github.io/; data CC BY-NC 4.0):
15 pianists, 153 pieces, 21 hand joints per hand at 59.94 fps from multi-view capture, refined against the
Disklavier's own key sensors, with the MIDI and per-frame pressed keys.

Their frame: x toward the player (white key fronts at x = 0.1475), y up the keyboard, z up (white key tops ~0).
Ours: x across the keys (C4's centre = 0), y into the key from the white key fronts, z up from the white tops.

    python -m blockout.furelise.ref build DATASET_DIR OUT.json [--pieces 81,32,...] [--hold 81]
    python -m blockout.furelise.ref piece DATASET_DIR 81 OUT.npz          # one piece in our frame (for comparisons)
"""
import argparse, glob, json, math, os

import numpy as np

from .. import human_score as hs
from .. import piano
from .load import load_pickle, motion

FPS = 60000 / 1001
MESHES = os.path.expanduser("~/data/for_elise_dl/piano_meshes")
KEY_FRONT_X = 0.1475
TIPS = (4, 8, 12, 16, 20)
SAMPLE = int(os.environ.get("FE_SAMPLE", "4"))   # every 4th frame (15 fps) for pose statistics


def key_centres():
    """Their y of each key's centre line (index 0 = A0 = MIDI 21), from the visualizer's key meshes."""
    ys = []
    for i in range(88):
        V = np.array([[float(x) for x in l.split()[1:4]] for l in open(f"{MESHES}/{i}.obj") if l.startswith("v ")])
        ys.append(-0.5 * (V[:, 2].min() + V[:, 2].max()))     # mesh z -> world -y (rotation.x = +90 deg)
    return np.array(ys)


_KY = None


def to_ours(J):
    """(..., 3) their coordinates -> ours."""
    global _KY
    if _KY is None:
        _KY = key_centres()
    J = np.asarray(J, float)
    out = np.empty_like(J)
    out[..., 0] = J[..., 1] - _KY[60 - 21]
    out[..., 1] = KEY_FRONT_X - J[..., 0]
    out[..., 2] = J[..., 2] + 0.0008
    return out


def load_piece(ds, pid):
    d = os.path.join(ds, f"{pid:03d}")
    m = motion(d)
    hands = {"L": to_ours(m["left"]["joints"]), "R": to_ours(m["right"]["joints"])}
    pressed = load_pickle(os.path.join(d, "vis", "pressed_keys.pkl")) > 0
    return hands, pressed


def onsets(pressed):
    """[(frame, pitch)] where a key goes down."""
    p = np.asarray(pressed, bool)
    on = p & ~np.vstack([np.zeros((1, p.shape[1]), bool), p[:-1]])
    f, k = np.nonzero(on)
    return sorted(zip(f.tolist(), (k + 21).tolist()))


def who_pressed(hands, frame, pitch, look=3):
    """The fingertip that is on the key as it goes down: (side, finger 0-4, tip point) or None."""
    kx = piano.key_x(pitch)
    best = None
    for j in range(frame, min(frame + look, len(hands["R"]))):
        for side, J in hands.items():
            for f, t in enumerate(TIPS):
                p = J[j, t]
                d = abs(p[0] - kx) + 2.0 * max(0.0, p[2] - (piano.BLACK_H if piano.is_black(pitch) else 0.0) - 0.012)
                if best is None or d < best[0]:
                    best = (d, side, f, p.copy(), j)
    if best is None or best[0] > 0.02:
        return None
    return best[1], best[2], best[3]


def placement(P, side):
    """Hand placement in our frame: wrist and knuckle-line height and depth, palm pitch (+ = wrist above the
    knuckles), yaw (+ = fingers turned toward +x) and roll (+ = thumb side down), degrees."""
    kn = P[[5, 9, 13, 17]].mean(0)
    fwd = kn - P[0]
    h = math.hypot(fwd[0], fwd[1])
    ac = P[17] - P[5]
    sgn = 1.0 if side == "R" else -1.0
    return {"wrist_z": P[0, 2], "wrist_y": P[0, 1], "knuckle_z": kn[2], "knuckle_y": kn[1],
            "pitch": math.degrees(math.atan2(P[0, 2] - kn[2], h)),
            "yaw": math.degrees(math.atan2(fwd[0], fwd[1])),
            "roll": math.degrees(math.atan2(sgn * (P[17, 2] - P[5, 2]), abs(ac[0]) + 1e-9))}


def piece_rows(ds, pid):
    hands, pressed = load_piece(ds, pid)
    N = len(hands["R"])
    ons = onsets(pressed)
    # which fingers hold a key, per frame (from the onsets' owners, held while the key stays down)
    holding = {s: np.zeros((N, 5), bool) for s in "LR"}
    strikes = []
    for f0, p in ons:
        w = who_pressed(hands, f0, p)
        if w is None:
            continue
        side, fi, tip = w
        f1 = f0
        while f1 < N and pressed[f1, p - 21]:
            f1 += 1
        holding[side][f0:f1, fi] = True
        nb = piano.is_black(p) or any(piano.is_black(q) and pressed[min(f0 + 2, N - 1), q - 21]
                                      for q in (p - 1, p + 1) if 21 <= q <= 108)
        strikes.append({"pid": pid, "frame": f0, "pitch": p, "side": side, "finger": fi,
                        "black": piano.is_black(p), "near_black": bool(nb), "tip_y": float(tip[1]),
                        "tip_z": float(tip[2]), "dx": float(tip[0] - piano.key_x(p))})
    rows, place = [], []
    for side, J in hands.items():
        for j in range(0, N, SAMPLE):
            P = J[j]
            if not np.isfinite(P).all():
                continue
            m = hs.measures(P, side)
            rows.append((m, (pid, j, side, holding[side][j].copy())))
            place.append(placement(P, side) | {"busy": bool(holding[side][j].any())})
    return rows, place, strikes, hands, holding, ons


def anticipation(hands, holding, side, min_gap=0.4):
    """For each gap of at least min_gap s between one onset of a hand and its next: the share of the wrist's
    across-key move done by the middle of the gap (1 = already there, 0 = waited, < 0 = went the other way)."""
    J = hands[side]
    busy = holding[side].any(1)
    starts = np.nonzero(busy & ~np.concatenate([[False], busy[:-1]]))[0]
    out = []
    for a, b in zip(starts, starts[1:]):
        if (b - a) / FPS < min_gap:
            continue
        a2 = a + int(0.05 * FPS)
        d = J[b, 0, 0] - J[a2, 0, 0]
        if abs(d) < 0.01:
            continue
        out.append(float((J[(a2 + b) // 2, 0, 0] - J[a2, 0, 0]) / d))
    return out


def _piece_job(args):
    ds, pid = args
    r, pl, st, hands, holding, _ = piece_rows(ds, pid)
    an = anticipation(hands, holding, "L") + anticipation(hands, holding, "R")
    return pid, r, pl, st, an


def build(ds, out_json, pieces=None, hold=()):
    if pieces is None:
        pieces = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(ds, "[0-9][0-9][0-9]")))
    pieces = [p for p in pieces if p not in hold]
    rows, place, strikes, antic = [], [], [], []
    press = {"pressing": {k: [] for k in hs.PER_FINGER}, "free": {k: [] for k in hs.PER_FINGER}}
    from multiprocessing import Pool
    with Pool(int(os.environ.get("FE_JOBS", "12"))) as pool:
        for pid, r, pl, st, an in pool.imap_unordered(_piece_job, [(ds, p) for p in pieces]):
            for m, (_, _, side, hold_f) in r:
                for i in range(4):
                    for k in hs.PER_FINGER:
                        press["pressing" if hold_f[i + 1] else "free"][k].append(m[k][i])
            rows += r; place += pl; strikes += st; antic += an
            print(f"piece {pid}: {len(r)} hand-frames, {len(st)} strikes", flush=True)
    pooled = hs.collect(rows)
    ref = {"source": "FürElise (Wang et al. 2024, CC BY-NC 4.0)", "pieces": pieces, "held_out": list(hold),
           "measures": {}, "per_finger": {}, "splits": {}, "n_hand_frames": len(rows)}
    for k, (unit, gated, desc) in hs.MEASURES.items():
        ref["measures"][k] = {"unit": unit, "gated": gated, "desc": desc, "pooled": hs._band(pooled[k])}
        if k in hs.PER_FINGER:
            for f in hs.FINGERS:
                ref["per_finger"][f"{k}.{f}"] = hs._band(pooled[f"{k}.{f}"])
            for sp in ("lowest", "others"):
                ref["splits"][f"{k}.{sp}"] = hs._band(pooled[f"{k}.{sp}"])
    ref["press_split"] = {t: {k: hs._band(v) for k, v in d.items() if v} for t, d in press.items()}
    ref["placement"] = {k: {"all": hs._band([p[k] for p in place]),
                            "playing": hs._band([p[k] for p in place if p["busy"]])}
                        for k in ("wrist_z", "wrist_y", "knuckle_z", "knuckle_y", "pitch", "yaw", "roll")}
    ref["strike"] = {}
    for f in range(5):
        for blk in (False, True):
            s = [x for x in strikes if x["finger"] == f and x["black"] == blk]
            if len(s) > 20:
                ref["strike"][f"f{f + 1}.{'black' if blk else 'white'}"] = {
                    "tip_y": hs._band([x["tip_y"] for x in s]), "dx": hs._band([x["dx"] for x in s]),
                    "share": round(len(s) / max(1, len(strikes)), 4)}
    for f in range(5):                                 # white keys played next to a sounding black key
        sw = [x for x in strikes if x["finger"] == f and not x["black"] and x["near_black"]]
        if len(sw) > 20:
            ref["strike"][f"f{f + 1}.white_near_black"] = {"tip_y": hs._band([x["tip_y"] for x in sw]),
                                                          "share": round(len(sw) / max(1, len(strikes)), 4)}
    ref["fingering_share"] = {f"f{f + 1}": round(sum(x["finger"] == f for x in strikes) / max(1, len(strikes)), 4)
                              for f in range(5)}
    ref["anticipation"] = hs._band(antic) if antic else None
    # finger-pair spans the pianists used, in semitones, right-hand convention (finger i < j, pitch_j - pitch_i;
    # the left hand mirrored): chords (onsets within 50 ms) and successive notes (within 0.5 s)
    span = {}
    by = {}
    for x in strikes:
        by.setdefault((x["pid"], x["side"]), []).append(x)
    for (pid, side), ss in by.items():
        ss.sort(key=lambda x: x["frame"])
        sg = 1 if side == "R" else -1
        for a_, b_ in zip(ss, ss[1:]):
            if a_["finger"] == b_["finger"] or (b_["frame"] - a_["frame"]) / FPS > 0.5:
                continue
            i, j, d = a_["finger"], b_["finger"], sg * (b_["pitch"] - a_["pitch"])
            if i > j:
                i, j, d = j, i, -d
            kind = "chord" if (b_["frame"] - a_["frame"]) / FPS < 0.05 else "seq"
            span.setdefault(f"{i}{j}", {"chord": [], "seq": []})[kind].append(d)
    ref["span"] = {k: {kind: [float(np.percentile(v, q)) for q in (1, 5, 25, 75, 95, 99)] + [len(v)]
                       for kind, v in d.items() if len(v) >= 10} for k, d in span.items()}
    json.dump(ref, open(out_json, "w"), indent=1)
    print(f"wrote {out_json}: {len(rows)} hand-frames, {len(strikes)} strikes from {len(pieces)} pieces")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build"); b.add_argument("ds"); b.add_argument("out")
    b.add_argument("--pieces"); b.add_argument("--hold", default="")
    p = sub.add_parser("piece"); p.add_argument("ds"); p.add_argument("pid", type=int); p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.ds, a.out, [int(x) for x in a.pieces.split(",")] if a.pieces else None,
              [int(x) for x in a.hold.split(",") if x])
    elif a.cmd == "piece":
        r, pl, st, hands, holding, ons = piece_rows(a.ds, a.pid)
        np.savez_compressed(a.out, L=hands["L"], R=hands["R"], holdL=holding["L"], holdR=holding["R"],
                            strikes=json.dumps(st))
        print(len(st), "strikes")


if __name__ == "__main__":
    main()

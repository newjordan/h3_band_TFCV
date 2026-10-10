"""Where a pianist's wrist goes, learned from FürElise: placement at each strike and the travel between strikes.

    python -m blockout.furelise.wristmodel fit DATASET_DIR OUT.json [--hold 81,32]
    python -m blockout.furelise.wristmodel check DATASET_DIR MODEL.json PIECE_ID

Placement: at a strike group (one hand's onsets within 50 ms), the wrist's position in our frame relative to the
struck keys: for each finger f, the median of (wrist - key) over all its strikes, split white / black key; a group's
placement is the mean over its notes of key + offset_f. Travel: between two strike groups of a hand, the share of
the wrist's move (across and along the keys) done at normalised time tau (0 = previous strike, 1 = next strike),
median curves for small (< 4 cm) and large moves.
"""
import argparse, glob, json, os

import numpy as np

from .. import piano
from . import ref as fr

TAUS = np.linspace(0, 1, 21)


def groups(strikes, side):
    """Strike groups of one hand: lists of strikes whose onsets lie within 50 ms."""
    ss = sorted((s for s in strikes if s["side"] == side), key=lambda s: s["frame"])
    out = []
    for s in ss:
        if out and (s["frame"] - out[-1][0]["frame"]) / fr.FPS < 0.05:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def piece_samples(ds, pid):
    r, pl, strikes, hands, holding, ons = fr.piece_rows(ds, pid)
    off = {}
    travel = {"small": [], "large": []}
    for side in "LR":
        J = hands[side]
        G = groups(strikes, side)
        for g in G:
            j = g[0]["frame"]
            w = J[j, 0]
            for s in g:
                k = (s["finger"], "black" if s["black"] else "white", side)
                off.setdefault(k, []).append(w - np.array([piano.key_x(s["pitch"]), 0.0, 0.0]))
        for a, b in zip(G, G[1:]):
            ja, jb = a[0]["frame"], b[0]["frame"]
            if jb - ja < 4 or (jb - ja) / fr.FPS > 1.5:
                continue
            d = J[jb, 0, :2] - J[ja, 0, :2]
            n = float(np.linalg.norm(d))
            if n < 0.005:
                continue
            path = J[ja:jb + 1, 0, :2]
            share = (path - J[ja, 0, :2]) @ d / (n * n)
            tt = np.linspace(0, 1, len(path))
            travel["large" if n > 0.04 else "small"].append(np.interp(TAUS, tt, share))
    return off, travel


def fit(ds, out, hold=()):
    pieces = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(ds, "[0-9][0-9][0-9]")))
    pieces = [p for p in pieces if p not in hold]
    off, travel = {}, {"small": [], "large": []}
    from multiprocessing import Pool
    with Pool(12) as pool:
        for o, t in pool.imap_unordered(_job, [(ds, p) for p in pieces]):
            for k, v in o.items():
                off.setdefault(k, []).extend(v)
            for k in travel:
                travel[k] += t[k]
    model = {"source": "FürElise (Wang et al. 2024, CC BY-NC 4.0)", "pieces": len(pieces), "held_out": list(hold),
             "offset": {}, "travel": {}}
    for (f, kind, side), v in off.items():
        v = np.array(v)
        model["offset"][f"{side}{f}.{kind}"] = {"median": np.median(v, 0).round(4).tolist(), "n": len(v),
                                                 "iqr": (np.percentile(v, 75, 0) - np.percentile(v, 25, 0)).round(4).tolist()}
    for k, v in travel.items():
        v = np.array(v)
        model["travel"][k] = {"tau": TAUS.round(3).tolist(), "median": np.median(v, 0).round(4).tolist(), "n": len(v)}
    json.dump(model, open(out, "w"), indent=1)
    print("wrote", out, {k: v["n"] for k, v in model["offset"].items()}, {k: v["n"] for k, v in model["travel"].items()})


def _job(a):
    return piece_samples(*a)


def plan_x(model, G, side):
    """Wrist placement (x, y, z) per strike group from the model (G: groups as from groups())."""
    P = []
    for g in G:
        v = []
        for s in g:
            k = f"{side}{s['finger']}.{'black' if s['black'] else 'white'}"
            o = model["offset"].get(k) or model["offset"][f"{side}{s['finger']}.white"]
            v.append(np.array([piano.key_x(s["pitch"]), 0.0, 0.0]) + np.array(o["median"]))
        P.append(np.mean(v, 0))
    return np.array(P)


def check(ds, model_path, pid):
    """Feed the pianist's own fingering to the model; compare the planned wrist to the real one at the strikes
    and halfway between them."""
    model = json.load(open(model_path))
    r, pl, strikes, hands, holding, ons = fr.piece_rows(ds, pid)
    for side in "LR":
        G = groups(strikes, side)
        P = plan_x(model, G, side)
        real = np.array([hands[side][g[0]["frame"], 0] for g in G])
        e = np.linalg.norm(P - real, axis=1)
        ex = np.abs(P[:, 0] - real[:, 0])
        print(f"piece {pid} {side}: {len(G)} strike groups; wrist placement error median {np.median(e) * 1000:.0f} mm "
              f"(across {np.median(ex) * 1000:.0f} mm), p90 {np.percentile(e, 90) * 1000:.0f} mm")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit"); f.add_argument("ds"); f.add_argument("out"); f.add_argument("--hold", default="")
    c = sub.add_parser("check"); c.add_argument("ds"); c.add_argument("model"); c.add_argument("pid", type=int)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.ds, a.out, [int(x) for x in a.hold.split(",") if x])
    else:
        check(a.ds, a.model, a.pid)


if __name__ == "__main__":
    main()

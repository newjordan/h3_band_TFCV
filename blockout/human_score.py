"""Human-tolerance gate for the animated pianist's hands.

Scores a piano anim JSON against blockout/human_ref.json: distributions of hand-pose measures taken from real
pianists' hands in YouTube footage (MediaPipe Hands world landmarks). The same code computes every measure on
both sides, from 21 points in MediaPipe order:

    0 wrist | 1-4 thumb CMC, MCP, IP, tip | 5-8 index MCP, PIP, DIP, tip | 9-12 middle | 13-16 ring | 17-20 pinky

An anim hand maps onto that layout directly: `wrist`, then `fingers[f]` (thumb CMC,MCP,IP,tip; fingers
MCP,PIP,DIP,tip). Our tip is the end of the distal bone; MediaPipe's is the fingertip (skin end), a few mm
further. Only directions and length ratios are measured, so units and scale drop out.

    python -m blockout.human_score ANIM.json [--ref blockout/human_ref.json] [--json OUT.json]
    python -m blockout.human_score --build-ref EXTRACT_DIR SOURCES.json OUT_REF.json   # rebuild the reference

Gate per measure: OUT if our median is outside the human 25-75% band, or our 5th / 95th percentile is outside
the human 5-95% range. Overall PASS = no gated measure OUT.
"""
import argparse, glob, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "human_ref.json")
FINGERS = ("index", "middle", "ring", "pinky")
BASE = (5, 9, 13, 17)
PCTS = (5, 25, 50, 75, 95)
VIDEO_CAP = 300                         # pooled-band weight cap per video, in hand-frames
CURLED, STRAIGHT = 60.0, 20.0           # PIP+DIP flexion (deg) above which a finger is curled / below which straight

# name: (unit, gated, description)
MEASURES = {
    "mcp_flex": ("deg", True, "fingers 2-5 MCP flexion: proximal phalanx off the metacarpal (wrist->MCP), + toward the palm"),
    "pip_flex": ("deg", True, "fingers 2-5 PIP flexion (+ toward the palm), about the finger plane's normal"),
    "dip_flex": ("deg", True, "fingers 2-5 DIP flexion (+ toward the palm; negative = collapsed/hyperextended)"),
    "tip_pitch": ("deg", True, "fingers 2-5 distal phalanx (DIP->tip) angle to the palm plane, + = pointing palmward"),
    "thumb_mcp_flex": ("deg", True, "thumb MCP: metacarpal (CMC->MCP) to proximal phalanx, + toward palm/fingers"),
    "thumb_ip_flex": ("deg", True, "thumb IP: proximal to distal phalanx, + toward palm/fingers, - = tip bent back"),
    "thumb_tip_pitch": ("deg", True, "thumb distal phalanx (IP->tip) angle to the palm plane, + = palmward"),
    "thumb_abd": ("deg", True, "3D angle between the thumb metacarpal (CMC->MCP) and the index metacarpal (wrist->MCP)"),
    "abd_im": ("deg", True, "index-middle spread: angle between proximal phalanges in the palm plane, + = apart"),
    "abd_mr": ("deg", True, "middle-ring spread, as abd_im"),
    "abd_rp": ("deg", True, "ring-pinky spread, as abd_im"),
    "span": ("ratio", True, "thumb tip to pinky tip distance / hand length (wrist->middle MCP + middle phalanges)"),
    "n_curled": ("count", True, f"fingers 2-5 with PIP+DIP > {CURLED:.0f} deg, per hand-frame"),
    "n_straight": ("count", True, f"fingers 2-5 with PIP+DIP < {STRAIGHT:.0f} deg, per hand-frame"),
}
PER_FINGER = ("mcp_flex", "pip_flex", "dip_flex", "tip_pitch")


def _n(v):
    return v / np.linalg.norm(v)


def palm_frame(P, side):
    """Palm plane from the wrist and the four finger MCPs (as piano.hand_frame): rows across (index -> pinky),
    forward (wrist -> knuckle centre), up (back of the hand). side 'R'/'L' fixes the chirality."""
    m = P[list(BASE)]
    c = m.mean(0)
    fwd = _n(c - P[0])
    ac = m[3] - m[0]
    ac = _n(ac - fwd * (ac @ fwd))
    up = np.cross(ac, fwd) * (1 if side == "R" else -1)
    return c, ac, fwd, up


def _signed(u, v, ax):
    return math.degrees(math.atan2(float(np.cross(u, v) @ ax), float(u @ v)))


def finger_joint_angles(P, side):
    """Per finger 2-5: (mcp, pip, dip, tip_pitch) in degrees, and the hand's palm frame."""
    c, ac, fwd, up = palm_frame(P, side)
    out = []
    for b in BASE:
        mc = _n(P[b] - P[0])
        upm = _n(up - mc * (up @ mc))
        s = [_n(P[b + k + 1] - P[b + k]) for k in range(3)]
        mcp = math.degrees(math.atan2(float(-(s[0] @ upm)), float(np.linalg.norm(s[0] - upm * (s[0] @ upm)))))
        a = _n(np.cross(s[0], -up))          # finger plane normal: proximal phalanx x palm normal
        pitch = math.degrees(math.asin(float(np.clip(-(s[2] @ up), -1, 1))))
        out.append((mcp, _signed(s[0], s[1], a), _signed(s[1], s[2], a), pitch))
    return out, (c, ac, fwd, up)


def hand_len(P):
    return float(np.linalg.norm(P[9] - P[0]) + sum(np.linalg.norm(P[9 + k + 1] - P[9 + k]) for k in range(3)))


def measures(P, side):
    """All measures for one hand pose. P: (21,3) MediaPipe-ordered points. Returns {name: value or [4 values]}."""
    P = np.asarray(P, float)
    fa, (c, ac, fwd, up) = finger_joint_angles(P, side)
    r = {k: [f[i] for f in fa] for i, k in enumerate(PER_FINGER)}
    # thumb: flexion moves the distal segment toward the palm and across it
    d = _n(-up + 0.5 * ac)
    t = [_n(P[k + 1] - P[k]) for k in (1, 2, 3)]

    def tflex(u, v):
        ang = math.degrees(math.acos(float(np.clip(u @ v, -1, 1))))
        perp = v - u * (u @ v)
        return ang if perp @ d >= 0 else -ang
    r["thumb_mcp_flex"] = tflex(t[0], t[1])
    r["thumb_ip_flex"] = tflex(t[1], t[2])
    r["thumb_tip_pitch"] = math.degrees(math.asin(float(np.clip(-(t[2] @ up), -1, 1))))
    r["thumb_abd"] = math.degrees(math.acos(float(np.clip(t[0] @ _n(P[5] - P[0]), -1, 1))))
    th = []
    for b in BASE:
        e = P[b + 1] - P[b]
        e = e - up * (e @ up)
        th.append(math.degrees(math.atan2(float(e @ ac), float(e @ fwd))))
    r["abd_im"], r["abd_mr"], r["abd_rp"] = th[1] - th[0], th[2] - th[1], th[3] - th[2]
    L = hand_len(P)
    r["span"] = float(np.linalg.norm(P[4] - P[20]) / L)
    curl = [f[1] + f[2] for f in fa]
    r["n_curled"] = int(sum(x > CURLED for x in curl))
    r["n_straight"] = int(sum(x < STRAIGHT for x in curl))
    # contact proxy: the finger (2-5) whose tip reaches furthest palmward of the palm plane, per hand length
    depth = [float(-(P[b + 3] - c) @ up) / L for b in BASE]
    r["lowest"] = int(np.argmax(depth))
    return r


def anim_points(h):
    """Anim hand dict -> (21,3) in MediaPipe order."""
    pts = [h["wrist"]]
    for ch in h["fingers"]:
        pts.extend(ch)
    return np.asarray(pts, float)


# ---------------------------------------------------------------- statistics

def wpct(x, w=None, q=PCTS):
    x = np.asarray(x, float)
    if w is None:
        return [float(v) for v in np.percentile(x, q)]
    w = np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(p / 100.0, cw, x)) for p in q]


def collect(rows):
    """rows: list of (measures dict, extra) -> {measure: list of samples}, finger measures pooled over 2-5,
    plus per-finger and lowest/others splits."""
    out = {}
    for m, _ in rows:
        for k in MEASURES:
            v = m[k]
            if isinstance(v, list):
                out.setdefault(k, []).extend(v)
                for i, f in enumerate(FINGERS):
                    out.setdefault(f"{k}.{f}", []).append(v[i])
                    out.setdefault(f"{k}.{'lowest' if i == m['lowest'] else 'others'}", []).append(v[i])
            else:
                out.setdefault(k, []).append(v)
    return out


def _band(vals, w=None):
    p = wpct(vals, w)
    return {f"p{q}": round(v, 3) for q, v in zip(PCTS, p)} | {"n": len(vals)}


# ---------------------------------------------------------------- building the reference

def _keep(det, size_min=70.0, score_min=0.8):
    """Quality gate for one MediaPipe detection; returns (side, reason) with side None when dropped."""
    if det["score"] < score_min:
        return None, "low_handedness_score"
    x0, y0, x1, y1 = det["bbox"]
    if math.hypot(x1 - x0, y1 - y0) < size_min:
        return None, "small"
    P = np.asarray(det["world"], float)
    side = det["label"][0]
    fa, _ = finger_joint_angles(P, side)
    if sum(f[1] + f[2] for f in fa) < -20.0:     # PIP+DIP of all four fingers bent backwards: chirality/shape garbage
        return None, "inconsistent_chirality"
    return side, None


def build_ref(extract_dir, sources_json, out_json):
    src = json.load(open(sources_json))
    rows, per_video = [], {}
    for s in src["sources"]:
        d = json.load(open(os.path.join(extract_dir, s["id"] + ".json")))
        if s.get("exclude"):
            s.update(n_sampled_frames=d["n_sampled"], sample_fps=d["sample_fps"], n_detections=len(d["dets"]), n_kept=0)
            continue
        lo, hi = s.get("range", [0, 1e9])
        drop, vr = {}, []
        for det in d["dets"]:
            if not (lo <= det["t"] <= hi):
                drop["outside_range"] = drop.get("outside_range", 0) + 1
                continue
            side, why = _keep(det)
            if side is None:
                drop[why] = drop.get(why, 0) + 1
                continue
            P = np.asarray(det["world"], float)
            vr.append((measures(P, side), (s["id"], det["t"], side)))
        s.update(n_sampled_frames=d["n_sampled"], sample_fps=d["sample_fps"], n_detections=len(d["dets"]),
                 n_kept=len(vr), dropped=drop, frames_with_kept_hand=len({r[1][1] for r in vr}))
        per_video[s["id"]] = collect(vr)
        rows.extend(vr)
    pooled = collect(rows)
    # per-sample weights for the pooled bands: a video counts by min(its kept hand-frames, VIDEO_CAP), so one
    # long clip can't dominate and a clip with a few dozen detections can't count as much as a well-tracked one
    vid_of = [r[1][0] for r in rows]
    nv = {v: vid_of.count(v) for v in set(vid_of)}
    ref = {"measures": {}, "per_finger": {}, "splits": {}}
    for k, (unit, gated, desc) in MEASURES.items():
        w = _weights_for(rows, k, nv)
        ref["measures"][k] = {"unit": unit, "gated": gated, "desc": desc, "pooled": _band(pooled[k], w),
                              "pooled_unweighted": _band(pooled[k]),
                              "per_video": {v: _band(per_video[v][k]) for v in per_video if per_video[v].get(k)}}
        if k in PER_FINGER:
            for f in FINGERS:
                ref["per_finger"][f"{k}.{f}"] = _band(pooled[f"{k}.{f}"])
            for sp in ("lowest", "others"):
                ref["splits"][f"{k}.{sp}"] = _band(pooled[f"{k}.{sp}"])
    for k in ("n_curled", "n_straight"):
        vals = np.asarray(pooled[k])
        ref["measures"][k]["histogram"] = {str(i): round(float(np.mean(vals == i)), 3) for i in range(5)}
    ref["n_hand_frames"] = len(rows)
    out = {**{k: v for k, v in src.items() if k != "sources"}, "sources": src["sources"], **ref}
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"wrote {out_json}: {len(rows)} hand-frames from {len(per_video)} videos")


def _weights_for(rows, k, nv):
    w = []
    for m, (vid, _, _) in rows:
        reps = len(m[k]) if isinstance(m[k], list) else 1
        w.extend([min(nv[vid], VIDEO_CAP) / nv[vid]] * reps)
    return w


# ---------------------------------------------------------------- scoring an anim

def anim_rows(anim):
    d = json.load(open(anim))
    rows, press_split = [], {"pressing": {k: [] for k in PER_FINGER}, "free": {k: [] for k in PER_FINGER}}
    for fr in d["frames"]:
        pressed = {(v[0], int(v[1]) - 1) for v in (fr.get("press") or {}).values()} if isinstance(fr.get("press"), dict) else set()
        for side, h in fr["hands"].items():
            m = measures(anim_points(h), side)
            rows.append((m, (side, fr.get("t"))))
            for i in range(4):
                tag = "pressing" if (side, i + 1) in pressed else "free"
                for k in PER_FINGER:
                    press_split[tag][k].append(m[k][i])
    return rows, press_split


def score(anim, ref_path=REF, out_json=None):
    ref = json.load(open(ref_path))
    rows, press = anim_rows(anim)
    ours = collect(rows)
    res, fails = {}, []
    print(f"anim {anim}: {len(rows)} hand-frames  |  reference: {ref['n_hand_frames']} human hand-frames, "
          f"{len([s for s in ref['sources'] if not s.get('exclude')])} videos")
    hdr = f"{'measure':16s} {'ours p5/p25/p50/p75/p95':>38s}   {'human p5/p25/p50/p75/p95':>38s}  {'%out':>5s}  verdict"
    print(hdr); print("-" * len(hdr))
    for k, spec in ref["measures"].items():
        if k not in ours:
            continue
        h = spec["pooled"]
        o = wpct(ours[k])
        x = np.asarray(ours[k], float)
        frac = float(np.mean((x < h["p5"]) | (x > h["p95"])))
        why = []
        if not (h["p25"] <= o[2] <= h["p75"]):
            why.append(f"median {o[2]:.1f} outside human IQR [{h['p25']:.1f},{h['p75']:.1f}]")
        if o[0] < h["p5"]:
            why.append(f"p5 {o[0]:.1f} < human p5 {h['p5']:.1f}")
        if o[4] > h["p95"]:
            why.append(f"p95 {o[4]:.1f} > human p95 {h['p95']:.1f}")
        out = bool(why) and spec.get("gated", True)
        if out:
            fails.append(k)
        fmt = (lambda v: f"{v:7.1f}") if spec["unit"] != "ratio" else (lambda v: f"{v:7.2f}")
        print(f"{k:16s} {''.join(fmt(v) for v in o):>38s}   {''.join(fmt(h[f'p{q}']) for q in PCTS):>38s}  "
              f"{100 * frac:5.1f}  {'OUT' if out else 'ok'}{'  (' + '; '.join(why) + ')' if why else ''}")
        res[k] = {"ours": dict(zip([f"p{q}" for q in PCTS], o)), "human": h, "frac_outside_h5_95": frac,
                  "out": out, "why": why}
    print("\nper finger (info): ours p50 vs human p25-p50-p75")
    for k, h in ref.get("per_finger", {}).items():
        if k in ours:
            print(f"  {k:22s} ours {np.median(ours[k]):7.1f}   human {h['p25']:7.1f} {h['p50']:7.1f} {h['p75']:7.1f}")
    print("\nlowest-finger proxy split (info; the finger whose tip reaches furthest palmward):")
    for k, h in ref.get("splits", {}).items():
        if k in ours:
            print(f"  {k:22s} ours p50 {np.median(ours[k]):7.1f}   human {h['p25']:7.1f} {h['p50']:7.1f} {h['p75']:7.1f}")
    print("\nours, true pressing vs free fingers (info), p50:")
    for k in PER_FINGER:
        a, b = press["pressing"][k], press["free"][k]
        if a and b:
            print(f"  {k:10s} pressing {np.median(a):7.1f} (n={len(a)})   free {np.median(b):7.1f} (n={len(b)})")
    verdict = "PASS" if not fails else "FAIL"
    print(f"\nOVERALL: {verdict}  ({len(fails)} of {sum(1 for k in res if ref['measures'][k].get('gated', True))} "
          f"gated measures OUT{': ' + ', '.join(fails) if fails else ''})")
    if out_json:
        json.dump({"anim": anim, "verdict": verdict, "out": fails, "measures": res}, open(out_json, "w"), indent=1)
    return verdict, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anim", nargs="?")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--json")
    ap.add_argument("--build-ref", nargs=3, metavar=("EXTRACT_DIR", "SOURCES_JSON", "OUT_REF"))
    a = ap.parse_args()
    if a.build_ref:
        build_ref(*a.build_ref)
    elif a.anim:
        v, _ = score(a.anim, a.ref, a.json)
        sys.exit(0 if v == "PASS" else 1)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

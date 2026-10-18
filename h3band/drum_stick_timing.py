#!/usr/bin/env python3
"""Per-hit stick timing of H3 takes: is there a stick stroke at each drum when its hit sounds?

  drum_stick_timing.py ANIM.json GREY.mp4 TAKE.mp4... --cam EX,EY,EZ,LX,LY,LZ [--lens 32] [--box 72] [--json]

ANIM.json is the blockout (blockout/drums.py) the takes were made from, and GREY.mp4 its render from the same --cam
(blender_drums.py --cam). Each hand hit's strike point is projected through that camera to a box of --box px
(1280x736). In each video the stroke at that drum is the mean |frame diff| inside the box, frames hit - 6 .. hit + 6
(cut short at the clip's ends): the stick coming down into the box, landing and rebounding out of it.
  lag    the shift (-4..4 frames) at which the take's stroke best matches the grey's (0 = on the hit, + = late)
  r      how well it matches there
  amp    the take's stroke swing over the grey's (std of the energy in the window): near 0 nothing moves at that drum
A hit reads when its stroke matches (r >= 0.5) within a frame of the hit (|lag| <= 1) with some swing (amp >= 0.3).
Per take: the share of hits on the frame, within 1 frame (42 ms), and that read, with the per-hit lags.
The grey against itself is the ceiling.
"""
import argparse, json, subprocess

import numpy as np

W, H, FPS = 1280, 736, 24
HALF, MAXLAG = 6, 4


def frames(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def projector(cam, lens):
    """blender_drums.py's camera: eye looking at look, -Z forward with Y up toward world Z, 36 mm sensor fit to the width"""
    eye, look = np.array(cam[:3]), np.array(cam[3:6])
    f = (look - eye) / np.linalg.norm(look - eye)
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    k = lens / 36.0 * W

    def px(p):
        d = np.asarray(p) - eye
        z = d @ f
        return W / 2 + (d @ right) / z * k, H / 2 - (d @ up) / z * k
    return px


def energy(v, x, y, box):
    """e[i] = mean |frame i - frame i-1| in the box round (x, y); e[0] is nan"""
    x0, y0 = int(np.clip(x - box / 2, 0, W - box)), int(np.clip(y - box / 2, 0, H - box))
    z = v[:, y0:y0 + box, x0:x0 + box]
    return np.concatenate([[np.nan], np.abs(z[1:] - z[:-1]).mean(axis=(1, 2))])


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def score(grey, take, hits, px, box):
    n = min(len(grey), len(take))
    rows = []
    for h in hits:
        k = round(h["tf"] * FPS)
        x, y = px(h["s"])
        ge, te = energy(grey[:n], x, y, box), energy(take[:n], x, y, box)
        a0, a1 = max(k - HALF, 1 + MAXLAG), min(k + HALF, n - 1 - MAXLAG)
        g = ge[a0:a1 + 1]
        r, lag = max((corr(g, te[a0 + L:a1 + 1 + L]), L) for L in range(-MAXLAG, MAXLAG + 1))
        amp = float(te[a0 + lag:a1 + 1 + lag].std() / max(g.std(), 1e-3))
        rows.append(dict(piece=h["piece"], hand=h["hand"], frame=k, lag=lag, r=round(r, 2), amp=round(amp, 2),
                         reads=bool(r >= 0.5 and abs(lag) <= 1 and amp >= 0.3)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anim"); ap.add_argument("grey"); ap.add_argument("takes", nargs="+")
    ap.add_argument("--cam", required=True); ap.add_argument("--lens", type=float, default=32.0)
    ap.add_argument("--box", type=int, default=72)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    hits = json.load(open(a.anim))["hits"]
    px = projector([float(v) for v in a.cam.split(",")], a.lens)
    grey = frames(a.grey)
    out = []
    for p in [a.grey] + a.takes:
        rows = score(grey, frames(p), hits, px, a.box)
        lags = np.array([r["lag"] for r in rows])
        out.append(dict(take=p.replace("\\", "/").split("/")[-1], on=round(float(np.mean(lags == 0)), 3),
                        within1=round(float(np.mean(abs(lags) <= 1)), 3), reads=round(float(np.mean([r["reads"] for r in rows])), 3),
                        median_r=round(float(np.median([r["r"] for r in rows])), 2), hits=rows))
    if a.json:
        print(json.dumps(out))
        return
    print(f"{len(hits)} hand hits | first row: the grey against itself")
    for o in out:
        print(f"{o['take']:22s} on the frame {o['on']:5.0%}  within 1 {o['within1']:5.0%}  read {o['reads']:5.0%}  "
              f"median r {o['median_r']:+.2f}")
        print("   lags:", " ".join(f"{r['lag']:+d}" for r in o["hits"]))
        print("   r:   ", " ".join(f"{r['r']:+.1f}" for r in o["hits"]))
        print("   amp: ", " ".join(f"{r['amp']:.1f}" for r in o["hits"]))


if __name__ == "__main__":
    main()

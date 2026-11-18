"""Missed key presses in a solved piano anim.

Usage: python h3_tools/miss.py DIR TAG [TAG ...]

DIR holds TAG.json (the anim) and TAG_notes.json.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blockout.piano import key_x, is_black


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: miss.py DIR TAG [TAG ...]")
    src = sys.argv[1]
    for tag in sys.argv[2:]:
        a = json.load(open(os.path.join(src, f"{tag}.json")))
        fr = a["frames"]; fps = a["fps"]; st = a["start"]
        notes = json.load(open(os.path.join(src, f"{tag}_notes.json")))
        fing = a["fingering"]
        fmap = {}
        for h, sl in fing.items():
            for s in sl:
                for t, p, f in s:
                    fmap[(round(t, 3), p)] = (h, f)
        print("##", tag)
        for s, e, p, v, tr in notes:
            t = s - st
            if t < 0 or t > len(fr) / fps - 0.3:
                continue
            j0, j1 = int(t * fps), min(len(fr) - 1, int((min(e, s + 0.25) - st) * fps) + 1)
            dd = [fr[j]["keys"].get(str(p), 0.0) for j in range(j0, j1 + 1)]
            d = max(dd)
            if d < 0.5:
                h, f = fmap.get((round(s, 3), p), ("?", 0))
                jj = j0 + int(np.argmin([abs(fr[j]["hands"][h]["pads"][f][2]) for j in range(j0, j1 + 1)]))
                pad = np.array(fr[jj]["hands"][h]["pads"][f])
                print(f" miss t={t:.2f} p={p} {'blk' if is_black(p) else 'wht'} {h}{f+1} depth {d:.2f} pad-key dx {1000*(pad[0]-key_x(p)):.0f} mm y {1000*pad[1]:.0f} z {1000*pad[2]:.0f}  wrist {np.round(np.array(fr[jj]['hands'][h]['wrist'])*1000).astype(int)}")


if __name__ == "__main__":
    main()

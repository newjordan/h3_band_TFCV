"""densify_take.py EDIT.json OUT.json F0 N [K] : the take's frames F0..F0+N-1 with K-1 in-between poses inserted after
each frame (K=2: twice the frame density), so H3 sees the same motion sampled K times as densely; the original frames
are kept exactly (every K-th output frame is an original). Joint angles, positions, key depths and the body are
interpolated linearly; root rotations (and the head's) by slerp. OUT has fps * K and only those frames (indices 0..)."""
import copy, json, sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def lerp(a, b, u):
    if isinstance(a, dict):
        return {k: lerp(a[k], b[k], u) if k in b else a[k] for k in a}
    if isinstance(a, list) and a and isinstance(a[0], list) and len(a) == 3 and len(a[0]) == 3:   # a 3x3 rotation
        R = Rotation.from_matrix(np.array([a, b], float))
        return Slerp([0, 1], R)([u]).as_matrix()[0].round(6).tolist()
    if isinstance(a, list):
        return [lerp(x, y, u) for x, y in zip(a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return round(a + (b - a) * u, 6)
    return a


def keys_lerp(a, b, u):
    ks = set(a) | set(b)
    return {k: round(a.get(k, 0.0) + (b.get(k, 0.0) - a.get(k, 0.0)) * u, 4) for k in ks}


def main():
    src, dst, f0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    d = json.load(open(src)); F = d["frames"]
    out = []
    for j in range(f0, f0 + n):
        a, b = F[j], F[min(j + 1, len(F) - 1)]
        for s in range(K):
            if s == 0:
                out.append(copy.deepcopy(a)); continue
            u = s / K
            fr = copy.deepcopy(a)
            fr["t"] = round(a["t"] + (b["t"] - a["t"]) * u, 5)
            fr["hands"] = lerp(a["hands"], b["hands"], u)
            fr["body"] = lerp(a["body"], b["body"], u)
            fr["keys"] = keys_lerp(a["keys"], b["keys"], u)
            if "keys_plan" in a:
                fr["keys_plan"] = keys_lerp(a["keys_plan"], b.get("keys_plan", {}), u)
            ca, cb = a.get("cam"), b.get("cam")
            if ca and cb and ca.get("shot") == cb.get("shot"):     # a moving camera glides; never across a cut
                fr["cam"] = dict(ca, **{k: lerp(ca[k], cb[k], u) for k in ("pos", "look", "up", "lens") if k in ca and k in cb})
            out.append(fr)
    d2 = {k: v for k, v in d.items() if k != "frames"}
    d2["frames"] = out; d2["fps"] = d["fps"] * K; d2["start"] = d["start"] + F[f0]["t"]
    d2["densify"] = {"source": src, "f0": f0, "n": n, "k": K}
    json.dump(d2, open(dst, "w"))
    print(len(out), "frames at", d2["fps"], "fps ->", dst)


if __name__ == "__main__":
    main()

"""Zig-zag (frame-to-frame shake): a joint/pad whose velocity flips sign on consecutive frames with both steps big.
Split by whether the finger is pressing (a key it holds is down) or free."""
import json, sys, numpy as np
a = json.load(open(sys.argv[1])); fr = a["frames"]; fps = a["fps"]
for h in "LR":
    F = [f["hands"][h] for f in fr]
    pads = np.array([x["pads"] for x in F])            # N,5,3
    names = sorted({(b, k) for b, d in F[0]["angles"].items() for k in d})
    Q = np.degrees(np.array([[x["angles"][b][k] for b, k in names] for x in F]))
    press = np.array([[any(v == f"{h}{fi+1}" for v in f_["press"].values()) for fi in range(5)] for f_ in fr])
    v = np.diff(pads, axis=0)                           # N-1,5,3
    zz = (np.einsum("nfk,nfk->nf", v[1:], v[:-1]) < 0) & (np.linalg.norm(v[1:], axis=2) > 0.0015) & (np.linalg.norm(v[:-1], axis=2) > 0.0015)
    pm = press[1:-1]
    print(f"{h} pad zig-zags (>1.5mm steps reversing) per finger-second: pressing {zz[pm].sum()/max(pm.sum(),1)*fps:.2f}  free {zz[~pm].sum()/max((~pm).sum(),1)*fps:.2f}  per finger:", np.round(zz.sum(0)/len(zz)*fps, 2))
    vq = np.diff(Q, axis=0)
    zq = (vq[1:] * vq[:-1] < 0) & (np.abs(vq[1:]) > 2) & (np.abs(vq[:-1]) > 2)
    worst = np.argsort(-zq.sum(0))[:5]
    print(f"   joint zig-zags (>2deg steps reversing) per joint-second {zq.sum()/zq.size*fps:.3f}; worst:", ", ".join(f"{names[i][0]}.{names[i][1]} {zq[:,i].sum()/len(zq)*fps:.2f}/s" for i in worst))

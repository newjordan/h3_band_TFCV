"""Jitter / smoothness / wrist metrics for a piano anim JSON."""
import json, sys, math
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

A = json.load(open(sys.argv[1])); fr = A["frames"]; fps = A.get("fps", 24)
a0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; a1 = int(sys.argv[3]) if len(sys.argv) > 3 else len(fr)
fr = fr[a0:a1]
for h in ("L", "R"):
    F = [f["hands"].get(h) for f in fr]
    if any(x is None for x in F): continue
    P = np.array([x["root"]["pos"] for x in F])
    Rm = np.array([x["root"]["rot"] for x in F])
    E = np.array([x["elbow"] for x in F])
    names = sorted({(b, k) for b, d in F[0]["angles"].items() for k in d})
    Q = np.array([[x["angles"][b][k] for b, k in names] for x in F])
    pads = np.array([x["pads"] for x in F])   # N,5,3
    def hf(X, w=9):
        return X - savgol_filter(X, w, 2, axis=0)
    rv = Rot.from_matrix(Rm)
    rel = (rv[:-1].inv() * rv[1:]).magnitude()   # frame-to-frame rotation, rad
    angacc = np.abs(np.diff(rel))
    acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=1) * fps * fps
    # tremor: velocity sign flips per second, per axis, ignoring tiny moves
    v = np.diff(P, axis=0)
    flips = sum(((v[1:, k] * v[:-1, k]) < 0) & (np.abs(v[1:, k]) > 2e-4) & (np.abs(v[:-1, k]) > 2e-4) for k in range(3)).sum()
    qhf = hf(Q)
    # wrist angle vs forearm: hand forward axis = knuckle centre - root
    kn = np.array([[x["fingers"][f][1] for f in range(1, 5)] for x in F]).mean(1)
    fwd = kn - P; fwd /= np.linalg.norm(fwd, axis=1)[:, None]
    fa = P - E; fa /= np.linalg.norm(fa, axis=1)[:, None]
    # flexion: vertical angle diff; deviation: horizontal angle diff
    pitch = lambda d: np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
    yaw = lambda d: np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    flex = pitch(fa) - pitch(fwd); dev = yaw(fwd) - yaw(fa)
    print(f"== {h}  frames {len(F)}")
    if "plan" in F[0]["root"]:
        PL = np.array([x["root"]["plan"] for x in F])
        print(f" PLAN HF jitter rms {np.sqrt((hf(PL)**2).sum(1).mean())*1000:.2f} mm;  root-plan offset median {np.median(np.linalg.norm(P-PL,axis=1))*1000:.1f} p95 {np.percentile(np.linalg.norm(P-PL,axis=1),95)*1000:.1f} mm")
    print(f" root HF jitter rms  {np.sqrt((hf(P)**2).sum(1).mean())*1000:.2f} mm   p99 {np.percentile(np.linalg.norm(hf(P),axis=1),99)*1000:.2f} mm")
    print(f" root accel  median {np.median(acc):.2f}  p95 {np.percentile(acc,95):.2f}  max {acc.max():.1f} m/s^2")
    print(f" root rot step deg  median {np.degrees(np.median(rel)):.2f}  p95 {np.degrees(np.percentile(rel,95)):.2f}; rot accel p95 {np.degrees(np.percentile(angacc,95)):.2f} deg/f^2")
    print(f" root velocity reversals {flips/ (len(F)/fps):.1f} /s")
    print(f" joint HF jitter rms {np.degrees(np.sqrt((qhf**2).mean())):.2f} deg   worst joints:",
          ", ".join(f"{names[i][0][-9:]}.{names[i][1]} {np.degrees(np.sqrt((qhf[:,i]**2).mean())):.2f}" for i in np.argsort(-(qhf**2).mean(0))[:4]))
    print(f" pad HF jitter rms {np.sqrt((hf(pads.reshape(len(F),-1)).reshape(len(F),5,3)**2).sum(2).mean())*1000:.2f} mm")
    print(f" wrist flex (+=hand below forearm line) median {np.median(flex):.1f} p5 {np.percentile(flex,5):.1f} p95 {np.percentile(flex,95):.1f}; |dev| median {np.median(np.abs(dev)):.1f} p95 {np.percentile(np.abs(dev),95):.1f} deg")
    print(f" wrist angle HF jitter flex {np.sqrt((hf(flex[:,None])**2).mean()):.2f} dev {np.sqrt((hf(dev[:,None])**2).mean()):.2f} deg")

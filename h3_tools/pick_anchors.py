"""Pick lockstep anchors from a pass-1 take: one frame per slot (every K frames), the best within +-R frames by
plate structure correlation x sharpness (a smeared frame loses Laplacian energy). Writes PNGs to Comfy input and
prints the guides JSON.  pick_anchors.py PLATE.mp4 TAKE.mp4 OUTNAME [K=12 R=3]"""
import sys, os, json, subprocess, numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from fidelity import frames, edges
plate, take, name = sys.argv[1:4]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 12; R = int(sys.argv[5]) if len(sys.argv) > 5 else 3
P, T = frames(plate), frames(take)
n = min(len(P), len(T))
r = np.array([np.corrcoef(edges(P[k]).ravel(), edges(T[k]).ravel())[0, 1] for k in range(n)])
sharp = np.array([ndimage.laplace(ndimage.gaussian_filter(T[k], 0.7)).var() for k in range(n)])
score = r / r.max() + 0.5 * sharp / np.median(sharp)
out = []
for c in range(0, n, K):
    lo, hi = max(0, c - R), min(n, c + R + 1)
    if c == 0: lo, hi = 0, 1
    if c + K >= n: lo, hi = max(0, n - 1 - R), n             # last slot: near the end
    k = lo + int(np.argmax(score[lo:hi]))
    if out and k == out[-1]["frame_idx"]: continue
    fn = f"band/suite/{name}_a{k:03d}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", take, "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
                    os.path.expanduser("~/comfyui-h3-audio/input/" + fn)], check=True)
    out.append({"image": fn, "frame_idx": k, "r": round(float(r[k]), 3), "sharp": round(float(sharp[k] / np.median(sharp)), 2)})
print(json.dumps(out))

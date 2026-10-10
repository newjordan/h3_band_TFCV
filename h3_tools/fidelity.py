"""Plate fidelity of an H3 take: per frame, structure correlation between the plate and the take (Sobel edges of
the luminance, blurred, Pearson r) and the global shift (phase correlation, px at 320 wide). A cut or reframe
shows as r falling toward 0 and/or a big shift.
    fidelity.py PLATE.mp4 TAKE.mp4 [label]"""
import sys, subprocess, numpy as np
from scipy import ndimage

def frames(path, w=320):
    h = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height", "-of", "csv=p=0", path]).decode().split(",")[1]) * w // \
        int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width", "-of", "csv=p=0", path]).decode().strip())
    h -= h % 2
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray",
                                   "-f", "rawvideo", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)

def edges(f):
    g = ndimage.gaussian_filter(f, 1.0)
    e = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return ndimage.gaussian_filter(e, 2.0)

def shift(a, b):
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b)); F /= np.abs(F) + 1e-9
    c = np.fft.ifft2(F).real; i, j = np.unravel_index(np.argmax(c), c.shape)
    i = i - c.shape[0] if i > c.shape[0] // 2 else i; j = j - c.shape[1] if j > c.shape[1] // 2 else j
    return float(np.hypot(i, j))

def main():
  P, T = frames(sys.argv[1]), frames(sys.argv[2])
  n = min(len(P), len(T))
  r = []; sh = []
  for k in range(n):
      a, b = edges(P[k]), edges(T[k])
      r.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1])); sh.append(shift(a, b))
  r = np.array(r); sh = np.array(sh)
  lab = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2].split("/")[-1]
  q = lambda a, b: f"{np.mean(r[a:b]):.2f}"
  print(f"{lab:28s} r mean {r.mean():.2f}  f0-24 {q(0,24)}  f24-48 {q(24,48)}  f48-{n} {q(48,n)}  min {r.min():.2f}@{int(r.argmin())}  "
        f"shift>8px frames {int((sh > 8).sum())}")
  print("   r per 6 frames:", " ".join(f"{np.mean(r[i:i+6]):.2f}" for i in range(0, n, 6)))


if __name__ == "__main__":
  main()

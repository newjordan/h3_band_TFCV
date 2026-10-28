"""Does a render move like its Blender plate where the hands are? Inside the plate's hand mask (Blender --pass mask):
dense optical flow (Farneback) of the plate vs the render, frame to frame, and edge-structure correlation.

    python h3_tools/motionmatch.py PLATE.mp4 RENDER.mp4 MASK.mp4 [label]

flow_err: median |flow_render - flow_plate| / (|flow_plate| + 0.5 px) over moving hand pixels (0 = moves exactly
like the plate, 1 = as wrong as the motion itself); flow_cos: median cosine between the two flows where the plate
moves > 1 px; hand_r: Pearson r of Sobel edges inside the (dilated) mask, per frame.
"""
import sys, subprocess, numpy as np, cv2


def frames(path, w=640):
    W, H = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(","))
    h = H * w // W; h -= h % 2
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def main():
    P, R, M = frames(sys.argv[1]), frames(sys.argv[2]), frames(sys.argv[3])
    n = min(len(P), len(R), len(M))
    errs, coss, rs = [], [], []
    k = np.ones((7, 7), np.uint8)
    for i in range(1, n):
        m = cv2.dilate((M[i] > 127).astype(np.uint8), k) > 0
        if m.sum() < 200:
            continue
        fp = cv2.calcOpticalFlowFarneback(P[i - 1], P[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fr = cv2.calcOpticalFlowFarneback(R[i - 1], R[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(fp, axis=2)
        mv = m & (mag > 1.0)
        if mv.sum() > 50:
            d = np.linalg.norm(fr - fp, axis=2)[mv] / (mag[mv] + 0.5)
            errs.append(float(np.median(d)))
            c = (fr[mv] * fp[mv]).sum(1) / (np.linalg.norm(fr[mv], axis=1) * mag[mv] + 1e-6)
            coss.append(float(np.median(c)))
        ep = cv2.Sobel(P[i], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(P[i], cv2.CV_32F, 0, 1) ** 2
        er = cv2.Sobel(R[i], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(R[i], cv2.CV_32F, 0, 1) ** 2
        a, b = np.sqrt(ep[m]), np.sqrt(er[m])
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    lab = sys.argv[4] if len(sys.argv) > 4 else sys.argv[2].split("/")[-1]
    print(f"{lab:28s} flow_err {np.median(errs):.2f}  flow_cos {np.median(coss):.2f}  hand_r {np.median(rs):.2f}  "
          f"(moving frames {len(errs)}, frames {len(rs)})")


if __name__ == "__main__":
    main()

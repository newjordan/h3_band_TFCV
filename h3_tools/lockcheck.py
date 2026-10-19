"""Does an H3 render sit on our reference? Per frame, inside the Blender hand mask: how far H3's edges are from the
reference geometry's edges (the Canny of Blender's normal pass, the same edges the ControlNet was given).

    python h3_tools/lockcheck.py RENDER.mp4 REF_NORMAL.mp4 MASK.mp4 OUT_PREFIX [--w 1280] [--tol 3] [--every K]

ref->h3 (recall): for each reference edge pixel, the distance to the nearest H3 edge pixel: is every bone of our hand
drawn where it is? h3->ref (precision): for each H3 edge pixel in the hand region, the distance to the nearest
reference edge: did H3 draw anything where our hand isn't? Both in px at the render's size, per frame: median, and the
share within the tolerance (--tol px, default 3; at a closer camera use the scale-matched one, e.g. 3 x 1.45 = 4.35).
--every K scores only every K-th render frame (a K-times dense render against the reference at the take's own rate).
Writes OUT_PREFIX.csv (per frame) and OUT_PREFIX_overlay.mp4 (the render with the reference edges
in red and the mask outline in yellow), and prints the summary.
"""
import sys, subprocess, numpy as np, cv2


def frames(path, w, h, gray=False):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}" + (",format=gray" if gray else ""),
                                   "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"])
    c = 1 if gray else 3
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, c) if c == 3 else np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def edges(im):
    g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 20, 60) > 0


def main():
    render, ref, mask, out = sys.argv[1:5]
    w = int(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 1280
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 3.0
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    h = (w * 704 // 1280) // 2 * 2
    R, N, M = frames(render, w, h)[::every], frames(ref, w, h), frames(mask, w, h, gray=True)
    n = min(len(R), len(N), len(M))
    rows = []
    vw = cv2.VideoWriter(out + "_overlay_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    k = np.ones((15, 15), np.uint8)
    for i in range(n):
        m = cv2.dilate((M[i] > 127).astype(np.uint8), k) > 0               # the hand region, a little wider
        er, en = edges(R[i]) & m, edges(N[i]) & m
        if en.sum() < 20:
            rows.append((i, np.nan, np.nan, np.nan, np.nan)); vw.write(R[i]); continue
        d_to_h3 = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
        d_to_ref = cv2.distanceTransform((~en).astype(np.uint8), cv2.DIST_L2, 3)
        rec = d_to_h3[en]; pre = d_to_ref[er] if er.sum() else np.array([np.inf])
        rows.append((i, float(np.median(rec)), float(np.mean(rec <= tol)), float(np.median(pre)), float(np.mean(pre <= tol))))
        ov = R[i].copy()
        ov[en] = (0, 0, 255)
        cnt, _ = cv2.findContours((M[i] > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnt, -1, (0, 220, 255), 1)
        cv2.putText(ov, f"f{i}  ref->h3 med {rows[-1][1]:.1f}px  within {tol:g}px {100 * rows[-1][2]:.0f}%", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(ov)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out + "_overlay_raw.mp4", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", out + "_overlay.mp4"], check=True)
    subprocess.run(["rm", "-f", out + "_overlay_raw.mp4"])
    a = np.array([r[1:] for r in rows], float)
    ok = ~np.isnan(a[:, 0])
    with open(out + ".csv", "w") as fh:
        fh.write("frame,ref_to_h3_med_px,ref_within_tol,h3_to_ref_med_px,h3_within_tol\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    q = lambda c: np.round(np.nanpercentile(a[ok, c], [10, 50, 90]), 2)
    print(f"{render.split('/')[-1]}: frames {n}  ref->h3 med px p10/50/90 {q(0)}  ref within {tol:g} px p10/50/90 {q(1)}  "
          f"h3->ref med px p10/50/90 {q(2)}  h3 within {tol:g} px p10/50/90 {q(3)}  worst {np.nanmin(a[ok, 1]):.2f}")


if __name__ == "__main__":
    main()

"""pose_control.py POSE2D.json NORMAL_DIR MASK_DIR OUT.mp4 [N] [--hands outline|edges] [--depth] : the standard-hand
overlay control.

--depth: perspective in the joint lines (nearer bones and joints thicker, near bones drawn over far ones): seen along
the fingers (from behind the hand or from above) the flat joint skeleton doesn't say which way the fingers point.
--keys DIR: the key edges from DIR, a normal pass rendered with the rig hidden (blender_piano.py --hide hands): no hand,
arm or sleeve edges at all and no blanked hole, so the keys under the fingers stay in the control. With the hand-mask
blanking the rig's shirt sleeves stayed in, and seen from across the keys H3 painted a person behind them.
--base DIR: the joints drawn over DIR's frames (a depth pass) instead of the key edges: one control carrying depth
and joints (as a second ControlNet patch at 0.5, depth made H3 paint no hands at all).
--width: the lines as wide as half a 12 mm bone through the frame's lens (POSE2D frames carry "lens"): close up, the
fixed 3 px lines got painted as thin wires.

--hands outline: also the stock hand's outer outline (where the hand is and how the fingers curl, nothing inside it);
--hands edges: keep all of the stock hand's normal-pass edges (outline + knuckles) under the joints. Seen from above,
curled fingers point at the camera and their joints collapse into stubs, so the joints alone read as an open hand.

Per frame: the rig's hands as an OpenPose hand skeleton (21 joints from the posed stock MPFB2 armature, projected through
the shot camera by blender_piano.py --pose2d; rainbow finger colours, red joints) plus the forearm (elbow -> wrist), over
the Canny edges (20/60) of the stock-hand normal pass with the hand region blanked (keys, case: where the keys are and
how far down they go). The control says where every joint and key is and nothing about what the hand looks like, so
the look comes from the prompt alone. Writes OUT.mp4 and OUT.png (frame 0 preview)."""
import colorsys, json, subprocess, sys
import cv2
import numpy as np

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
ARM = {"R": (0, 255, 255), "L": (0, 255, 0)}          # BGR: OpenPose body colours of the right / left lower arm


def draw(img, hand, elbow, side, scale, f=None):
    """f(depth) (--depth / --width): a thickness factor per bone and joint from its depth, and the hand's bones drawn
    far to near, so the near ones cross over the far ones."""
    p = [(int(round(x)), int(round(y))) for x, y, _ in hand]
    z = [q[2] for q in hand]
    zref, f = f is not None, f or (lambda d: 1.0)
    e = (int(round(elbow[0])), int(round(elbow[1])))
    cv2.line(img, e, p[0], ARM[side], max(4, round(7 * scale * f(z[0]))), cv2.LINE_AA)
    order = sorted(range(len(EDGES)), key=lambda k: -(z[EDGES[k][0]] + z[EDGES[k][1]])) if zref else range(len(EDGES))
    for k in order:
        a, b = EDGES[k]
        rgb = colorsys.hsv_to_rgb(k / len(EDGES), 1.0, 1.0)
        th = max(2, round(3 * scale * f((z[a] + z[b]) / 2)))
        cv2.line(img, p[a], p[b], tuple(int(255 * c) for c in rgb[::-1]), th, cv2.LINE_AA)
    for i in (sorted(range(len(p)), key=lambda i: -z[i]) if zref else range(len(p))):
        g = f(z[i])                               # joints grow slower than the bones: knuckle knobs, not blobs
        cv2.circle(img, p[i], max(3, round(4 * scale * (g if g <= 1 else 1 + 0.4 * (g - 1)))), (0, 0, 255), -1,
                   cv2.LINE_AA)


def main():
    hands_mode = sys.argv[sys.argv.index("--hands") + 1] if "--hands" in sys.argv else "none"
    depth, width = "--depth" in sys.argv, "--width" in sys.argv
    keys = sys.argv[sys.argv.index("--keys") + 1] if "--keys" in sys.argv else None
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
    args = [a for i, a in enumerate(sys.argv) if a not in ("--hands", "--depth", "--width", "--keys", "--base")
            and (i == 0 or sys.argv[i - 1] not in ("--hands", "--keys", "--base"))]
    pose, ndir, mdir, out = args[1:5]
    P = json.load(open(pose))
    W, H = P["size"]
    frames = sorted(P["frames"], key=int)
    if len(args) > 5:
        frames = frames[:int(args[5])]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", out],
                            stdin=subprocess.PIPE)
    k = np.ones((25, 25), np.uint8)
    for j, f in enumerate(frames):
        nrm = cv2.resize(cv2.imread(f"{keys or ndir}/f_{int(f):05d}.png"), (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(cv2.imread(f"{mdir}/f_{int(f):05d}.png", 0), (W, H), interpolation=cv2.INTER_AREA) > 127
        ed = cv2.Canny(cv2.GaussianBlur(nrm, (3, 3), 0), 20, 60) > 0
        if hands_mode != "edges" and not keys:
            ed &= ~(cv2.dilate(msk.astype(np.uint8), k) > 0)        # no hand outline or skin detail in the control
        img = np.zeros((H, W, 3), np.uint8)
        img[ed] = 255
        if base:                                  # a depth pass under the joints instead of the key edges
            img = cv2.resize(cv2.imread(f"{base}/f_{int(f):05d}.png"), (W, H), interpolation=cv2.INTER_AREA)
        if hands_mode == "outline":                                  # the hand's outer contour only
            cnt, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, (255, 255, 255), max(1, round(W / 1280)), cv2.LINE_AA)
        hands = P["frames"][f]
        # far hand first (larger camera depth), so the near hand's lines sit on top
        hands = {s: v for s, v in hands.items() if s in ("L", "R")}
        fz = None
        if depth and hands:                       # relative: zref / depth, zref = the frame's median joint depth
            zref = float(np.median([q[2] for s in hands for q in hands[s]["hand"]]))
            fz = lambda d, zref=zref: float(np.clip(zref / max(d, 1e-3), 0.5, 2.5))
        if width and "lens" in P["frames"][f]:    # absolute: half a 12 mm bone's width through the lens (sensor 36 mm)
            fpx = P["frames"][f]["lens"] / 36.0 * W
            fz = lambda d, fpx=fpx: float(np.clip(fpx * 0.006 / max(d, 1e-3) / (3 * W / 1280), 0.67, 8.0))
        for side in sorted(hands, key=lambda s: -np.mean([q[2] for q in hands[s]["hand"]])):
            draw(img, hands[side]["hand"], hands[side]["elbow"], side, W / 1280, fz)
        if j == 0:
            cv2.imwrite(out.rsplit(".", 1)[0] + ".png", img)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()
    print(len(frames), "frames ->", out)


if __name__ == "__main__":
    main()

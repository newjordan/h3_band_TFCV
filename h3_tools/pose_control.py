"""pose_control.py POSE2D.json NORMAL_DIR MASK_DIR OUT.mp4 [N] : the standard-hand overlay control for H3 Fun ControlNet.

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


def draw(img, hand, elbow, side, scale):
    th, r = max(2, round(3 * scale)), max(3, round(4 * scale))
    p = [(int(round(x)), int(round(y))) for x, y, _ in hand]
    e = (int(round(elbow[0])), int(round(elbow[1])))
    cv2.line(img, e, p[0], ARM[side], max(4, round(7 * scale)), cv2.LINE_AA)
    for k, (a, b) in enumerate(EDGES):
        rgb = colorsys.hsv_to_rgb(k / len(EDGES), 1.0, 1.0)
        cv2.line(img, p[a], p[b], tuple(int(255 * c) for c in rgb[::-1]), th, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, q, r, (0, 0, 255), -1, cv2.LINE_AA)


def main():
    pose, ndir, mdir, out = sys.argv[1:5]
    P = json.load(open(pose))
    W, H = P["size"]
    frames = sorted(P["frames"], key=int)
    if len(sys.argv) > 5:
        frames = frames[:int(sys.argv[5])]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", out],
                            stdin=subprocess.PIPE)
    k = np.ones((25, 25), np.uint8)
    for j, f in enumerate(frames):
        nrm = cv2.resize(cv2.imread(f"{ndir}/f_{int(f):05d}.png"), (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(cv2.imread(f"{mdir}/f_{int(f):05d}.png", 0), (W, H), interpolation=cv2.INTER_AREA) > 127
        ed = cv2.Canny(cv2.GaussianBlur(nrm, (3, 3), 0), 20, 60) > 0
        ed &= ~(cv2.dilate(msk.astype(np.uint8), k) > 0)            # no hand outline or skin detail in the control
        img = np.zeros((H, W, 3), np.uint8)
        img[ed] = 255
        hands = P["frames"][f]
        # far hand first (larger camera depth), so the near hand's lines sit on top
        for side in sorted(hands, key=lambda s: -np.mean([q[2] for q in hands[s]["hand"]])):
            draw(img, hands[side]["hand"], hands[side]["elbow"], side, W / 1280)
        if j == 0:
            cv2.imwrite(out.rsplit(".", 1)[0] + ".png", img)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()
    print(len(frames), "frames ->", out)


if __name__ == "__main__":
    main()

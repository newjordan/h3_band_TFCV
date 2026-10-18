#!/usr/bin/env python3
"""A zone video for H3 Band Zone Latent: where the sticks and hands are, frame by frame, seen from the blockout's camera.

  drum_zone_mask.py ANIM.json OUT.mp4 --cam EX,EY,EZ,LX,LY,LZ [--lens 32] [--fps 24] [--stick 1.0] [--hand 0.7] [--fore 0.4]

ANIM.json is the blockout (blockout/drums.py) and --cam its camera (blender_drums.py --cam). Each frame draws, white
on black at 1280x736: each stick (butt to tip) at --stick, a disc round each grip at --hand and each forearm (elbow to
wrist) at --fore. The node runs a zone pixel of value v at video_denoise + (zone_denoise - video_denoise) * v, so the
sticks can start from far less noise than the rest of the frame and the hands holding them stay with them.
OUT.mp4 is written at --fps (24 for a half-speed blockout played as slow motion), one frame per blockout frame.
"""
import argparse, json, os, subprocess, sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drum_stick_timing import W, H, projector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anim"); ap.add_argument("out")
    ap.add_argument("--cam", required=True); ap.add_argument("--lens", type=float, default=32.0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--stick", type=float, default=1.0); ap.add_argument("--hand", type=float, default=0.7)
    ap.add_argument("--fore", type=float, default=0.4)
    a = ap.parse_args()
    px = projector([float(v) for v in a.cam.split(",")], a.lens)
    frames = json.load(open(a.anim))["frames"]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
                            "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", a.out],
                           stdin=subprocess.PIPE)
    for fr in frames:
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        for level, draw in sorted(((a.fore, "fore"), (a.hand, "hand"), (a.stick, "stick"))):
            v = round(255 * level)
            for hd in fr["hands"].values():
                if draw == "fore":
                    d.line([px(hd["elbow"]), px(hd["wrist"])], fill=v, width=44)
                elif draw == "hand":
                    x, y = px(hd["grip"])
                    d.ellipse([x - 48, y - 48, x + 48, y + 48], fill=v)
                else:
                    d.line([px(hd["butt"]), px(hd["tip"])], fill=v, width=24)
                    x, y = px(hd["tip"])
                    d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=v)
        enc.stdin.write(im.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"{len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()

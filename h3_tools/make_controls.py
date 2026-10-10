"""Control videos for H3 Fun ControlNet Union from a blockout plate's frames.
    make_controls.py PLATE_DIR OUT_PREFIX [W H]  -> OUT_PREFIX_canny.mp4, OUT_PREFIX_gray.mp4"""
import sys, glob, subprocess, cv2, numpy as np
d, out = sys.argv[1], sys.argv[2]
W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1280, 704)
files = sorted(glob.glob(d + "/*.png"))
def write(name, frames):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24",
                          "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", name], stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
gray, canny = [], []
for f in files:
    g = cv2.cvtColor(cv2.resize(cv2.imread(f), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray.append(g)
    canny.append(cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 110))
write(out + "_canny.mp4", canny); write(out + "_gray.mp4", gray)
print(len(files), "frames")

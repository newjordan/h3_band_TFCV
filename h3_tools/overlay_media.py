"""overlay_media.py OUT_DIR : the Band Aid page's look media from the overlay runs (skeleton_piano.py ov_s1_<look>), every
H3 input from the stock MPFB2 hand. Writes into OUT_DIR:
  looks_cycle.mp4 (+ _poster.jpg)   the five looks one after another on the same 1.3 s, labelled
  looks_grid.mp4 (+ _still.jpg)     the five looks and our stock-hand blockout in a 3x2 grid
  look_<look>_trio.mp4              our stock hand | the control (rig joints + keys) | the H3 render
Every frame labelled H3 is H3's; the blockout and the control are Blender's (and pose_control.py's)."""
import pathlib, subprocess, sys

HOME = pathlib.Path.home()
ROOT = HOME / "h3/results/skeleton_piano"
COMFY_IN = HOME / "comfyui-h3-audio/input/band/skeleton"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LOOKS = [("skeleton", "ov_s1_skeleton", "skeleton"), ("monster", "ov_s1_monster", "monster"),
         ("robot", "ov_s1_robot", "robot"), ("paw", "ov_s1_paw2", "kitty-cat paw"), ("real", "ov_s1_real", "real hands")]
K, N = 4, 31                                   # the probe: 124 dense frames at 4x = 31 take frames


def label(t, w, h, size=20):
    return (f"scale={w}:{h},drawbox=x=0:y=0:w=iw:h={size + 14}:color=black@0.55:t=fill,"
            f"drawtext=fontfile={FONT}:text='{t}':x=10:y=7:fontsize={size}:fontcolor=white")


def ff(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, check=True)


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    finals = {k: ROOT / r / f"{r}_final.mp4" for k, r, _ in LOOKS}
    dec = f"select='not(mod(n\\,{K}))',setpts=N/24/TB"
    # cycle: the looks one after another
    ins, fc = [], ""
    for i, (k, r, name) in enumerate(LOOKS):
        ins += ["-i", str(finals[k])]
        fc += f"[{i}:v]{label('H3 over the stock hand: ' + name, 1280, 704, 24)}[v{i}];"
    fc += "".join(f"[v{i}]" for i in range(len(LOOKS))) + f"concat=n={len(LOOKS)}:v=1:a=0[v]"
    ff(ins + ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
              str(out / "looks_cycle.mp4")])
    ff(["-ss", "0.6", "-i", str(out / "looks_cycle.mp4"), "-frames:v", "1", "-q:v", "3", str(out / "looks_cycle_poster.jpg")])
    # grid: five looks + the stock-hand blockout (Blender), every 4th dense plate frame = the take's frames
    plate = COMFY_IN / "std_s1_probe_c0_plate.mp4"
    ins = ["-i", str(plate)] + sum((["-i", str(finals[k])] for k, _, _ in LOOKS), [])
    fc = f"[0:v]{dec},{label('our rig: the stock MPFB2 hand (Blender)', 640, 352, 16)}[p];"
    for i, (k, r, name) in enumerate(LOOKS, start=1):
        fc += f"[{i}:v]{label('H3: ' + name, 640, 352, 16)}[g{i}];"
    fc += "[g1][g2][g3]hstack=3[top];[g4][g5][p]hstack=3[bot];[top][bot]vstack[v]"
    ff(ins + ["-filter_complex", fc, "-map", "[v]", "-frames:v", str(N), "-c:v", "libx264", "-crf", "18",
              "-pix_fmt", "yuv420p", str(out / "looks_grid.mp4")])
    ff(["-ss", "0.6", "-i", str(out / "looks_grid.mp4"), "-frames:v", "1", "-q:v", "3", str(out / "looks_grid_still.jpg")])
    # trio per look: stock hand | control | H3
    ctrl = COMFY_IN / "std_s1_probe_c0_posekeys.mp4"
    for k, r, name in LOOKS:
        ff(["-i", str(plate), "-i", str(ctrl), "-i", str(finals[k]), "-filter_complex",
            f"[0:v]{dec},{label('our rig: stock MPFB2 hand', 640, 352, 16)}[a];"
            f"[1:v]{dec},{label('control: rig joints + keys', 640, 352, 16)}[b];"
            f"[2:v]{label('H3: ' + name, 640, 352, 16)}[c];[a][b][c]hstack=3[v]",
            "-map", "[v]", "-frames:v", str(N), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out / f"look_{k}_trio.mp4")])
    print("overlay media ->", out)


if __name__ == "__main__":
    main()

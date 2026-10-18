"""Review videos for the H3 hand looks (ffmpeg only).

    hand_looks_video.py cycle SHOT OUT.mp4    one timeline that switches the hand look every 41 frames (1.7 s): the shot
                                              plays 5 times, looks in order skeleton > monster > robot > paw > real > ...,
                                              so every look covers every third of the motion once; cuts on the same frame
    hand_looks_video.py grid  SHOT OUT.mp4    3x2: the five H3 renders + the Blender grey plate (bare hand), labelled
    hand_looks_video.py trio  SHOT LOOK OUT   Blender grey plate (look geometry) | ControlNet input (Canny of the Blender
                                              normal pass) | H3 render
Labels name the tool that made each picture: H3 renders, Blender plates/controls.
"""
import subprocess, sys
from hand_looks import COMFY_IN, LOOKS, N, ROOT, SHOTS, clip, render_path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAMES = {"skeleton": "skeleton", "monster": "monster", "robot": "robot", "paw": "kitty-cat paw", "real": "real hands",
         "armour": "armour (K22)"}
SEG = (0, 41, 82, N)


def label(txt, size=30, y="20"):
    t = txt.replace(":", r"\:").replace("'", "")
    return (f"drawtext=fontfile={FONT}:text='{t}':x=20:y={y}:fontsize={size}:fontcolor=white:"
            f"box=1:boxcolor=black@0.6:boxborderw=8")


def run(cmd):
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + cmd, check=True)


def cycle(shot, out):
    ins, fl, segs = [], [], []
    for lk in LOOKS:
        ins += ["-i", str(render_path(shot, lk))]
    s = 0
    for k in range(5):                      # 5 passes of the shot x 3 segments = 15 segments, 1.7 s each
        for j in range(3):
            i = s % 5
            fl.append(f"[{i}:v]trim=start_frame={SEG[j]}:end_frame={SEG[j + 1]},setpts=PTS-STARTPTS,"
                      f"{label('H3 render  |  hand look: ' + NAMES[LOOKS[i]], 34)},"
                      f"{label('same Blender blockout motion, shot ' + shot + ', seed 7', 20, 'h-th-24')}[s{s}]")
            segs.append(f"[s{s}]")
            s += 1
    fl.append("".join(segs) + f"concat=n={len(segs)}:v=1:a=0[v]")
    fl.append("[0:a]asplit=5" + "".join(f"[a{k}]" for k in range(5)) + ";" + "".join(f"[a{k}]" for k in range(5))
              + "concat=n=5:v=0:a=1[a]")
    run(ins + ["-filter_complex", ";".join(fl), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out])


def grid(shot, out):
    ins = []
    for lk in LOOKS:
        ins += ["-i", str(render_path(shot, lk))]
    ins += ["-i", str(COMFY_IN / clip(shot, "real", "plate"))]
    fl = [f"[{i}:v]scale=640:352,{label('H3: ' + NAMES[lk], 22)}[c{i}]" for i, lk in enumerate(LOOKS)]
    fl.append(f"[5:v]scale=640:352,{label('Blender grey plate, bare hand: the motion all five follow', 18)}[c5]")
    fl.append("[c0][c1][c2]hstack=3[r0];[c3][c4][c5]hstack=3[r1];[r0][r1]vstack[v]")
    run(ins + ["-filter_complex", ";".join(fl), "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out])


def trio(shot, look, out):
    ins = ["-i", str(COMFY_IN / clip(shot, look, "plate")), "-i", str(COMFY_IN / clip(shot, look, "ncanny")),
           "-i", str(render_path(shot, look))]
    pl = "Blender grey plate: bare MPFB hand" if look == "real" else f"Blender grey plate + {NAMES[look]} geometry"
    fl = [f"[0:v]scale=640:352,{label(pl, 20)}[a]",
          f"[1:v]scale=640:352,{label('ControlNet input: Canny of Blender normal pass', 20)}[b]",
          f"[2:v]scale=640:352,{label('H3 render: ' + NAMES[look], 20)}[c]", "[a][b][c]hstack=3[v]"]
    run(ins + ["-filter_complex", ";".join(fl), "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out])


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "cycle":
        cycle(sys.argv[2], sys.argv[3])
    elif c == "grid":
        grid(sys.argv[2], sys.argv[3])
    elif c == "trio":
        trio(sys.argv[2], sys.argv[3], sys.argv[4])

"""Skeleton hands play the piano: H3 over a locked-off shot of a piano take, with the hand-looks recipe
(h3_tools/hand_looks.py), in chained clips so the look carries across clip boundaries.

    skeleton_piano.py passes RUN     Blender grey plate, normal pass and mask of every clip (skeleton geometry)
    skeleton_piano.py run    RUN     H3, clip after clip: clip k > 0 starts from clip k-1's last H3 frame
    skeleton_piano.py score  RUN     per clip: flow agreement with the normal pass inside the hand mask (motionmatch)
    skeleton_piano.py join   RUN     the clips back to back (+ the take's sound) -> ROOT/RUN/RUN.mp4, and a
                                     side-by-side with the Blender plate -> ROOT/RUN/RUN_sbs.mp4

RUN = a key of RUNS below: the edit (cameras.py --style still on the take), its first frame, the clip lengths
(17k+5 frames each), the view words for the prompt and the sound (the take's hands on the working piano).
The recipe: fl2va mode B, the grey plate WITH the look's geometry at video_strength 0.88, Fun ControlNet Union 2.0
(strength 1.0) on the Canny (20/60) of the normal pass of the same geometry; the look comes from that geometry and
the prompt. Every frame here is H3's; the plates and controls are Blender's.
"""
import json, os, pathlib, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()
ROOT = HOME / "h3/results/skeleton_piano"
COMFY_IN = HOME / "comfyui-h3-audio/input"
COMFY_OUT = HOME / "comfyui-h3-audio/output/ks_lipsync"
RUNNER = HOME / "h3audio/h3audio_run.py"
LEASE = HOME / "h3/gpu_lease.sh"
PY = str(HOME / "h3/mv_tools/.venv/bin/python")
LOOK = "skeleton"
SEED = 7
RUNS = {
    # Rachmaninoff, the opening (song 81.3 - 94.7 s), the overhead still shot of the C19 take
    "rach_open": {"edit": ROOT / "edit_c19_still.json", "f0": 993, "clips": [124, 124, 73],
                  "view": "Seen from directly above the keyboard",
                  "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
    # the first 62 frames of clip 0 at twice the frame density (h3_tools/densify_take.py: an in-between pose after each
    # frame, originals kept), the sound slowed to half speed: H3's 4-frame latent spans half the motion. Lock test.
    "rach_open_x2": {"edit": ROOT / "edit_c19_x2.json", "f0": 0, "clips": [124],
                     "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x2_c0.wav"},
    # the close camera at 4x the frame density: one H3 latent frame (4 video frames) per original frame (T5)
    "rach_open_close_x4": {"edit": ROOT / "edit_c19_close_x4.json", "f0": 0, "clips": [124],
                           "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_c0.wav"},
    # "super size": the close camera at 1920x1088 (2.25x the H3 tokens; the same 124 frames as T2) (T6)
    "rach_open_close_hd": {"edit": ROOT / "edit_c19_close.json", "f0": 993, "clips": [124], "size": (1920, 1088),
                           "view": "Seen from directly above the keyboard",
                           "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
    # both: the close camera at twice the frame density (T4)
    "rach_open_close_x2": {"edit": ROOT / "edit_c19_close_x2.json", "f0": 0, "clips": [124],
                           "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x2_c0.wav"},
    # the same first clip with a locked-off overhead camera fitted to the hands only (~1.45x closer): lock test
    "rach_open_close": {"edit": ROOT / "edit_c19_close.json", "f0": 993, "clips": [124],
                        "view": "Seen from directly above the keyboard",
                        "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
}


def clip_frames(run):
    R = RUNS[run]; out, f = [], R["f0"]
    for n in R["clips"]:
        out.append((f, n)); f += n
    return out


def render(run, kind, extra, procs=4):
    """Blender frames of the whole run (all clips) of one pass into ROOT/run/kind/."""
    R = RUNS[run]
    out = ROOT / run / kind
    out.mkdir(parents=True, exist_ok=True)
    a0, n = R["f0"], sum(R["clips"])
    if len(list(out.glob("f_*.png"))) >= n:
        return out
    env = dict(os.environ, RIG_SLEEVE="0.06")
    step = -(-n // procs)
    ps = []
    for k in range(procs):
        a, b = a0 + k * step, min(a0 + (k + 1) * step, a0 + n)
        W, H = R.get("size", (1280, 704))
        cmd = ["blender", "-b", "--factory-startup", "-P", str(REPO / "blockout/blender_piano.py"), "--",
               str(R["edit"]), str(out), "--view", "edit", "--frames", f"{a}:{b}", "--hand-style", LOOK,
               "--res", f"{W}x{H}"] + extra
        ps.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / f"blender_{k}.log", "w"),
                                   stderr=subprocess.STDOUT))
    for p in ps:
        p.wait()
    got = len(list(out.glob("f_*.png")))
    assert got >= n, f"{out}: {got} frames"
    return out


def encode(src, dst, f0, n, gray=False, crf=14, size=(1280, 704)):
    vf = f"scale={size[0]}:{size[1]}" + (",format=gray" if gray else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-start_number", str(f0), "-i", f"{src}/f_%05d.png",
                    "-frames:v", str(n), "-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)],
                   check=True)


def canny(src, dst, f0, n, size=(1280, 704)):
    import cv2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size[0]}x{size[1]}", "-r", "24",
                          "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for f in range(f0, f0 + n):
        im = cv2.resize(cv2.imread(str(src / f"f_{f:05d}.png")), tuple(size), interpolation=cv2.INTER_AREA)
        p.stdin.write(cv2.Canny(cv2.GaussianBlur(im, (3, 3), 0), 20, 60).tobytes())
    p.stdin.close(); p.wait()


def cin(run, k, kind, ext="mp4"):
    return f"band/skeleton/{run}_c{k}_{kind}.{ext}"


def passes(run):
    R = RUNS[run]
    (COMFY_IN / "band/skeleton").mkdir(parents=True, exist_ok=True)
    plate = render(run, "plate", [])
    normal = render(run, "normal", ["--pass", "normal"])
    mask = render(run, "mask", ["--pass", "mask"])
    for k, (f0, n) in enumerate(clip_frames(run)):
        sz = R.get("size", (1280, 704))
        encode(plate, COMFY_IN / cin(run, k, "plate"), f0, n, size=sz)
        encode(normal, ROOT / run / f"c{k}_normal.mp4", f0, n, crf=12, size=sz)
        encode(mask, ROOT / run / f"c{k}_mask.mp4", f0, n, gray=True, crf=10, size=sz)
        canny(normal, COMFY_IN / cin(run, k, "ncanny"), f0, n, size=sz)
        t0 = f0 / 24.0                             # seconds into the take (model.wav starts at the take's frame 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-t", f"{n / 24.0:.4f}", "-i", str(R["sound"]),
                        "-ac", "2", "-ar", "48000", str(COMFY_IN / cin(run, k, "sound", "wav"))], check=True)
    print("passes", run, "ok")


def prompt(run):
    P = json.loads((REPO / "h3_tools/hand_look_prompts.json").read_text())
    return P["base"].format(**P["looks"][LOOK]).replace("Seen from high above and behind the player", RUNS[run]["view"])


def tag(run, k):
    return f"sk_{run}_c{k}"


def out_mp4(run, k):
    return COMFY_OUT / f"{tag(run, k)}_00001_.mp4"


def job(run, k, first=None):
    n = clip_frames(run)[k][1]
    W, H = RUNS[run].get("size", (1280, 704))
    j = {"steps": 20, "width": W, "height": H, "length": n, "scheduler": "beta", "sampler": "res_multistep",
         "seed": SEED, "audio": cin(run, k, "sound", "wav"), "prompt": prompt(run), "mode": "B",
         "video": cin(run, k, "plate"), "video_strength": 0.88,
         "control": {"video": cin(run, k, "ncanny"), "strength": 1.0}, "tag": tag(run, k)}
    if first:                                      # chained: this clip opens on the previous clip's last H3 frame
        j["image"] = first
    else:
        j["no_first"] = True
    (ROOT / run / "cfg").mkdir(parents=True, exist_ok=True)
    p = ROOT / run / "cfg" / f"{tag(run, k)}.json"
    p.write_text(json.dumps(j, indent=1))
    return p


def last_frame(mp4, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(mp4), "-frames:v", "1", "-update", "1",
                    str(dst)], check=True)


def run_all(run, need=int(os.environ.get("H3_NEED_GB", 30)), poll=180):
    for k in range(len(RUNS[run]["clips"])):
        if out_mp4(run, k).exists():
            print("exists:", out_mp4(run, k)); continue
        first = None
        if k:
            first = cin(run, k, "first", "png")
            last_frame(out_mp4(run, k - 1), COMFY_IN / first)
        p = job(run, k, first)
        while True:
            r = subprocess.run(["bash", str(LEASE), "take", "h3-skeleton", f"H3 skeleton piano {run} clip {k}", str(need)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            print(time.strftime("%H:%M:%S"), "waiting:", r.stdout.strip(), flush=True)
            time.sleep(poll)
        try:
            print(time.strftime("%H:%M:%S"), "H3", tag(run, k), flush=True)
            subprocess.run(["python3", str(RUNNER), str(p)], check=True)
        finally:
            subprocess.run(["bash", str(LEASE), "release", "h3-skeleton"], capture_output=True)
        score(run, k)


def score(run, k):
    ref, mask = ROOT / run / f"c{k}_normal.mp4", ROOT / run / f"c{k}_mask.mp4"
    plate = COMFY_IN / cin(run, k, "plate")
    mm = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(ref), str(out_mp4(run, k)), str(mask),
                                  f"{run} c{k} vs normal"]).decode().strip()
    mp = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(plate), str(out_mp4(run, k)), str(mask),
                                  f"{run} c{k} vs plate"]).decode().strip()
    with open(ROOT / run / "scores.txt", "a") as fh:
        fh.write(mm + "\n" + mp + "\n")
    print(mm); print(mp)


def join(run):
    R = RUNS[run]; d = ROOT / run
    n = len(R["clips"])
    lst = d / "clips.txt"
    lst.write_text("".join(f"file '{out_mp4(run, k)}'\n" for k in range(n)))
    t0, dur = R["f0"] / 24.0, sum(R["clips"]) / 24.0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{t0:.4f}",
                    "-t", f"{dur:.4f}", "-i", str(R["sound"]), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(d / f"{run}.mp4")], check=True)
    plates = d / "plates.txt"
    plates.write_text("".join(f"file '{COMFY_IN / cin(run, k, 'plate')}'\n" for k in range(n)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(plates), "-i", str(d / f"{run}.mp4"),
                    "-filter_complex", "[0:v]scale=640:352[a];[1:v]scale=640:352[b];[a][b]hstack", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                    str(d / f"{run}_sbs.mp4")], check=True)
    print("joined", d / f"{run}.mp4")


if __name__ == "__main__":
    cmd, run = sys.argv[1:3]
    if cmd in ("passes", "all"):
        passes(run)
    if cmd in ("run", "all"):
        run_all(run)
    if cmd == "score":
        for k in range(len(RUNS[run]["clips"])):
            score(run, k)
    if cmd in ("join", "all"):
        join(run)

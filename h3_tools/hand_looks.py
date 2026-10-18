"""H3 hand looks: the leg-K recipe (K22/K23) with one of five hand looks on a blockout shot.

    hand_looks.py passes SHOT LOOK    Blender: grey plate, normal pass and mask, all with the look's geometry
    hand_looks.py job    SHOT LOOK    write the H3 job JSON (cfg/<tag>.json)
    hand_looks.py run    SHOT LOOK    wait for memory and the GPU lease, then run the job (h3audio_run.py)
    hand_looks.py score  SHOT LOOK    flow agreement (motionmatch.py) inside the look's mask vs its normal pass and vs its
                                      grey plate; whole-frame edge r vs its plate (fidelity.py) -> ROOT/scores.csv
    hand_looks.py all    SHOT LOOK    passes + job + run + score

SHOT: kh111 | ov125 (v17 still edit, 124 frames each). LOOK: skeleton | monster | robot | paw | real | armour.

The recipe (what makes H3 follow the Blender hands):
  - fl2va (MiniMaxH3ImageToVideo), mode B, no first frame pinned (no_first)
  - the video stream = a GREY Workbench plate of the hand WITH the look's geometry, video_strength 0.88
  - Fun ControlNet Union 2.0 (MiniMaxH3FunControlNetApply, strength 1.0) on the Canny (20/60) of Blender's
    NORMAL pass (MATCAP) of the same hand + geometry; clips 17k+5 frames (124 = ok: 124 = 17*7+5)
  - the prompt names the look; the scene words stay fixed
  The look comes from the control's geometry + the prompt. Not from plate pixels (chrome plates drift), and not from
  ref2va (ref2va ignores the Fun ControlNet).
Geometry: blockout/hand_styles.py (--hand-style), written by Codex; renderer blockout/blender_piano.py.
"""
import csv, json, os, pathlib, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()
ROOT = HOME / "h3/results/hand_looks"
EDIT = ROOT / "edit_v17_looks.json"          # v17 still edit; kh111 frames 2664-2670 pinned to the 2671 shot camera
COMFY_IN = HOME / "comfyui-h3-audio/input"
COMFY_OUT = HOME / "comfyui-h3-audio/output/ks_lipsync"
RUNNER = HOME / "h3audio/h3audio_run.py"
LEASE = HOME / "h3/gpu_lease.sh"
PY = str(HOME / "h3/mv_tools/.venv/bin/python")
N = 124
SHOTS = {"kh111": {"f0": 2664, "audio": "band/suite/v17_kh111.wav", "view": "Seen from high above and behind the player"},
         "ov125": {"f0": 3000, "audio": "band/suite/v17_ov125.wav", "view": "Seen from directly above the keyboard"}}
LOOKS = ("skeleton", "monster", "robot", "paw", "real")
SEED = 7


def style_args(look):
    return ["--gauntlet"] if look == "armour" else ["--hand-style", look]


def d(shot, look, kind):
    return ROOT / shot / f"{look}_{kind}"


def render(shot, look, kind, extra, procs=4):
    """Blender frames f0..f0+N-1 of one pass into ROOT/shot/look_kind/, split over `procs` processes."""
    out = d(shot, look, kind)
    out.mkdir(parents=True, exist_ok=True)
    f0 = SHOTS[shot]["f0"]
    if len(list(out.glob("f_*.png"))) >= N:
        return out
    env = dict(os.environ, RIG_SLEEVE="0.06")      # the v17 plates' sleeve length (K22/K23 used 0.06)
    step = -(-N // procs)
    ps = []
    for k in range(procs):
        a, b = f0 + k * step, min(f0 + (k + 1) * step, f0 + N)
        cmd = ["blender", "-b", "--factory-startup", "-P", str(REPO / "blockout/blender_piano.py"), "--",
               str(EDIT), str(out), "--view", "edit", "--frames", f"{a}:{b}"] + extra + style_args(look)
        ps.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / f"blender_{k}.log", "w"),
                                   stderr=subprocess.STDOUT))
    for p in ps:
        p.wait()
    n = len(list(out.glob("f_*.png")))
    assert n >= N, f"{out}: {n} frames"
    return out


def encode(src, dst, f0, gray=False, crf=14):
    vf = "scale=1280:704" + (",format=gray" if gray else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-start_number", str(f0), "-i", f"{src}/f_%05d.png",
                    "-frames:v", str(N), "-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)],
                   check=True)


def canny(src, dst):
    """Control clip: Canny (20, 60) of the normal pass, as K21-K23."""
    import cv2
    fs = sorted(src.glob("f_*.png"))[:N]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "1280x704", "-r", "24",
                          "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for f in fs:
        im = cv2.resize(cv2.imread(str(f)), (1280, 704), interpolation=cv2.INTER_AREA)
        p.stdin.write(cv2.Canny(cv2.GaussianBlur(im, (3, 3), 0), 20, 60).tobytes())
    p.stdin.close(); p.wait()


def clip(shot, look, kind):
    """Input clips live in ComfyUI's input dir (the job refers to them relative to it)."""
    return f"band/looks/{shot}_{look}_{kind}.mp4"


def passes(shot, look):
    f0 = SHOTS[shot]["f0"]
    (COMFY_IN / "band/looks").mkdir(parents=True, exist_ok=True)
    plate = render(shot, look, "plate", [])
    normal = render(shot, look, "normal", ["--pass", "normal"])
    mask = render(shot, look, "mask", ["--pass", "mask"])
    encode(plate, COMFY_IN / clip(shot, look, "plate"), f0)
    encode(normal, ROOT / shot / f"{look}_normal.mp4", f0, crf=12)
    encode(mask, ROOT / shot / f"{look}_mask.mp4", f0, gray=True, crf=10)
    canny(normal, COMFY_IN / clip(shot, look, "ncanny"))
    print("passes", shot, look, "ok")


def prompt(shot, look):
    P = json.loads((REPO / "h3_tools/hand_look_prompts.json").read_text())
    p = P["base"].format(**P["looks"][look])
    return p.replace("Seen from high above and behind the player", SHOTS[shot]["view"])


def tag(shot, look):
    return f"hl_{shot}_{look}"


def job(shot, look):
    j = {"steps": 20, "width": 1280, "height": 704, "length": N, "scheduler": "beta", "sampler": "res_multistep",
         "seed": SEED, "audio": SHOTS[shot]["audio"], "prompt": prompt(shot, look), "mode": "B",
         "video": clip(shot, look, "plate"), "video_strength": 0.88, "no_first": True,
         "control": {"video": clip(shot, look, "ncanny"), "strength": 1.0}, "tag": tag(shot, look)}
    (ROOT / "cfg").mkdir(parents=True, exist_ok=True)
    p = ROOT / "cfg" / f"{tag(shot, look)}.json"
    p.write_text(json.dumps(j, indent=1))
    return p


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // (1024 * 1024)


def run(shot, look, need=int(os.environ.get("H3_NEED_GB", 60)), poll=180):
    """One H3 render. Waits (polling) until the GPU lease can be taken with `need` GB + the 12 GB floor free."""
    p = ROOT / "cfg" / f"{tag(shot, look)}.json"     # an existing job file wins (`job` rewrites it)
    if not p.exists():
        p = job(shot, look)
    out = render_path(shot, look)
    if out.exists():
        print("exists:", out); return
    while True:
        r = subprocess.run(["bash", str(LEASE), "take", "h3-K", f"H3 hand looks {shot} {look}", str(need)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(time.strftime("%H:%M:%S"), "waiting:", r.stdout.strip(), flush=True)
        time.sleep(poll)
    try:
        subprocess.run(["python3", str(RUNNER), str(p)], check=True)
    finally:
        subprocess.run(["bash", str(LEASE), "release", "h3-K"], capture_output=True)


def render_path(shot, look):
    return COMFY_OUT / f"{tag(shot, look)}_00001_.mp4"


def score(shot, look, render_mp4=None):
    r = str(render_mp4 or render_path(shot, look))
    ref = ROOT / shot / f"{look}_normal.mp4"
    mask = ROOT / shot / f"{look}_mask.mp4"
    plate = COMFY_IN / clip(shot, look, "plate")
    mm = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(ref), r, str(mask), look + " vs normal"]).decode()
    mp = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(plate), r, str(mask), look + " vs plate"]).decode()
    fi = subprocess.check_output([PY, str(REPO / "h3_tools/fidelity.py"), str(plate), r, look]).decode()
    w, v = mm.split(), mp.split()
    row = {"shot": shot, "look": look, "flow_cos": w[w.index("flow_cos") + 1], "flow_err": w[w.index("flow_err") + 1],
           "hand_r": w[w.index("hand_r") + 1], "flow_cos_plate": v[v.index("flow_cos") + 1],
           "flow_err_plate": v[v.index("flow_err") + 1], "frame_r": fi.split("r mean")[1].split()[0],
           "shift_frames": fi.split("shift>8px frames")[1].split()[0], "render": r}
    f = ROOT / "scores.csv"
    rows = [x for x in (list(csv.DictReader(open(f))) if f.exists() else []) if (x["shot"], x["look"]) != (shot, look)]
    rows.append(row)
    with open(f, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row)); wr.writeheader(); wr.writerows(rows)
    print(mm.strip()); print(mp.strip()); print(fi.strip())
    return row


if __name__ == "__main__":
    cmd, shot, look = sys.argv[1:4]
    if cmd in ("passes", "all"):
        passes(shot, look)
    if cmd in ("job", "all"):
        print(job(shot, look))
    if cmd in ("run", "all"):
        run(shot, look)
    if cmd in ("score", "all"):
        score(shot, look, sys.argv[4] if len(sys.argv) > 4 else None)

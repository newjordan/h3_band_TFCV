"""H3 render of a grey piano edit, locked to the plates: Fun ControlNet Union 2.0 (edges of the plate) + the plate
itself in the video stream + the stem locked in the audio stream, shot by shot in 124-frame chunks.

    python -m blockout.h3_lock plan   EDIT.json FRAMES_DIR AUDIO.wav OUTDIR [--start-s 0]
    python -m blockout.h3_lock run    OUTDIR [--only c003,c004] [--limit N]
    python -m blockout.h3_lock assemble OUTDIR

plan: one entry per chunk (a still shot is cut into chunks of 124 frames, H3's trained range and a valid control
      clip length, 17k+5; the last one is padded by holding its last plate frame). Writes plates, edge controls
      and audio into the Comfy input dir.
run:  queues the chunks one after another on h3audio-comfy (~14 min each). A shot's first chunk starts free; each
      later chunk pins the previous chunk's last frame as its first frame, so the knight carries across the cut.
assemble: joins the chunks (dropping the padding and the pinned first frame), then the shots, with the audio.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

COMFY_IN = os.path.expanduser("~/comfyui-h3-audio/input")
COMFY_OUT = os.path.expanduser("~/comfyui-h3-audio/output")
RUNNER = os.path.expanduser("~/h3audio/h3audio_run.py")
SUB = "band/v17c"         # (per run: band/<tag>, tag = OUTDIR name without "h3_")
FPS = 24
CHUNK = 124
VALID = [5 + 17 * k for k in range(1, 8)]          # 22 .. 124
W, H = 1280, 704

KNIGHT = ("the shiny armored knight's hands: mirror-polished chrome silver articulated gauntlets with jointed steel "
          "fingers, gleaming chrome vambraces over chainmail sleeves")
VIEW = {"keys_high": "Seen from high above and behind the player",
        "overhead": "Seen from directly above the keyboard",
        "keys_3q_R": "Seen from high above and to the side of the player",
        "keys_3q_L": "Seen from high above and to the side of the player"}
MOOD = {"debussy": "a solo grand piano, soft and dreamy", "rach": "a solo grand piano, dramatic and stormy",
        "mozart": "a solo grand piano, bright and playful", "chopin": "a solo grand piano, slow and tender"}


def prompt(view, mood):
    return (f"One continuous locked-off shot from a high-budget 1987 rock music video, pristine 35mm film transfer, "
            f"deep blacks, fine film grain, a crisp fast-shutter image with no motion blur. A smoky late-night jazz cafe: "
            f"cigarette smoke curls and drifts over the keyboard, warm amber lamp light and candle glow, glints on the "
            f"chrome. {VIEW.get(view, VIEW['keys_high'])}: {KNIGHT} play the keys of a glossy black grand piano. The "
            f"camera does not move. Each key goes "
            f"down under the steel finger that presses it, exactly following the motion. No cuts, no new objects, no "
            f"faces, no text, no logo. Audio: {MOOD.get(mood, MOOD['rach'])}.")


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def section_of(t, starts):
    s = "debussy"
    for name, t0 in starts:
        if t >= t0:
            s = name
    return s


def cmd_plan(a):
    anim = json.load(open(a.edit))
    shots = anim["shots"]
    od = a.outdir
    os.makedirs(od, exist_ok=True)
    starts = json.loads(a.sections) if a.sections else [("debussy", 0.0)]
    chunks = []
    for si, s in enumerate(shots):
        f0, f1 = int(round(s["start"] * FPS)), int(round(s["end"] * FPS))
        k = 0
        f = f0
        while f < f1:
            n = min(CHUNK, f1 - f)
            gen = min(v for v in VALID if v >= max(n, VALID[0]))
            cid = f"s{si:02d}c{k}"
            chunks.append({"id": cid, "shot": si, "k": k, "view": s["shot"], "start": f, "n": n, "gen": gen,
                           "section": section_of(s["start"] + a.start_s, starts)})
            f += n - (1 if f + n < f1 else 0)       # chunks overlap by one frame: the pinned first frame
            k += 1
    SUB = f"band/{run_tag(od)}"
    os.makedirs(os.path.join(COMFY_IN, SUB), exist_ok=True)
    for c in chunks:
        d = os.path.join(od, "plates", c["id"])
        os.makedirs(d, exist_ok=True)
        for i in range(c["gen"]):
            src = os.path.join(a.frames, f"f_{c['start'] + min(i, c['n'] - 1):05d}.png")
            dst = os.path.join(d, f"p_{i:05d}.png")
            if not os.path.exists(dst):
                os.symlink(src, dst)
        base = os.path.join(COMFY_IN, SUB, c["id"])
        if not os.path.exists(base + ".mp4"):
            sh(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(d, "p_%05d.png"),
                "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", base + ".mp4"])
            sh([os.path.expanduser("~/comfyui-h3-audio/.venv/bin/python"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h3_tools", "make_controls.py"), d, base + "_ctl", str(W), str(H)])
            sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{a.start_s + c['start'] / FPS:.5f}", "-i", a.audio, "-af", "apad",
                "-t", f"{c['gen'] / FPS:.5f}", "-ar", "48000", "-ac", "2", base + ".wav"])
    json.dump({"edit": a.edit, "frames": a.frames, "audio": a.audio, "start_s": a.start_s, "chunks": chunks},
              open(os.path.join(od, "plan.json"), "w"), indent=1)
    print(f"{len(chunks)} chunks from {len(shots)} shots; ~{len(chunks) * 14 / 60:.1f} GPU hours")


def run_tag(od):
    return os.path.basename(os.path.normpath(od)).replace("h3_", "")


def config(c, prev_last=None, tag="v17c"):
    base = f"band/{tag}/{c['id']}"
    j = {"tag": f"{tag}_{c['id']}", "mode": "B", "video": base + ".mp4", "audio": base + ".wav",
         "video_strength": 0.88, "control": {"video": base + "_ctl_canny.mp4", "strength": 1.0},
         "prompt": prompt(c["view"], c["section"]), "seed": 7, "steps": 20, "width": W, "height": H,
         "length": c["gen"], "scheduler": "beta", "sampler": "res_multistep"}
    if prev_last:
        j["image"] = prev_last
    else:
        j["no_first"] = True
    return j


def output_frames(tag):
    return sorted(glob.glob(os.path.join(COMFY_OUT, "ks_lipsync", tag, "frame_*.png")))


def cmd_run(a):
    od = a.outdir
    P = json.load(open(os.path.join(od, "plan.json")))
    log = open(os.path.join(od, "run.log"), "a")
    done = 0
    for c in P["chunks"]:
        if a.only and c["id"] not in a.only.split(","):
            continue
        T = run_tag(od)
        SUB = f"band/{T}"
        tag = f"{T}_{c['id']}"
        if output_frames(tag):
            continue
        prev_last = None
        if c["k"] > 0:
            pc = next(x for x in P["chunks"] if x["shot"] == c["shot"] and x["k"] == c["k"] - 1)
            fr = output_frames(f"{T}_{pc['id']}")
            if not fr:
                print(f"{c['id']}: previous chunk not rendered yet, skipping", file=log, flush=True)
                continue
            last = fr[pc["n"] - 1]
            prev_last = f"{SUB}/{c['id']}_first.png"
            shutil.copy2(last, os.path.join(COMFY_IN, prev_last))
        cfg = os.path.join(od, "configs", c["id"] + ".json")
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        json.dump(config(c, prev_last, T), open(cfg, "w"), indent=1)
        t = time.time()
        r = subprocess.run([sys.executable, RUNNER, cfg], capture_output=True, text=True)
        print(f"{time.strftime('%H:%M:%S')} {c['id']} ({c['view']}, {c['section']}, {c['n']}/{c['gen']} f) "
              f"{'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f}s {r.stdout.strip()[-200:]}", file=log, flush=True)
        done += 1
        if a.limit and done >= a.limit:
            break


def cmd_assemble(a):
    od = a.outdir
    P = json.load(open(os.path.join(od, "plan.json")))
    seq = os.path.join(od, "assembled")
    shutil.rmtree(seq, ignore_errors=True); os.makedirs(seq)
    i = 0
    missing = 0
    for c in P["chunks"]:
        fr = output_frames(f"{run_tag(od)}_{c['id']}")
        take = range(1 if c["k"] > 0 else 0, c["n"])
        for k in take:
            src = fr[k] if k < len(fr) else os.path.join(P["frames"], f"f_{c['start'] + k:05d}.png")   # grey if not rendered
            missing += k >= len(fr)
            os.symlink(src, os.path.join(seq, f"a_{i:05d}.png")); i += 1
    out = os.path.join(od, f"{run_tag(od)}_knight.mp4")
    sh(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(seq, "a_%05d.png"), "-ss",
        f"{P['start_s']:.3f}", "-i", P["audio"], "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out])
    print(f"{out}: {i} frames, {missing} still grey")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan"); p.add_argument("edit"); p.add_argument("frames"); p.add_argument("audio"); p.add_argument("outdir")
    p.add_argument("--start-s", type=float, default=0.0); p.add_argument("--sections", default="")
    r = sub.add_parser("run"); r.add_argument("outdir"); r.add_argument("--only", default=""); r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("assemble"); s.add_argument("outdir")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()

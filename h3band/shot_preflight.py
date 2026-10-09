#!/usr/bin/env python3
"""Shot pre-flight for isolated-run H3 shots: pre-isolate -> isolated runs -> merge.

  shot_preflight.py SHOT.json [--no-stage]

Reads a shot spec, checks it, cuts the shot's stems out of the song, writes a per-latent-frame activity
timeline for each zone, and emits ready-to-queue h3_run.py jobs (base or plate, one isolated run per
zone, merge). CPU only; touches nothing but files.

Shot spec (JSON, see examples/shot_example.json):
  id            shot name
  song_start    seconds into the song where the render's frame 0 sits (include any preroll)
  length        frames at 24 fps (snapped up to the 17k+5 grid)
  width, height canvas (multiples of 32)
  image         first frame for the base run, relative to the ComfyUI input dir
  plate         optional: an existing take (input-relative mp4) to start from instead of a base run
  prompt        base prompt; prompts: {zone: prompt} for the isolated runs; prompt_merge (default: prompt)
  seed, steps, scheduler (optional)
  stems         optional {name: path} overriding config.STEMS_DIR/<STEM_FILES[name]>
  zones         [{name, stem, box_start "x0,y0,x1,y1", box_end (optional), denoise (default 1.0)}]
                stem: vocals | backing | guitar | drums | bass | mix; boxes in canvas pixels, linear
                first->last frame. Runs go in list order.
  merge_denoise default 0.25; "merge": false skips the merge job

Zone sizing: a lip sync works with a head box at ~0.92, but instrument motion needs denoise 1.0 and a
box over the whole performer AND the whole instrument: anything locked (a guitar neck, a raised arm)
pins the pose.

Outputs: SHOTS_DIR/<id>/{preflight.json, report.txt, jobs/*.json}; cut stems staged into
COMFY_INPUT/<OUTPUT_PREFIX>/<id>/ (the paths the jobs reference).
"""
import json, math, pathlib, sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

FPS = 24
AUDIO_LATENT_FPS = 40
TOK = 32          # px per DiT token (16x VAE * 2x2 patch)
CELL = 16         # px per latent cell (zone box granularity)
VOICED_DB = 40.0  # same rule as vocal_activity.py

import config

INPUT = config.COMFY_INPUT
GUITAR_EVENTS = config.GUITAR_EVENTS


def stem_paths(spec):
    over = spec.get("stems", {})
    out = {}
    for name, fname in config.STEM_FILES.items():
        p = pathlib.Path(over[name]).expanduser() if name in over else config.STEMS_DIR / fname
        out[name] = (p, "spec" if name in over else f"{config.STEMS_DIR.name}/{fname}")
    for name, path in over.items():
        out.setdefault(name, (pathlib.Path(path).expanduser(), "spec"))
    return out


def align_frame_count(n):
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def latent_index(f):
    clip, r = divmod(f, 17)
    return clip * 5 + (0 if r == 0 else 1 + (r - 1) // 4)


def envelope(x, sr):
    x = sosfiltfilt(butter(4, [150, 4000], btype="band", fs=sr, output="sos"), x.mean(axis=1))
    hop = sr / FPS
    n = int(len(x) / hop)
    return 20 * np.log10(np.array([np.sqrt(np.mean(x[int(k * hop):int((k + 1) * hop)] ** 2)) for k in range(n)]) + 1e-8)


def parse_box(s):
    b = [float(v) for v in (s.split(",") if isinstance(s, str) else s)]
    if len(b) != 4:
        raise ValueError(f"box needs x0,y0,x1,y1: {s}")
    return b


def zone_tokens(zone, frame_count, latent_t, W, H):
    """Per latent frame: the set of tokens the zone's box regenerates (any touched 16-px cell frees its token)."""
    b0 = parse_box(zone["box_start"]); b1 = parse_box(zone.get("box_end") or zone["box_start"])
    fpt = (1, 4, 4, 4, 4)
    starts, f = [], 0
    for k in range(latent_t):
        starts.append(f); f += fpt[k % 5]
    w_tok, h_tok = math.ceil(W / TOK), math.ceil(H / TOK)
    out, boxes = [], []
    for k in range(latent_t):
        u = (starts[k] + (fpt[k % 5] - 1) / 2.0) / max(1, f - 1)
        x0, y0, x1, y1 = [a + (b - a) * u for a, b in zip(b0, b1)]
        cx0, cy0 = max(0, int(x0 // CELL)), max(0, int(y0 // CELL))
        cx1, cy1 = min(W // CELL, int(-(-x1 // CELL))), min(H // CELL, int(-(-y1 // CELL)))
        m = np.zeros((h_tok, w_tok), dtype=bool)
        if cx1 > cx0 and cy1 > cy0:
            m[cy0 // 2:(cy1 + 1) // 2, cx0 // 2:(cx1 + 1) // 2] = True
        out.append(m); boxes.append((x0, y0, x1, y1))
    return np.stack(out), boxes, (b0, b1)


def latent_activity(active_frames, latent_t):
    acc = np.zeros(latent_t); cnt = np.zeros(latent_t)
    for f, a in enumerate(active_frames):
        k = latent_index(f)
        if k < latent_t:
            acc[k] += a; cnt[k] += 1
    return acc / np.maximum(cnt, 1)


def main():
    spec_path = pathlib.Path(sys.argv[1])
    stage = "--no-stage" not in sys.argv
    s = json.loads(spec_path.read_text())
    sid = s["id"]
    W, H = int(s["width"]), int(s["height"])
    frame_count = align_frame_count(int(s["length"]))
    latent_t = (frame_count - 5) // 17 * 5 + 2
    dur = frame_count / FPS
    audio_t = round(dur * AUDIO_LATENT_FPS)
    t0 = float(s["song_start"]); t1 = t0 + dur
    checks, warn, fail = [], 0, 0

    def check(level, msg):
        nonlocal warn, fail
        checks.append(f"{level:4s}  {msg}")
        warn += level == "WARN"; fail += level == "FAIL"

    # grid + canvas
    check("OK" if frame_count == int(s["length"]) else "WARN",
          f"length {s['length']} -> {frame_count} frames ({dur:.3f} s), {latent_t} latent frames, {audio_t} audio latents")
    for name, v in (("width", W), ("height", H)):
        check("OK" if v % 32 == 0 else "FAIL", f"{name} {v} {'is' if v % 32 == 0 else 'is NOT'} a multiple of 32")
    w_tok, h_tok = math.ceil(W / TOK), math.ceil(H / TOK)
    check("OK", f"token grid {w_tok}x{h_tok} per latent frame, {w_tok * h_tok * latent_t} video tokens")
    if s.get("plate"):
        p = INPUT / s["plate"]
        check("OK" if p.exists() else "FAIL", f"plate take {'found' if p.exists() else 'MISSING'}: {p}")
    elif s.get("image"):
        p = INPUT / s["image"]
        check("OK" if p.exists() else "FAIL", f"base image {'found' if p.exists() else 'MISSING'}: {p}")
    else:
        check("FAIL", "need either image (base run) or plate (existing take)")

    # stems + window
    stems = stem_paths(s)
    song_len = sf.info(str(stems["mix"][0])).duration if stems["mix"][0].exists() else 0.0
    check("OK" if t1 <= song_len else "FAIL", f"song window {t0:.3f}-{t1:.3f} s of {song_len:.2f} s")
    required = {z["stem"] for z in s["zones"]} | {"mix"}
    used = required | {"vocals", "backing"}  # vocals/backing feed the backing-vocal check when present
    cut, act = {}, {}
    shot_dir = config.SHOTS_DIR / sid
    (shot_dir / "jobs").mkdir(parents=True, exist_ok=True)
    stage_dir = INPUT / config.OUTPUT_PREFIX / sid
    for name in sorted(used):
        if name not in stems:
            check("FAIL", f"unknown stem '{name}' (have {', '.join(stems)})"); continue
        path, src = stems[name]
        if not path.exists():
            if name in required:
                check("FAIL", f"stem {name} missing: {path}")
            continue
        x, sr = sf.read(str(path), always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        env = envelope(x, sr)
        p95 = float(np.percentile(env, 95))
        a, b = int(round(t0 * sr)), int(round(t1 * sr))
        seg = x[a:b]
        f0 = int(round(t0 * FPS))
        frames = env[f0:f0 + frame_count]
        frames = np.pad(frames, (0, frame_count - len(frames)), constant_values=-160)
        active = frames > p95 - VOICED_DB
        act[name] = {"active_frac": round(float(active.mean()), 3),
                     "latent_active": [round(float(v), 2) for v in latent_activity(active, latent_t)]}
        out = (stage_dir if stage else shot_dir / "stems") / f"{name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), seg, sr, subtype="PCM_16")
        cut[name] = {"src": str(path), "model": src, "out": str(out), "sr": sr,
                     "seconds": round(len(seg) / sr, 3)}
        check("OK" if len(seg) >= (b - a) else "FAIL", f"stem {name:8s} cut {len(seg) / sr:.3f} s ({src}), active {active.mean():.0%}")

    # zones: geometry, overlap, seams, drive signal
    zinfo, ztok = [], {}
    names = [z["name"] for z in s["zones"]]
    if len(set(names)) != len(names):
        check("FAIL", "zone names must be unique")
    for z in s["zones"]:
        tok, boxes, (b0, b1) = zone_tokens(z, frame_count, latent_t, W, H)
        ztok[z["name"]] = tok
        inside = all(0 <= v[0] < v[2] <= W and 0 <= v[1] < v[3] <= H for v in (b0, b1))
        check("OK" if inside else "FAIL", f"zone {z['name']}: box {'inside' if inside else 'OUTSIDE'} canvas, "
              f"{tok.sum(axis=(1, 2)).mean():.0f} tokens/latent frame ({tok.mean():.1%} of frame)")
        info = {"name": z["name"], "stem": z["stem"], "denoise": z.get("denoise", 1.0),
                "box_start": b0, "box_end": b1, "tokens_per_latent_frame": tok.sum(axis=(1, 2)).tolist(),
                "frame_share": round(float(tok.mean()), 4)}
        if z["stem"] in act:
            la = np.array(act[z["stem"]]["latent_active"])
            info["drive_latent_active"] = act[z["stem"]]["latent_active"]
            if la.max() == 0:
                check("WARN", f"zone {z['name']}: stem {z['stem']} is silent for the whole shot; the isolated run has nothing to follow")
            else:
                check("OK", f"zone {z['name']}: {z['stem']} active in {np.mean(la > 0.5):.0%} of latent frames")
        if z["stem"] == "guitar" and GUITAR_EVENTS.exists():
            ev = [e for e in json.loads(GUITAR_EVENTS.read_text())["events"] if t0 <= e["t"] < t1]
            per_lat = np.zeros(latent_t, dtype=int)
            for e in ev:
                per_lat[min(latent_t - 1, latent_index(int((e["t"] - t0) * FPS)))] += 1
            info["guitar_onsets"] = len(ev)
            info["guitar_onsets_per_latent_frame"] = per_lat.tolist()
            info["guitar_down"] = sum(e["direction"] == "down" for e in ev)
            info["guitar_up"] = sum(e["direction"] == "up" for e in ev)
            dense = int((per_lat > 1).sum())
            check("OK" if len(ev) else "WARN", f"zone {z['name']}: {len(ev)} strum onsets in shot "
                  f"({info['guitar_down']} down / {info['guitar_up']} up); {dense} latent frames carry >1 onset "
                  f"(finer than a video token: timing must come from the 40 Hz audio tokens)")
        zinfo.append(info)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = ztok[names[i]], ztok[names[j]]
            ov = (a & b).sum(axis=(1, 2))
            if ov.max():
                check("WARN", f"zones {names[i]}/{names[j]} share up to {ov.max()} tokens in {int((ov > 0).sum())} latent frames: "
                      f"the later run ({names[j]}) overwrites them; tighten boxes or accept that band as merge seam")
            else:
                # gap in tokens between the zones (Chebyshev), worst frame
                gaps = []
                for k in range(latent_t):
                    ya, xa = np.nonzero(a[k]); yb, xb = np.nonzero(b[k])
                    if len(ya) and len(yb):
                        gaps.append(max(0, max(xb.min() - xa.max(), xa.min() - xb.max(), yb.min() - ya.max(), ya.min() - yb.max()) - 1))
                g = min(gaps) if gaps else None
                check("OK" if g is None or g >= 1 else "WARN", f"zones {names[i]}/{names[j]}: no shared tokens, "
                      f"closest gap {g} token(s) ({'touching: the merge pass owns that seam' if g == 0 else 'clear'})")

    # the lead singer must not lip backing lines
    if "vocals" in act and "backing" in act:
        lv, bv = np.array(act["vocals"]["latent_active"]), np.array(act["backing"]["latent_active"])
        risk = int(((bv > 0.5) & (lv < 0.5)).sum())
        check("WARN" if risk else "OK", f"backing vocals without lead in {risk} latent frames"
              + (" (lead-singer zone should stay closed-mouthed there; prompt it)" if risk else ""))

    # jobs (h3_run.py format)
    base = {k: s[k] for k in ("seed", "steps", "width", "height") if k in s}
    base["length"] = frame_count
    if s.get("scheduler"):
        base["scheduler"] = s["scheduler"]
    pre = f"{config.OUTPUT_PREFIX}/{sid}"
    rel = lambda name: f"{pre}/{name}.wav"
    jobs = []
    if s.get("plate"):
        prev_video = s["plate"]
        prev = "plate"
    else:
        jobs.append(dict(base, tag=f"{sid}_00_base", mode="A", image=s.get("image", ""), audio=rel("mix"),
                         prompt=s["prompt"], plan="base: staging plate, full mix frozen in the audio stream"))
        prev = jobs[0]["tag"]
        prev_video = f"{pre}/{prev}.mp4"
    for n, z in enumerate(s["zones"], 1):
        zi = zinfo[n - 1]
        jobs.append(dict(base, tag=f"{sid}_{n:02d}_{z['name']}", mode="B", video=prev_video,
                         audio=rel(z["stem"]), prompt=s.get("prompts", {}).get(z["name"], s["prompt"]),
                         video_denoise=0.0, zone_box_start=",".join(f"{v:g}" for v in zi["box_start"]),
                         zone_box_end=",".join(f"{v:g}" for v in zi["box_end"]), zone_denoise=zi["denoise"],
                         depends_on=prev, plan=f"isolated run: only {z['name']} regenerates, driven by the {z['stem']} stem; "
                                               f"everything else locked to {prev}"))
        prev = jobs[-1]["tag"]
        prev_video = f"{pre}/{prev}.mp4"
    if s.get("merge", True):
        jobs.append(dict(base, tag=f"{sid}_{len(s['zones']) + 1:02d}_merge", mode="B", video=prev_video,
                         audio=rel("mix"), prompt=s.get("prompt_merge", s["prompt"]),
                         video_denoise=float(s.get("merge_denoise", 0.25)), depends_on=prev,
                         plan="merge: whole frame at low denoise under the full mix, resolves seams"))
    for j in jobs:
        (shot_dir / "jobs" / f"{j['tag']}.json").write_text(json.dumps(j, indent=1))
    check("OK", f"{len(jobs)} jobs written; run_chain.py stages each take to {INPUT}/{pre}/<tag>.mp4 for the next job")

    verdict = "FAIL" if fail else ("GO with warnings" if warn else "GO")
    report = [f"PRE-FLIGHT {sid}: {verdict}  ({fail} fail, {warn} warn)",
              f"song {t0:.3f}-{t1:.3f} s | {W}x{H} | {frame_count} frames | {latent_t} latent | {audio_t} audio latents", ""]
    report += checks + ["", "RUN ORDER"] + [f"  {j['tag']}: {j['plan']}" for j in jobs]
    (shot_dir / "report.txt").write_text("\n".join(report) + "\n")
    (shot_dir / "preflight.json").write_text(json.dumps(
        {"id": sid, "verdict": verdict, "song_window": [t0, t1], "frame_count": frame_count, "latent_t": latent_t,
         "audio_t": audio_t, "canvas": [W, H], "token_grid": [w_tok, h_tok], "stems": cut, "activity": act,
         "zones": zinfo, "checks": checks, "jobs": [j["tag"] for j in jobs]}, indent=1))
    print("\n".join(report))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()

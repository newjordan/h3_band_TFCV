#!/usr/bin/env python3
"""Run a shot's isolated-run chain in order, handing each take to the next job, scoring as it goes.

  run_chain.py SHOT_ID [--from TAG] [--dry-run]

Jobs come from SHOTS_DIR/<id>/jobs/*.json (written by shot_preflight.py), run in file-name order through
h3_run.py. After each success the take (COMFY_OUTPUT/<OUTPUT_PREFIX>/<tag>_*.mp4) is copied to
COMFY_INPUT/<OUTPUT_PREFIX>/<id>/<tag>.mp4, where the next mode-B job reads it. Every zone is scored on
every take: vocal zones with sync_score.py against their stem, guitar zones with strum_score.py against
the guitar onsets (GUITAR_EVENTS). Stops at the first failure. Log: SHOTS_DIR/<id>/chain.json.
--dry-run only builds each job's graph (no server needed).
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
PY = sys.executable


def score_take(mp4, pre):
    """Score every zone on this take, whatever stage it is: shows sync appear and survive."""
    out = {}
    for z in pre["zones"]:
        box = ",".join(f"{v:g}" for v in z["box_start"])
        if z["stem"] == "guitar":
            cmd = [PY, str(HERE / "strum_score.py"), str(mp4), str(config.GUITAR_EVENTS),
                   str(pre["song_window"][0]), "--box", box, "--json"]
        elif z["stem"] in pre["stems"]:
            cmd = [PY, str(HERE / "sync_score.py"), str(mp4), pre["stems"][z["stem"]]["out"], "--box", box]
        else:
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            out[z["name"]] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            out[z["name"]] = {"error": (r.stderr or r.stdout)[-500:]}
    return out


def main():
    sid = sys.argv[1]
    dry = "--dry-run" in sys.argv
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    shot = config.SHOTS_DIR / sid
    pre = json.loads((shot / "preflight.json").read_text())
    if pre["verdict"] == "FAIL":
        sys.exit(f"pre-flight FAIL for {sid}; fix the spec first")
    jobs = sorted((shot / "jobs").glob("*.json"))
    stage_dir = config.COMFY_INPUT / config.OUTPUT_PREFIX / sid
    log_path = shot / "chain.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"shot": sid, "stages": []}
    started = start is None
    for jp in jobs:
        j = json.loads(jp.read_text())
        if not started:
            started = j["tag"] == start
            if not started:
                continue
        if dry:
            r = subprocess.run([PY, str(HERE / "h3_run.py"), str(jp), "--dump"], capture_output=True, text=True)
            ok = r.returncode == 0 and '"SaveVideo"' in r.stdout
            print(f"[dry] {j['tag']}: graph {'OK' if ok else 'FAILED'}" + ("" if ok else f"\n{r.stderr[-800:]}"))
            continue
        if j.get("mode") == "B" and not (config.COMFY_INPUT / j["video"]).exists():
            sys.exit(f"{j['tag']}: missing input take {j['video']}")
        t0 = time.time()
        print(f"[chain] {j['tag']}: {j.get('plan', '')}", flush=True)
        subprocess.run([PY, str(HERE / "h3_run.py"), str(jp)], check=False)
        res = json.loads((config.RESULTS_DIR / f"{j['tag']}.json").read_text())
        stage = {"tag": j["tag"], "status": res["status"], "wall_s": res["wall_s"]}
        if res["status"] != "success":
            log["stages"].append(stage)
            log_path.write_text(json.dumps(log, indent=1))
            sys.exit(f"[chain] {j['tag']} {res['status']}: stopping")
        take = sorted((config.COMFY_OUTPUT / config.OUTPUT_PREFIX).glob(f"{j['tag']}_*.mp4"))[-1]
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(take, stage_dir / f"{j['tag']}.mp4")
        stage["take"] = str(take)
        stage["scores"] = score_take(take, pre)
        stage["chain_s"] = round(time.time() - t0, 1)
        log["stages"].append(stage)
        log_path.write_text(json.dumps(log, indent=1))
        keys = ("jaw_peak_xcorr", "jaw_closed_in_silence", "peak_corr", "margin_vs_control")
        print(f"[chain] {j['tag']} done: " + json.dumps({k: {m: v.get(m) for m in keys if m in v}
                                                         for k, v in stage["scores"].items()}), flush=True)


if __name__ == "__main__":
    main()

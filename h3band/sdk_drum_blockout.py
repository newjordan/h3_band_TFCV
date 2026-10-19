#!/usr/bin/env python3
"""Drum stem (or GM drum MIDI) -> grey drum-kit blockout video, run on ComfyUI through the Comfy SDK.

  sdk_drum_blockout.py SRC OUT.mp4 [--audio DRUMS.wav] [--start S] [--frames N] [--fps 24]
                       [--view front|three4|side|over|top] [--camera ex,ey,ez,lx,ly,lz] [--res 1280x704]
                       [--heads 0:groove] [--seed 0] [--workflow API.json]

SRC is a drum stem (.wav/.flac/...; hits detected by H3 Band Drum Events) or a .mid (H3 Band Drum Hits).
--audio is the track under the video: required with a MIDI (e.g. a blockout.drumsynth render of it),
otherwise SRC itself. Files are uploaded as SDK assets; the job runs examples/workflows/
drum_blockout_api.json with the inputs set here and the SaveVideo output is downloaded to OUT.mp4.

Talks to COMFY_BASE_URL (default http://127.0.0.1:8189: comfy-api-proxy in front of a local ComfyUI with
comfy_nodes/ installed and Blender on the server). COMFY_API_KEY is passed if set.
"""
import argparse, json, os, sys
from pathlib import Path

from comfy_sdk import Comfy, OutputReady, Progress, StatusChange

WORKFLOW = Path(__file__).resolve().parent.parent / "examples" / "workflows" / "drum_blockout_api.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--audio", help="audio under the video (default: SRC when it is audio)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--view", default="front")
    ap.add_argument("--camera", default="")
    ap.add_argument("--res", default="1280x704")
    ap.add_argument("--heads", default="0:groove", help="head style timeline, e.g. 0:focused,8:groove,16:wild")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workflow", default=str(WORKFLOW))
    a = ap.parse_args()
    is_midi = a.src.lower().endswith((".mid", ".midi"))
    audio = a.audio or (None if is_midi else a.src)
    if audio is None:
        sys.exit("a MIDI source needs --audio for the video's soundtrack")
    w, h = map(int, a.res.split("x"))

    os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
    client = Comfy(api_key=os.environ.get("COMFY_API_KEY"))
    graph = json.load(open(a.workflow))
    if is_midi:
        graph["2"] = {"class_type": "H3BandDrumHitsMIDI", "inputs": {"midi": ""}}
    wf = client.workflows.from_json(graph)
    wf.set_input("1", "audio", client.assets.from_file(audio))
    if is_midi:
        wf.set_input("2", "midi", client.assets.from_file(a.src))
    for k, v in {"start": a.start, "frames": a.frames, "fps": a.fps, "width": w, "height": h, "view": a.view,
                 "camera": a.camera, "head_schedule": a.heads, "seed": a.seed}.items():
        wf.set_input("3", k, v)
    wf.set_input("4", "start_index", a.start)
    wf.set_input("4", "duration", a.frames / a.fps)
    wf.set_input("5", "fps", float(a.fps))

    job = client.submit(wf)
    print(f"[sdk] job {job.id} on {os.environ['COMFY_BASE_URL']}", flush=True)
    shown = -1
    for ev in job.events():
        if isinstance(ev, Progress) and int(ev.value * 10) > shown:
            shown = int(ev.value * 10)
            print(f"[sdk] {ev.value:.0%}", flush=True)
        elif isinstance(ev, OutputReady):
            print(f"[sdk] output {ev.output.name}", flush=True)
        elif isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled"):
            break
    job = job.result()
    out = job.get_outputs("6")[0]
    out.to_file(a.out)
    print(f"[sdk] {out.name} -> {a.out}")


if __name__ == "__main__":
    main()

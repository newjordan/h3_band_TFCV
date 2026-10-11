"""Reproducible isolated piano realism probes, without Blender or H3.

Cached example (uses original v19 part, preserving its complete fingering):
  python lab/realism_probe.py --source-root /home/frosty40/h3b_wrist \
    --cached /tmp/claude-1000/suite_v19/part_14.json --start 248 --dur 4 --out artifacts/realism/e36_crossing

Fresh comparable solve: omit --cached, use --warmup 1 and the same start/dur/fps.
Change --source-root to select the original or candidate checkout. Default E36
overrides can be replaced with --overrides JSON. BLAS threads are capped before
numpy imports. Run separate processes for candidates so import state cannot leak.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

E36 = {"PRESS_AXES": [20, 1, 20], "PRESS_AXES_BLACK": [20, 20, 20],
       "REPAIR_ITERS": 2, "PRESS_OVERSHOOT": .003, "REACH_SPLIT": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cached", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--dur", type=float, default=3.)
    parser.add_argument("--warmup", type=float, default=1.)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hand-scale", type=float, default=1.)
    parser.add_argument("--overrides", default=json.dumps(E36))
    parser.add_argument("--out", type=Path, required=True, help="output prefix, without .json")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    os.environ["HAND_SCALE"] = str(args.hand_scale)
    os.environ["PIANO_OVR"] = json.dumps(json.loads(args.overrides))
    sys.path.insert(0, str(args.source_root))
    source_files = [p for p in ("blockout/piano.py", "blockout/rig/hand.py", "blockout/rig/skeleton.py",
                               "blockout/rig/ik.py", "blockout/performer.py", "blockout/contact.py")
                    if (args.source_root / p).exists()]
    fingerprint = lambda: {p: hashlib.sha256((args.source_root / p).read_bytes()).hexdigest() for p in source_files}
    source_fingerprint = fingerprint()
    import numpy as np
    from blockout import midi, feeling, piano
    spec = importlib.util.spec_from_file_location("realism_audit", Path(__file__).resolve().parents[1] / "blockout/rig_qa.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit_fingerprint = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    mid = args.source_root / "examples/piano/knights_suite.mid"
    chart_path = args.source_root / "examples/piano/knights_suite.feeling.json"
    notes = midi.read(mid)
    beats, downbeats = midi.beats(mid)
    chart = feeling.Chart(chart_path, downbeats)
    notes, timing = feeling.perform(notes, chart, 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.cached:
        anim = json.loads(args.cached.read_text())
    else:
        piano.set_hand_model("mpfb")
        # Observe the expensive solver without changing any inputs or results.
        # The small progress file makes a live run distinguishable from a hang.
        original_solve = piano._solve_frame
        progress = {"calls": 0, "frame": 0, "passes": 1}
        def observed_solve(*a, **kw):
            frame = int(a[1])
            if frame < progress["frame"]:
                progress["passes"] += 1
            progress["calls"] += 1
            progress["frame"] = frame
            if progress["calls"] % 24 == 1:
                args.out.with_suffix(".progress.json").write_text(json.dumps(progress | {"elapsed_s": time.monotonic() - t0}))
            return original_solve(*a, **kw)
        piano._solve_frame = observed_solve
        energy = lambda t: float(np.clip(1.4 * chart.at("strength", np.interp(t, timing.tp, timing.ts)), 0, 1))
        s = max(0., args.start - args.warmup)
        anim = piano.animate(notes, args.fps, s, args.start - s + args.dur + .6, 1.,
                             [float(timing(b)) for b in beats], [float(timing(b)) for b in downbeats],
                             [(float(timing(t)), h) for t, h in chart.heads], energy, 0)
        args.out.with_suffix(".anim.json").write_text(json.dumps(anim))
    solve_s = time.monotonic() - t0
    report = audit.analyze(anim, notes, start=args.start, duration=args.dur)
    report["provenance"] = {"source_root": str(args.source_root), "cached": str(args.cached) if args.cached else None,
                            "overrides": json.loads(args.overrides), "hand_scale": args.hand_scale,
                            "warmup_s": args.warmup, "load_or_solve_s": solve_s, "audit_s": time.monotonic() - t0 - solve_s,
                            "source_sha256": source_fingerprint, "source_unchanged_during_run": fingerprint() == source_fingerprint}
    report["provenance"]["audit_sha256"] = audit_fingerprint
    report["provenance"]["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True).strip()
    dest = args.out.with_suffix(".qa.json")
    dest.write_text(json.dumps(report, indent=2) + "\n")
    args.out.with_suffix(".progress.json").write_text(json.dumps({"complete": True, "elapsed_s": time.monotonic() - t0}))
    print(json.dumps({"report": str(dest), "seconds": time.monotonic() - t0,
                      "notes": report["notes"], "separation": report["separation"], "bounds": report["bounds"], "motion": report["motion"]}, indent=2))


if __name__ == "__main__":
    main()

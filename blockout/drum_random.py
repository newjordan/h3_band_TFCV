"""Random beats against overfitting: plays the drummer on seeded random grooves (drum_groove.random_bars) that
nothing was tuned on, at random tempos, on random kits, under random emotions, with and without the mocap head and
trunk, and checks every take the way the fixed grooves are checked (drums.check and drum_collide's clip report).

    python -m blockout.drum_random [--n 12] [--seed 0] [--out work/random] [--jobs N] [--song random|tool|baker]
                                   [--kits big,standard]

A take fails when a stick is more than 1 mm off its strike point on the hit frame, a tip goes through a head, a
beater isn't on the head for a kick, the arm reaches further than it is long, anything clips, a hand note goes
unplayed, or drums.py stops. Prints a line per take, the command that re-makes each failed one, and a summary;
writes OUT/summary.json. Take k is the same every run (its seed is --seed + k). Takes run one per core, each
with numpy on one thread (--jobs, default all cores but two). --song plays the drummer on another seeded generator
instead (tool: Danny Carey's idiom at 95-135 bpm, baker: Ginger Baker's at 100-170 bpm), and its mocap takes
use that drummer's head and trunk (drum_blend.DRUMMERS; Chad Smith's on the random grooves); --kits sets the kits
that take turns (default every preset).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import drum_groove, drum_kit, drum_style, drums, midi

KITS = list(drum_kit.PRESETS)
MOODS = list(drum_style.PRESETS)
MOCAP = {"random": "chad", "tool": "carey", "baker": "baker"}     # drum_blend.DRUMMERS: whose mocap a song's takes use
CONTACT_MM = 1.0
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
TEMPO = {"random": (80, 170), "tool": (95, 135), "baker": (100, 170)}      # bpm, inclusive


def takes(n, seed=0, song="random", kits=KITS):
    """n take settings: tempo in the song's range (80-170 bpm for random), one emotion or two (the second from
    halfway); the kits take turns, with the mocap head and trunk every other round."""
    out = []
    for s in range(seed, seed + n):
        rng = np.random.default_rng([s, 11])
        bpm = int(rng.integers(TEMPO[song][0], TEMPO[song][1] + 1))
        a, b = rng.choice(MOODS, 2, replace=False)
        half = drum_groove.seconds(drum_groove.song_bars(song, s, bpm), bpm) / 2
        out.append({"seed": s, "bpm": bpm, "kit": kits[s % len(kits)],
                    "emotion": f"0:{a}" if rng.random() < 0.5 else f"0:{a},{half:.1f}:{b}",
                    "mocap": bool(s // len(kits) % 2), "song": song})
    return out


def run(t, out):
    song = t.get("song", "random")
    name = f"{song[0]}{t['seed']:03d}"
    mid, js = out / f"{name}.mid", out / f"{name}.json"
    bars = drum_groove.song_bars(song, t["seed"], t["bpm"])
    drum_groove.write(str(mid), drum_groove.notes(t["bpm"], t["seed"], bars=bars), t["bpm"],
                      drum_groove.meter_map(bars))
    cmd = [sys.executable, "-m", "blockout.drums", str(mid), str(js), "--kit", t["kit"], "--emotion", t["emotion"],
           "--seed", str(t["seed"]), "--teach", "--clips"] + (["--mocap", MOCAP[song]] if t["mocap"] else [])
    # one take is one core's work: numpy's own thread pool in every take would fight the others for the cores
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **ONE_THREAD})
    (out / f"{name}.log").write_text(r.stdout + r.stderr)
    res = {**t, "name": name, "cmd": " ".join(c if " " not in c else f'"{c}"' for c in cmd), "rc": r.returncode}
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            res.update(d if "hand_hits" in d else {"clip": d})
    res["hand_notes"] = sum(h["piece"] not in ("kick", "hihat_pedal")
                            for h in drums.hits_from_midi(midi.read(str(mid))))
    res["faults"] = faults(res)
    return res


def faults(r):
    if r["rc"] or "hand_hits" not in r:
        return ["drums.py failed"]
    f, c = [], r.get("clip", {})
    if r["contact_err_mm_max"] > CONTACT_MM:
        f.append(f"stick {r['contact_err_mm_max']} mm off a strike point")
    if r["tip_penetrating_frames"]:
        f.append(f"tip through a head on {r['tip_penetrating_frames']} frames")
    if r["kicks_beater_on_head"] < r["kicks"]:
        f.append(f"beater on the head for {r['kicks_beater_on_head']} of {r['kicks']} kicks")
    if r["max_reach_m"] > r["arm_length_m"]:
        f.append(f"reach {r['max_reach_m']} m")
    if c.get("frames_clipping"):
        f.append(f"clipping on {c['frames_clipping']} of {c['frames']} frames {json.dumps(c['pairs'])}")
    if r["hand_hits"] < r["hand_notes"]:
        f.append(f"{r['hand_notes'] - r['hand_hits']} of {r['hand_notes']} hand notes unplayed")
    return f


def line(r):
    head = f"{r['name']} {r['bpm']:3d} bpm {r['kit']:12s} {r['emotion']:22s} {'mocap' if r['mocap'] else '     '}"
    if "hand_hits" not in r:
        return f"{head} | FAIL drums.py failed (rc {r['rc']})"
    body = (f"contact {r['contact_err_mm_max']} mm, reach {r['max_reach_m']}, kicks {r['kicks_beater_on_head']}/"
            f"{r['kicks']}, hands {r['hand_hits']}/{r['hand_notes']}, clipping {r.get('clip', {}).get('frames_clipping')}"
            f"/{r.get('clip', {}).get('frames')}" + (f", skipped {r['skipped']}" if r.get("skipped") else ""))
    return f"{head} | {body} | " + ("; ".join(r["faults"]) or "ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="work/random")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--song", choices=TEMPO, default="random")
    ap.add_argument("--kits", default=",".join(KITS), help="kits that take turns, e.g. big,big,standard")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rs = []
        for r in ex.map(lambda t: run(t, out), takes(a.n, a.seed, a.song, a.kits.split(","))):
            print(line(r), flush=True)
            rs.append(r)
    bad = [r for r in rs if r["faults"]]
    for r in bad:
        print(f"{r['name']}: {r['cmd']}")
    print(f"{len(rs) - len(bad)} of {len(rs)} takes clean")
    json.dump(rs, open(out / "summary.json", "w"), indent=1)

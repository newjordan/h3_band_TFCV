# Piano handoff: fresh start on v18, 2026-10-16

## Goal
Graceful fingers playing the piano. Regular hand. The Comfy dev challenge is due **2026-10-19**.

## Start here
- **Repo / branch:** `~/h3b_v18`, branch `piano-v18-base`. It is a worktree of `~/h3b_wrist` (public remote newjordan/h3_band_TFCV). **Not pushed.**
  - `7a33f52`: ks-wrist head. Solver and renderer source are unchanged since the v18 solve.
  - `52e9659`: `tools/v18/`. Reproducible solve, render, compare, review metrics.
  - `2badbed`: the v18 settings are now the code defaults. No env overrides needed.
- **Hand:** the regular stock MPFB2 hand at scale 1.0 (the default here: `furelise_cal.json` has no hand_scale). **Never shrink, shorten or reshape the hand.** The user called the shrunk hand a mistake. This is going open source and people expect a normal hand.
- **Solve the whole suite:** `tools/v18/solve.sh OUT` (24 parallel parts, ~48 min on 20 cores, writes `OUT/suite.json` and `OUT/notes.json`).
- **Render:** `tools/v18/render.sh OUT`. Still keyboard edit, Salamander audio, `OUT/suite_grey.mp4`, a few minutes.
- **Score:** `python3 tools/v18/review_metrics.py OUT/suite.json` (sections + suite).
- **Diff two solves:** `python3 tools/v18/compare.py A.json B.json`.
- **Python:** `/home/frosty40/h3_band_TFCV/.venv_blockout/bin/python`. MIDI and feeling inputs: `~/h3b_suite/examples/piano/knights_suite.{mid,feeling.json}` (unchanged since Oct 9).
- **Experiments:** `PIANO_OVR='{"NAME": value, ...}'` still overrides any solver constant (`NAME_S` multiplies it).

## Proof the baseline is solid
- A fresh full-suite re-solve on `52e9659` equals the saved v18 take **byte for byte**: 8,415 frames, every field, and notes.json identical.
- After the defaults change (`2badbed`), the Rachmaninoff (100 s) and Chopin (250 s) test windows are byte-identical to the old code run with the v18 override.
- Saved takes (anim + notes + exact scripts): `~/h3/hands_loop/rebase_20261016/takes/{v17d,v18,v19}/`, plus `ks.wav`. The fresh re-solve is in `.../resolve_v18/`.

## v18 baseline numbers (beat these)
From `review_metrics.py` on the v18 take:

| window | keys in time | finger speed °/s | frozen joints | broken frames | pops/min | tips crossed | PIP curl |
|---|---|---|---|---|---|---|---|
| Debussy 39.9–79.9 s | 90.7% | 35 | 12.5% | 1.13% | 11.3 | **19.5%** | 26° |
| Rach 79.9–127 s | 92.0% | **93** | 4.4% | 1.01% | **82.8** | 15.8% | 38° |
| Chopin 127–390.5 s | 94.0% | 35 | 13.8% | 0.26% | 11.4 | 10.1% | 31° |
| **suite** | **93.2%** | 43 | 12.4% | 0.46% | 21.0 | 11.9% | 31° |

Times are score seconds (ks.wav). Suite video time = score − 39.9. The lab's own ±0.25 s key metric reads v18 as 95.5%; the table uses the stricter −1.5..+3 frame window.

## What's wrong with v18 (where to hit it)
1. **Rachmaninoff agitato is jittery:** 83 pops/min and finger speed 93 °/s, about 7× the rest of the suite. This is the biggest visible problem. v19's hard press mask made it worse (205 pops/min). More press weight is not the fix.
2. **Left hand swings off the keys at score 288–292 s** (suite video 4:08–4:12; the same in v19). The animation is smooth there (`lab/probe_glitch_248.py`). Prime suspect is the renderer: the forearm `Seg.set` drawn 5 cm short for mpfb, or the `pose_mpfb` lowerarm aim at extreme wrist yaw. Next test: render the window `--hands capsule` vs `mpfb` from the same JSON. See `HANDOFF_20261010_chain_glitch.md`.
3. **Crossed fingertips:** 19.5% of hand-frames in Debussy, 11.9% suite. The Oct 11 hand-order pass (rig_realism commit `1cc214d`) cut this to 2.9% in its minute, but hooked tips rose to 3.4%. Port it as an option and A/B it on the review.
4. **The thumb sticks straight out along the keys** in the close-ups (every take, v17d onward). It reads stiff even when the other fingers curl.
5. **Renderer weight typo:** one right-hand finger vertex follows the left hand. rig_realism fixed it with `clean_hand_weights` in `blockout/skin.py`. Cheap port.

## How to work (the user's calls)
- **Look at footage and data, every change.** Review harness (run it from its own folder): `~/h3/hands_loop/rebase_20261016/` renders any take on one fixed camera at matched windows, side by side with audio:
  - `jobs.py` (cameras and windows; extend `V`), `mkgrid.sh WIN CAM START DUR COLS OUT names...`, `metrics.py`.
  - A take renders through the renderer of its own era: old long-finger takes with `trees/wrist`, current takes with the branch.
- **Results go on the band site** (https://toymaker.tail2d8af4.ts.net/band/, section `rebase-review`). Rebuild with `python3 ~/h3/mv_tools/band_site.py`. The user is on atlas.
- **One change at a time against v18,** judged by eye and on these numbers. Keep a change only if the fingers look better and nothing in the table regresses badly.
- **Stay out of local proof loops.** Oct 11–16 in `~/h3/rig_realism` went into static repairs and 43 ms certificates and the playing got stiff (finger speed halved, frozen joints ~3×, keys 41%). Archive only: its branch and ~700 uncommitted files are left untouched. Cherry-pick ideas, not the tree.
- **Standing rules:** the user decides on rigging and hands. Strike near the key front with the fingertip. Wrist-led; idle fingers float and glide. No camera moves (still wide shots of keys and hands).
- **Commits:** `newjordan <11369410+newjordan@users.noreply.github.com>`, no Co-Authored-By or Claude-Session lines. Push only when asked.

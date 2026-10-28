# HANDOFF 2026-10-10 ~22:50: chain E37–E14 done; 248–252 s glitch narrowed to the renderer; angelX pinned to gpt-6-sol

## Stop state (read first)
- **The paused lab chain is finished.** All five queued experiments ran and are in `lab/results.csv` and `lab/runs/`: E37, E35, E34, E12, E14. **Nothing is running now.** No scheduled wakeups. GPU idle.
- The chain was run via **systemd user units** (`lab-chain2`, then `lab-chain-rest`). Plain `nohup ... &` from the sandbox dies when the tool call exits — use `systemd-run --user --unit=NAME --working-directory=DIR CMD` for anything that must survive the turn.
- **angelX launcher updated** (`~/.local/bin/angelX`): now exports `ANGEL_OPENAI_MODEL=gpt-6-sol` + `ANGEL_OPENAI_REASONING_EFFORT=max` (env-overridable), so the Codex seat serves **Sol 6**. The catalog (`~/.codex/models_cache.json`) has `gpt-6-sol` with efforts low..ultra; there is **no "6.1" slug** — gpt-6-sol is the newest Sol. Selection order: env > `~/.codex/config.toml` (`model = "gpt-6-astra"`) > built-in `gpt-5.6-luna`. Takes effect on the next angelX launch. `bash -n` verified.
- **Git:** ks-wrist at `8ec6cb3` (notebook + E37 entry). **Not pushed** — push on next commit (newjordan identity, no trailers).

## Results this session (whole-suite, 1,918 notes; metric = key >=50% down within 0.25 s)

| run | change | keysync | Debussy | Rach | Chopin | jitter mm | tremor | verdict |
|---|---|---|---|---|---|---|---|---|
| E29 | stack (parent) | 95.46 | 92.9 | 94.4 | 96.8 | 1.55 | 0.35 | |
| **E36** | E29 + hard mask [20,1,20] | **96.98** | **100.0** | 96.5 | 97.1 | 1.76 | 0.67 | **still best** |
| E37 | E29 + middle mask [8,1,8] | 96.56 | 93.9 | 96.3 | 97.1 | 1.70 | 0.52 | keeps ~97% of the gain, **no tremor relief** — E36 stays the pick |
| E35 | E29 + OWN_KEY_GIVE 3 mm | 95.2 | 96.9 | 93.8 | 96.3 | 1.57 | 0.36 | neutral-negative; drop |
| E34 | E29 with SPLIT_WIDE off | 95.57 | 92.9 | 94.5 | 96.9 | 1.52 | 0.34 | +0.1 vs E29, within noise; SPLIT_WIDE stays on |
| E12 | E11 + wrist smoothing lambda 2 | 68.87 | 73.5 | 67.1 | 70.1 | 4.21 | 0.35 | ragdoll leg: park it |
| E14 | E12 + wrist joint term 0.015 | 71.32 | 70.4 | 62.3 | 80.1 | 3.63 | 0.30 | ragdoll leg: helps (+2.5) but far behind; parked |

Also in `lab/NOTEBOOK.md`. **100% is now 58 notes away (E36).**

## The 248–252 s glitch (v18/v19 cut): what it is NOT
- **NOT `_close_roles`** (the previous guess). Instrumented window 247–253 s of the performed timeline: hand-centre gap never below +0.20 m, `near` = 0 everywhere, zero role swaps — its output is inert here.
- **NOT animation discontinuity.** `lab/probe_glitch_248.py`: max per-frame hand-root rotation 8.4 deg (L) / 3.7 (R), no wrist jump > 50 mm.
- **NOT IK overstretch.** Upper/forearm bone lengths exact to 1e-16 every frame.
- **What remains (next session, look here first): the render side.** Frame-by-frame read of the v19 cut (248–251.2 s) shows: L hand off the front edge of the keys fully splayed (~250.4 s), open-ended **tube segments with gaps** on the forearms, arms reading as crossed. Suspects in order:
  1. `blender_piano.py` forearm `Seg.set` (~line 468): for `HANDS=mpfb` the seg is drawn 5 cm short (`wr - 0.05`) to swallow the hand's forearm stub — an extreme wrist yaw exposes the cut edge. Also check roll/normals flip.
  2. `pose_mpfb` lowerarm01/02 aim (`blockout/blender_piano.py:286+`).
  3. The L-hand wander is real in the animation (xh jumps -0.154 -> -0.024, pad span to 195 mm) — legal but ugly; render the same frames JSON with `--hands capsule` vs `mpfb` to separate rig from renderer.
- Cut video copied to `~/h3/hands_loop/glitch_20261011/suite_grey_v19.mp4` (internal study only).
- Also: E36 `w_249.9` has **anatomy_out = 286** with only 3 key misses — this glitch class is invisible to the keysync metric. Port an anatomy/overlap QA into the hands-loop ledger as the next column.

## Standing rules (unchanged)
- Band site rebuild: `python3 ~/h3/mv_tools/band_site.py`. Line/dot plots only.
- No bulk H3 renders; single shots only when asked. GPU lease `h3-K`.
- Commits: `newjordan <11369410+newjordan@users.noreply.github.com>`, no Co-Authored-By, push for the drums work.
- Lab runs: `systemd-run --user --unit=lab-<ID> --working-directory=/home/frosty40/h3b_wrist /home/frosty40/h3_band_TFCV/.venv_blockout/bin/python lab/run.py <ID>`, or chain via `lab/queue.sh`. Pausing: kill run.py + eval_suite children + `ssh sparky|turbo pkill -f lab_<ID>_w_`, then delete `lab/runs/<ID>`.

## Next ideas (by expected value)
1. Renderer diff for the glitch (capsule vs mpfb on the 249.9 window frames) -> fix Seg/mpfb forearm -> re-render the Chopin window -> re-cut v20.
2. Sideways misses (41 left in E36): `lab/lit/sideways_diagnosis.md` already names the fixes — a2 octave-span vs HOME layout (13), a1 impossible chords (Rach 120.55/128.5 need a musical roll decision), c reassignment (4).
3. Tremor control on E36 (stronger FINGER_REFINE prior on press frames) to buy back the x1.9 tremor.
4. Ragdoll leg parked; deadline work first.
- Deadline: **Comfy Developer Platform Challenge, Mon 2026-10-19 9am PT.**

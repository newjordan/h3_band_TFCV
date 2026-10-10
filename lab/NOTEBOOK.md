# Piano-hand lab notebook

**Question.** Can a rigged pianist's hand, driven only by a score (MIDI), play every note of a real piano suite
(Clair de Lune -> Rachmaninoff Op. 3/2 bars 35-61 -> Chopin Nocturne in C# minor, complete; 350 s, ~1,900 notes)
with every key going down under the finger that plays it (100% key sync), while moving like a pianist: straight-ish
wrists, no shake, no finger tremor, joints inside human range, a real-size hand?

**Why it matters.** The animation is the motion plate for MiniMax-H3 (ControlNet + plate lock): whatever the plate's
fingers do, the rendered knight does. A missed key or a twitching finger is visible in the final film.

## Metrics (blockout/eval_suite.py, lab/run.py)

| metric | definition |
|---|---|
| key sync | % of notes whose key goes >= 50% down within 0.25 s of its onset (key depth follows the solved fingertip pad) |
| miss kinds | *unfingered*: no finger assigned (e.g. > 5 notes for one hand); *sideways*: pad off the key across it; *short*: pad off the key along it (in front of a black key); *shallow*: on the key, pressed < 50% |
| wrist deviation | angle between the forearm (elbow -> wrist) and the hand (wrist -> knuckle centre), median and p95, degrees |
| wrist shake | rms of the wrist path minus its Savitzky-Golay (9 frames) smoothing, mm |
| finger tremor | joint reversals > 2 deg on consecutive frames, per joint-second |
| anatomy | joint-frames outside the anatomical range table, per minute |

Protocol: the suite is cut into 10 s windows (39.9-390.5 s), each solved independently with 2 s of pre-roll, notes
counted by onset in [start, start+10); windows run in parallel on toymaker (12 slots), sparky (18), turbo (28).
Every experiment = one row in `results.csv`, its windows in `runs/<id>/`, its definition in `experiments.json`.

## Hypothesis legs

- **H0 baseline**: the current rig, oversized MPFB hand (24.3 cm).
- **H1 ragdoll-first**: loosen every motion constraint until key sync is ~100% (a ragdoll wrist that goes wherever the
  fingers need it), then *reverse into it*: re-introduce smoothness, the wrist joint, the hand size, one at a time,
  keeping 100%. Finds the frontier between accuracy and natural motion and which constraint costs what.
- **H2 data-driven hand**: a real-size hand (FürElise median 17.7 cm) driven by models learned from 15 concert pianists
  (wrist placement per finger/black key, ballistic travel, later fingering).
- **H3 constraint ablations**: remove one component at a time from the best rig to measure its contribution.

## Pre-lab history (2026-10-09, 10 s test windows, not whole-suite; for context)

| step | Rach bars 35-61 | Rach agitato | wrist dev median | note |
|---|---|---|---|---|
| v16 (whole-hand per-frame solve) | 98.4 | 82.8 | 17-32 deg | wrist shake 7.6-21 mm, finger stalls ~400 per 10 s |
| v17 wrist-led (smoothed wrist, split finger solve) | 96.7 | 81.0 | 3-6 | shake 1-2 mm, stalls < 10 |
| v17C (FürElise calibration, ragdoll ease, torso lean/turn) | 95.1 | 75.9 | 31-47 | elbow lift bent the wrist |
| elbow aligned to hand, capsule torso | 80.3 | 71.6 | 15-16 | |
| torso hitbox fix (shoulders were inside it) | 88.5 | 73.3 | 7-8 | big hand |
| same, real-size hand + FürElise wrist model | 82.0 | 64.7 | 8-13 | |

Calibration source: FürElise (Wang et al., SIGGRAPH Asia 2024; CC BY-NC 4.0), 151 pieces / 15 pianists, 420k
hand-frames, 329k fingered strikes; held out: Clair de Lune (81), Chopin Nocturne Op. 27/2 (32).

## Experiments

(Each entry: hypothesis -> change -> result -> conclusion. Numbers are whole-suite unless stated.)

### E00 (H0 baseline): 81.9% whole suite
Big hand, all fixes as of 23:00, strike bands 12 mm toward the key fronts. Debussy 74.5, Rach 74.4, Chopin 90.7.
Misses 330: shallow 206 (finger on the key, pressed < 50%), sideways 99, unfingered 24 (> 5 notes per hand), short 1.
Wrist deviation 11.9 deg median, shake 1.6 mm, tremor 0.34. The whole-suite Debussy is much harder than the one
10 s window used during development (100% there). **Conclusion:** shallow presses are the largest error class.

### E01 (H2): real-size hand: 75.0%
Debussy 90.8 (+16 vs E00), Rach 57.8 (-17), Chopin 89.8. Sideways misses 99 -> 175, shallow 206 -> 260.
**Conclusion:** a pianist-sized hand helps the lyrical music and hurts the octave/chord-heavy Rachmaninoff: the
planner/fingering does not yet use a small hand the way pianists do (they reach octaves; ours land 13-21 mm short).

### E10 (H1): ragdoll ceiling: 76.4% -- hypothesis refuted
Every motion constraint loosened (no wrist smoothing, no wrist joint term, 10 cm ease on a weak spring, no finger
smoothing). Sideways misses 175 -> 291, shake 9.8 mm, tremor 1.67 (5-7x). **Conclusion:** the misses are not caused
by the motion constraints holding the hand back; loosening makes the hand wander off its keys. The error sits in
where the fingers are aimed (targets, fingering) and in the per-finger solve. The H1 "reverse" ladder (E11-E15) is
still run to map the frontier, but the route to 100% is the targets/fingering (H4) and the real-size reach (H2).

### Knuckle depth (side test, 3 windows, not whole-suite)
Our knuckles sat 3.8-4.7 cm into the keys vs the pianists' 0.7 cm (FürElise playing median): the wrist model's
depth offsets were not scaled to the hand's size. Scaled: Chopin window knuckles 4.3 -> 0.8 cm; key sync per window
-4 / +2 / -12 (2 of 16 notes) points. Whole-suite measurement: E04.

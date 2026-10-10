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

### Protocol incident (00:16-00:55): E02 and E20 discarded
Two queue chains ran concurrently and a code change (fingering every note) landed mid-run: local windows used the
new code, remote windows the old. Both runs were stopped and deleted. Rule from here: one queue chain; code changes
only between runs (remotes sync at run start, local windows import per window).

### Code changes before E05
- Every note fingered: `playable()` used to drop notes of chords wider than a hand or with more than five keys
  (they sounded in the audio, no finger pressed them); now such chords are played as rolled sub-chords in onset
  order, doubled keys (both staves on one key) are struck once, and a chord too wide for any block fingering still
  gets the least-bad stretch. All 2,046 suite notes now have a finger (was 2,021).
- Wrist model depth offsets scale with the hand (knuckles ~0.7 cm into the keys like the pianists).

## Leg K: H3 internals (making the render match the Blender plate)

**Problem.** Structure lock (Fun ControlNet Union 2.0 on plate edges + plate in the video stream) fixed the camera
and the hand placement, but the look went generic and the hands are not the Blender hands. Effort per frame is not
the lever: 60 steps vs 20 (same seed) and a 0.5-strength refinement pass changed nothing measurable.

**Metric (h3_tools/motionmatch.py).** Blender renders an exact hand mask (`--pass mask`); inside it: edge-structure
correlation plate vs render (hand_r) and dense optical-flow agreement (flow_cos, flow_err). The earlier whole-frame
plate correlation (0.7+) was carried by the keyboard: inside the hands the renders score hand_r 0.05-0.27.

| render (shot kh111, 124 f, 1280x704) | hand_r | flow_cos | flow_err |
|---|---|---|---|
| r1: fl2va + canny + plate 0.88 (candy-cane sleeves) | 0.27 | 0.82 | 0.69 |
| jazz3: + chrome/jazz prompt | 0.09 | 0.75 | 0.74 |
| q60: jazz3 at 60 steps | 0.09 | 0.76 | 0.73 |
| refine: q60 re-run at strength 0.5 | 0.05 | 0.73 | 0.76 |
| r2/r3: ref2va bass-knight reference (+ plate) | 0.09 / 0.17 | 0.62 / 0.67 | 0.85 / 0.79 |

**Hypotheses.** K-a: the control lacks finger information (grey fingers on grey: few edges inside the hand) ->
controls that resolve each finger (normal-shaded pass edges, depth). K-b: appearance and geometry live in different
heads; a KV pull toward the reference look on the appearance heads gives the look without moving the geometry
(custom_nodes ks_h3_attn.py: KSH3AttnProbe, KSH3RefPull). K-c: plates in target materials need less strength.

### E05 (H0, new baseline): 82.2% (+0.3 vs E00)
Every note fingered + knuckle-depth scaling, big hand. Unfingered 24 -> 0, but those notes now mostly miss as
sideways (99 -> 113) or shallow (206 -> 226): fingering a rolled tenth does not by itself put a finger on its key in
time. Debussy 74.5, Rach 74.2, Chopin 90.6; wrist dev 11.5 deg, shake 1.5 mm, tremor 0.31. **Conclusion:** the
structural class is gone; the remaining errors are contact (shallow) and reach (sideways), the targets of H4 and H2.

### K1 (K-a): control from the normal-shaded pass: hand_r 0.09 -> 0.42, look lost
Same prompt/seed/plate as q60; the canny control is taken from Blender's normal-shaded pass (every finger has its own
shading, so its own edges) instead of the grey beauty pass. Inside the hand mask: hand_r 0.42 (q60 0.09, best before
0.27), flow_cos 0.85 (0.76), flow_err 0.67 (0.73); whole frame r 0.83, no shifts. But the gauntlets became bare human
hands (lab/plots/k1cmp.jpg: plate | q60 | K1). **Conclusion:** K-a supported: the plate geometry reaches the render only
when the control resolves the fingers. The dense per-finger edges now pin a *hand*, and the model fills it with its
prior (skin); the chrome look must be pulled in separately (K-b: KV pull toward a chrome reference on the
appearance heads, K-c: chrome-material plates) while keeping this control.

### E06 (H2): real-size hand on the E05 code: 74.6% (E05 82.2)
Debussy 91.8 (+17.3 vs E05), Rach 58.1 (-16.1), Chopin 88.6 (-2.0); sideways 201, shallow 271, short 15. Wrist dev
10.6 deg, shake 1.5 mm, tremor 0.26 (smoother than the big hand). **Conclusion:** replicates E01 on the new code: the
small hand is the better pianist in the lyrical music and fails the chords/octaves. The Rachmaninoff losses are reach
(sideways) and contact (shallow) under stretch; H2 needs reach (a wider span model / wrist that rotates into octaves)
before it can pass the big hand. Note: E06's local windows ran after the default-off PRESS_OVERSHOOT/REPAIR_ITERS
switches landed (identical code path at their defaults; checked).

### K2 (K-a): depth-pass control: chrome kept, hands lost (hand_r -0.01)
Depth control instead of edges: the gauntlets stay chrome (lab/plots/k2cmp.jpg: plate | K1 | K2) but the hands do not
follow the plate (hand_r -0.01, flow_cos 0.66, frame r 0.65, first second r 0.49). **Conclusion:** depth is too coarse to
pin fingers and leaves the model free (look kept); dense finger edges pin the hand and take the look. Geometry and
appearance trade against each other through the control; K6 (normal edges 0.6 + depth) and K7 (normal edges + chrome
references) test whether both can be held at once.

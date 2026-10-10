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

### Pilot (one window, Rach 119.9-129.9 s, big hand, E05 code): contact switches
| variant | key sync | sideways | shallow | short | finger tremor (L,R) |
|---|---|---|---|---|---|
| E05 settings | 120/146 (82.2%) | 18 | 8 | 0 | 0.99, 1.13 |
| REPAIR_ITERS 2 | 129/146 (88.4%) | 12 | 4 | 1 | 1.28, 1.25 |
| PRESS_OVERSHOOT 3 mm | 125/146 (85.6%) | 16 | 4 | 1 | 1.00, 1.11 |
| hard press mask [20,1,20] | 128/146 (87.7%) | 14 | 0 | 4 | 2.13, 1.97 |
Shallow misses in E05 are mostly late arrivals (half have the key not moving at all, pad a median 2 mm above it), so
repair (boost the finger's press around the onset) fits the failure. The hard mask fixes depth but doubles tremor.
Whole-suite runs queued ahead of the ablation ladder: E25 (repair + overshoot), E24, E23, E22.

### K3: attention probe (ref2va, bass-knight reference, canny + plate; 13 blocks x 56 heads x steps 0/4/8/12)
Video queries put on average 0.3-5% of their attention on the reference image, but a few heads specialise:
block 12 head 8 0.79 (stable over steps), block 8 head 32 0.40, block 4 head 13 0.49 -> 0.13 (early steps only), block
16 head 55 0.46 -> 0.23, block 24 head 35 0.35. Reference reading lives in blocks 4-24; from block 28 on it is < 0.13
everywhere, while text reading peaks late (block 44: 0.83 on one head). lab/plots/k3_probe.png, raw lab/k3_probe.json.
**Conclusion (K-b):** appearance transfer from the reference runs through ~25 identifiable heads in the first half of
the network, so a pull can be targeted at them only (rather than all heads, which would also drag the geometry).
Queued: K8/K9 = K7 (chrome refs + normal-pass control) + KV pull x4 / x8 on the heads with reference mass >= 0.12 in
blocks 4-36 (24 heads).

### E20 (H4 contact): press height weighted like position: 87.5% (+5.3 vs E05)
Hypothesis: shallow misses come from the press target weighting height (1.5) half as much as across-key position (3).
Change: PRESS_AXES [3,1,1.5] -> [3,1,3] (black [3,3,3]). Result: Debussy 84.7 (+10.2), Rach 82.0 (+7.8), Chopin 93.0
(+2.4); shallow 226 -> 127, sideways 113 -> 107, short 6; wrist dev 11.6 deg, shake 1.56 mm, tremor 0.33 (E05 0.31).
**Conclusion:** supported, and nearly free in motion quality: the finger solve was trading depth for comfort.
(Clean re-run of the E20 discarded in the protocol incident.) E23/E24 re-based on E20 before they started; E25 (repair +
overshoot) was already running on E05 weights and stays as defined.

### K4 (lit review: control weight 0.8): hand_r 0.07
Beauty-pass canny at strength 0.8 instead of 1.0 (q60 settings otherwise): hand_r 0.07 (q60 0.09), flow_cos 0.69, frame
r 0.71. **Conclusion:** lowering the weight of a control that carries no finger detail loosens everything and gains
nothing; the lever is what the control contains (K1), not its weight.

### E25 (H4 contact): repair loop + 3 mm overshoot (E05 weights): 92.3% (+10.2 vs E05)
Hypothesis: shallow misses are late arrivals and stops at the key bed; verify-and-repair (re-solve the fingers with a
missed note's press x4 around its onset, 2 passes) plus aiming 3 mm below the bed recovers them. Result: Debussy 82.7,
Rach 90.5 (+16.3), Chopin 95.2; shallow 226 -> 65, sideways 113 -> 77; wrist dev 11.5 deg, shake 1.55 mm, tremor 0.35
(+0.04); 33 min vs 29. **Conclusion:** supported; the largest single gain so far, at almost no motion cost. Queued
next (after E24 = E20 + repair): E26 = E20 weights + repair + overshoot (stack), E27 = 4 repair passes, E28 = E26
with the real-size hand; then the remaining contact singles and the ablation ladder.

### K5 (K-c/lit review: role refs): chrome references + beauty canny: chrome yes, hands no (hand_r 0.07)
ref2va with three chrome-gauntlet reference images at ref_image_size=max, beauty-pass canny, plate 0.88. The gauntlets
are chrome (lab/plots/k5cmp.jpg: plate | K1 | K5) but the hands do not follow the plate (hand_r 0.07, flow_cos 0.72,
frame r 0.69; one reframe at f6). **Conclusion:** the references deliver the look, the beauty canny cannot deliver the
fingers: same split as K2. K7 (these references + the normal-pass control of K1) is the direct test of both at once.

### K6 (K-a): normal-pass canny 0.6 + depth (two ControlNet patches): hand_r 0.41, still skin
hand_r 0.41, flow_cos 0.80, frame r 0.83: geometry as good as K1 at 60% edge weight, but the hands are still bare skin
(lab/plots/k6cmp.jpg: plate | K1 | K6). **Conclusion:** depth adds nothing to the look; the per-finger edges alone decide
both the match and the "human hand" reading. The look has to come from the reference side (K7-K9).

### K7: chrome references + normal-pass control: identical to K5 -> ref2va ignores the ControlNet
K7 differs from K5 only in the control video (normal-pass vs beauty canny) and scores the same to two decimals
(hand_r 0.07, frame r 0.69, the same worst frame). The two videos differ by 1.4/255 mean luma (max 7). The Fun
ControlNet patch is wired correctly for the ref2va layout (control rows go to the target video tokens, reference rows
get zero), so with the ref2va checkpoint the control's residual is effectively drowned out: **in ref2va mode there is no
structure lock at all**; every ref2va hand result so far (r2, r3, K3, K5, K7) was the model's own hands.
**Consequence:** K8/K9 (KV pull on ref2va) cancelled (K8 interrupted on the GPU). The look must come in through the
fl2va (t2v/i2v) model, where the control works (K1: hand_r 0.42):
- K-c "pixel setup": Blender renders the plate in the target materials (`--pass chrome`: Cycles, mirror-chrome hands
  and forearms, glossy black piano, warm low light; lab/plots/kh111_chrome_f0.png), same camera and poses as the plate
  and control.
- K10 = K1 with the chrome plate in the video stream; K11 = K10 + the chrome frame 0 pinned as first frame;
  K12 = K11 with the attention probe (which heads read the pinned frame -> targets for a KV pull in fl2va).

### E24 (H4 contact): E20 weights + repair loop: 93.2% (new best)
Debussy 88.8, Rach 92.3, Chopin 94.6; shallow 41 (E20 127), sideways 83, short 6; wrist dev 11.6 deg, shake 1.56 mm,
tremor 0.37 (E20 0.33). 60 min (the toymaker share ran beside the chrome-plate Cycles render). **Conclusion:** repair
is the main lever (E20 -> E24 +5.7); with shallow nearly solved, **sideways (83) is now the largest class**: the pad is
beside its key, which repair (a stronger pull on the same target) cannot fix when the target or the reach is wrong.
Next (after E26-E28): a sideways study -- per-miss geometry (finger, interval to the neighbouring notes, hand span at
the onset) to separate fingering errors from reach limits.

### K10 (K-c pixel setup): K1 + the chrome-material plate: chrome gauntlets, hands drift (hand_r 0.09)
Same as K1 except the plate in the video stream is Blender's chrome pass. The render is armoured chrome gauntlets
(lab/plots/k10cmp.jpg: chrome plate | K1 | K10) but hand_r falls 0.42 -> 0.09 (flow_cos 0.81, frame r 0.75): with the
same control, the plate's pixels decide whether H3 reads "hand" (skin, follows the edges) or "gauntlet" (its own
armour prior, shape not held). **Conclusion:** the plate at 0.8% weight sets the *category*, the control sets the
*shape* only when the category is a hand. Next: K11 (chrome first frame pinned), K12 (probe: which heads read it).

### Sideways study, part 1 (E24): what the 83 sideways misses are
66/83 on black keys; |dx| median 35 mm (25% under 20 mm, 30 over 60 mm); Chopin 47, Rach 30, Debussy 6; the error points
inward (R pinky lands toward the thumb, R index/thumb toward the pinky): the fingers fall short of the span the
fingering asks for, often in fast wide figures (e.g. 190.22 s R4 on G#5 then 190.27 s R1 on D#4: 17 semitones in 50 ms).
Pilot (2 Chopin windows): easing the wrist smoothing for leaps (LEAP_V 1.2 m/s) gains 1 note of 71 and adds shake
(+0.1-0.8 mm): not the lever. A per-miss diagnosis (impossible as fingered / fingering choice / hand assignment /
held-note conflict / black-key aim) is running as an agent; report -> lab/lit/sideways_diagnosis.md.

### K11: K10 + the chrome frame 0 pinned as first frame: chrome and half the geometry back (hand_r 0.18)
hand_r 0.09 -> 0.18, flow_cos 0.83, flow_err 0.67 (= K1), frame r 0.72; warm amber cafe light. The hands read as
liquid-chrome hands shaped like the plate's (lab/plots/k11cmp.jpg: chrome plate | K10 | K11) rather than K10's
armour of its own shape. **Conclusion:** a pinned frame in the target look is the first input that moves both axes at
once (look kept, geometry up). Still well under K1's 0.42; next, holding the chrome plate harder (K13 strength 0.70,
K14 lockstep anchors) and the K12 probe for the heads that read the pinned frame.

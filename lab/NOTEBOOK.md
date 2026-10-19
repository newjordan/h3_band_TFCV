# Piano-hand lab notebook

**Question.** Can a rigged pianist's hand, driven only by a score (MIDI), play every note of a real piano suite
(Clair de Lune -> Rachmaninoff Op. 3/2 bars 35-61 -> Chopin Nocturne in C# minor, complete; 350 s, ~1,900 notes)
with every key going down under the finger that plays it (100% key sync), while moving like a pianist: straight-ish
wrists, no shake, no finger tremor, joints inside human range, a real-size hand?

**Why it matters.** The animation is the motion plate for MiniMax-H3 (ControlNet + plate lock): whatever the plate's
fingers do, the rendered knight does. A missed key or a twitching finger is visible in the final film.

## Overnight summary (2026-10-10, 00:30-08:15)

**Hands (whole suite, 1,918 notes, key at least half down within 0.25 s):** 82.2% (E05) -> **95.5% (E29)**, motion
unchanged (wrist deviation ~11.5 deg, shake ~1.55 mm, finger tremor 0.31 -> 0.35).
- Contact (H4): press height weighted like position (E20 87.5), a verify-and-repair finger pass (E24 93.2), 3 mm
  press overshoot (E23 93.0); together E26 94.7. Shallow presses 226 -> 25.
- Reach (H5): an over-wide chord gives its inner note to the other hand when it can take it (E29 95.5). Roll
  staggering adds nothing (E31). Remaining: ~57 sideways (four-octave chords, fast leaps), 25 shallow, 5 short.
- Real-size hand (H2): 74.6 -> 90.5% (E30) with the same fixes; Debussy at **100%**; its gap is Rachmaninoff octaves
  (84%) -> E32/E33 (anatomical thumb/pinky spread, open-hand planning) running.
- Ragdoll-first (H1) refuted early (E10): loosening constraints made the hand wander off its keys.

**H3 (leg K, single shot kh111 + one check on ov125):** the render follows the Blender hands only when the ControlNet
input carries per-finger geometry. Chrome plates, pinned frames, plate strength, lockstep anchors and attention (KV)
pulls on the 32 identified reference-reading heads all left the hands drifting (hand_r 0.12-0.19). ref2va ignores the
ControlNet entirely. **Recipe that works (K22, confirmed on ov125 as K23):** grey plate + gauntlet/vambrace geometry
added to the Blender hand (joint rings, cuff, forearm bands) + canny of its normal pass as the control + the
chrome-knight prompt: steel gauntlets that move with the plate (motion agreement vs the armour geometry 0.95 / 0.92).
Videos: band site, lab section.

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

### E26 (H4 contact): E20 weights + repair + overshoot: 94.7% (new best)
Debussy 92.9, Rach 94.3, Chopin 95.4; misses 101: sideways 71, shallow 24, short 6; wrist dev 11.4 deg, shake 1.56
mm, tremor 0.35. **Conclusion:** the three contact changes stack (E05 82.2 -> E20 87.5 -> E24 93.2 -> E26 94.7) with
no motion cost; shallow is nearly gone (226 -> 24). The remaining 5% is 70% sideways: reach/fingering, not contact.
E27 (4 repair passes) and E28 (real-size hand) run next; the sideways diagnosis decides the following leg.

### K12: probe of the fl2va model with the pinned chrome frame
The heads that read the pinned first frame ("cond") sit where ref2va's reference heads sit: blocks 4-24 (block 4 h4
0.38, h12 0.24, h13 0.21; block 8 h2/h44/h35/h46 ~0.2; block 24 h27/h35 0.2), several the same head indices as in
ref2va (shared base model). Text reading is late again (block 16 h17 0.92, block 44 h38 0.90). lab/plots/k12_probe.png.
Queued: K15 = K11 + KV pull x4 toward the pinned frame on the 32 heads with cond mass >= 0.10; K16 = x6 only on
steps 6-19 (the early steps that set geometry left alone).

### K13: chrome plate held harder (video strength 0.88 -> 0.70): hand_r 0.19, flow_cos 0.85
Articulated chrome gauntlets (plates, knuckle joints) shaped like the plate's hands; hand_r 0.19 (K10 0.09, K11 0.18),
flow_cos 0.85 (= K1), frame r 0.73. But the plate's environment now comes through: the Blender chrome pass lit a beige
studio, not a dark cafe (lab/plots/k13cmp.jpg: chrome plate | K11 | K13). **Conclusion:** plate strength is a usable
geometry lever once the plate is in the target look; the plate must then be the *whole* target look (dark smoky
room, amber practicals), not only the hands. Next: darker cafe environment in `--pass chrome`, then K13 + K11's pinned
frame.

### K14: K10 + lockstep anchors (every 3rd latent frame held at 0.6, others 0.88): hand_r 0.15
Articulated gauntlets, frame r 0.77 (best of the chrome runs), hand_r 0.15, flow_cos 0.83 (lab/plots/k14cmp.jpg:
chrome plate | K13 | K14). **Conclusion:** anchors hold the scene (frame r) better than a uniform 0.70 but the hands
slightly less (0.15 vs 0.19); uniform plate strength is the better hand lever. Both inherit the bright plate backdrop:
the v2 dark-cafe plate (K17/K18) is the fix.

### E27 (H4 contact): E26 with 4 repair passes: 94.9% (+0.2), 50 min (+50%)
Debussy 92.9, Rach 94.7, Chopin 95.4; sideways 70, shallow 22, short 5; motion unchanged. **Conclusion:** repair
saturates at 2 passes; the remainder is not a contact problem. E26's 2 passes stay the default for the stack.

### Sideways study, part 2: agent diagnosis (lab/lit/sideways_diagnosis.md) and a spread pilot
Of E24's 83 sideways misses (rules in order): 32 physically impossible as planned (sub-chords of an over-wide chord
start at the same instant, or leaps > 1.5 m/s; 14 of them with the other hand within reach), 13 octave chords that the
rig's finger layout (HOME spread 120 mm vs 165 mm octave) cannot open to, 9 fingering choice, 4 hand assignment, 4
held key pinning the hand, 13 black-key aim (mostly octave voicings again), 8 unexplained (likely hand-hand avoidance).
New switches (branch ks-wrist-dev, default off): ROLL_STAGGER (a rolled sub-chord starts when the hand can get there,
travel / 1.5 m/s, at most 0.20 s late; note copies, scoring keeps the score's onsets) and REACH_SPLIT (an over-wide
chord gives its inner-edge note to the other hand when that hand can take it; moves 11 suite notes, several of them
known misses).
Spread pilot (4 Rach windows, E26 settings, RIG_LIMITS pinky 35 deg / thumb CMC 45 deg vs 25/30): 408 -> 414 of 432
(94.4 -> 95.8%), sideways 18 -> 14, but joint-frames outside the QA anatomy table 2663 -> 4177 (+57%; the table caps
finger 2-5 abduction at 25 deg). **Conclusion:** reach helps but the pinky part is not free; next pilot opens the thumb
only (not in the 2-5 check; CMC abduction ~45 deg is anatomical) plus pinky 30.

### K15: K11 + KV pull x4 toward the pinned chrome frame on its 32 reading heads: no effect (hand_r 0.17)
Visually and by every metric K15 = K11 (hand_r 0.17 vs 0.18, flow_cos 0.82, frame r 0.72; lab/plots/k15cmp.jpg).
**Conclusion:** the pinned frame already delivers its look in full through those heads; boosting them changes nothing,
neither the look nor the geometry. The binding constraint is the plate (category + backdrop), not how strongly the
model reads its conditioning. K16 (x6, late steps) will confirm; the v2 dark-cafe plate (K17/K18) is the main bet.

### K16: K11 + KV pull x6 on steps 6-19 only: confirms K15 (hand_r 0.19, mean pixel change 3.3/255 vs K11)
**Conclusion (leg K-b closed for now):** attention re-weighting toward conditioning is not the lever on this shot:
the model already uses the conditioning it gets; what it gets (plate content, control content) decides the result.

### E28 (H2): contact stack (E26) on the real-size hand: 89.7% (E06 74.6, +15.1); Debussy 100%
Debussy **100.0** (first section at 100%), Rach 84.2, Chopin 93.9; sideways 152, shallow 28, short 18; wrist dev
10.4 deg (big hand 11.4), shake 1.55 mm, tremor 0.33. **Conclusion:** the contact fixes lift the pianist-sized hand by
15 points and make it the better hand for lyrical music; it now fails only on reach (sideways is 76% of its misses,
concentrated in the Rachmaninoff chords/octaves): exactly the classes the sideways switches target (REACH_SPLIT,
ROLL_STAGGER, thumb-led spread). The real-size hand is the one to take to 100%.

### K17: dark-cafe chrome plate (v2) at 0.70 + its frame 0 pinned: the plate's look, hand_r 0.13
The render takes on the v2 plate's look (dark room, black piano, liquid-chrome hands; lab/plots/k17cmp.jpg: v2 plate |
K13 | K17), flow_cos 0.86 (best so far), but hand_r 0.13 (K13 0.19) and frame r 0.70. **Conclusion:** across K10-K17
the chrome runs sit at hand_r 0.13-0.19 whatever the plate strength, anchors, pinned frame or attention pull; only a
skin-hand reading (K1/K6) follows the finger edges (0.41-0.42). A smooth chrome hand is read as a single blob whose
fingers H3 re-draws. Hypothesis K-d: give the plate *gauntlet geometry* (segmented finger plates, knuckle rings, a
cuff) so that the control edges describe a gauntlet's fingers, not a hand's, and the "gauntlet" reading inherits the
edges the way the "hand" reading does.
K-d setup: `blender_piano.py --gauntlet` adds a ring at every finger joint and a wrist cuff, bone-parented (they follow
the solved pose exactly), radius from the skinned mesh around each bone (lab/plots/gauntlet_pass.jpg). Queued: K20 =
K1 (grey plate, the only setup that pins fingers) with the canny of the *gauntlet* normal pass; K19 = K17 with the
gauntlet chrome plate, its frame 0 pinned and the gauntlet canny.

### K18: K17 at video strength 0.60: hand_r 0.12, flow_cos 0.89
Holding the v2 plate harder (0.70 -> 0.60) gives the best motion agreement so far (flow_cos 0.86 -> 0.89, flow_err
0.73 -> 0.65) but hand_r 0.13 -> 0.12 and frame r 0.70 -> 0.69: the hands move with the plate but their edge structure
is not the plate's. Still inside the chrome band; the K-d gauntlet-geometry runs (K20, K19) are next.
(Corrected entry: a first version of this entry carried wrong numbers typed before the score was read.)

### Sideways study, part 3: reach pilots (4 windows: 119.9 Rach, 189.9 and 269.9 Chopin, 379.9 ending; 247 notes; E26 settings)
| variant | key sync | sideways | other misses | shake mm | tremor | anatomy-out frames |
|---|---|---|---|---|---|---|
| E26 | 220 (89.1%) | 24 | 3 | 1.81 | 0.44 | 1125 |
| REACH_SPLIT | 226 (91.5%) | 17 | 4 | 1.73 | 0.44 | 1156 |
| REACH_SPLIT + thumb 45 / pinky 30 deg | 227 (91.9%) | 18 | 2 | 1.74 | 0.46 | 2204 |
| ROLL_STAGGER | 224 (90.7%) | 11 | 12 | 2.03 | 0.49 | 1177 |
| both | 223 (90.3%) | 11 | 13 | 1.90 | 0.49 | 1191 |
ROLL_STAGGER's 11 "unfingered" are the two four-octave Rachmaninoff chords (120.55 s: C#2-C#6, 128.5 s), missed
either way; only their label changes (the scorer looks fingers up by onset and the staggered copies have new onsets).
Capping each sub-chord's lag at its note's length changed nothing (identical per window). **Conclusion:** REACH_SPLIT is
a clean gain (+6, no motion cost); the spread mostly costs anatomy; the stagger trades reach for shake and does not add
to REACH_SPLIT. Whole-suite queue after E23 (dev branch merged between runs): E29 = E26 + REACH_SPLIT, E30 = E28 (real
hand) + REACH_SPLIT, E31 = E29 + ROLL_STAGGER; then the remaining ablations (E21, E07, E02, E22, E11-E14).

### E23 (H4 contact): E20 + 3 mm overshoot (no repair): 93.0% (+5.5 vs E20)
Debussy 92.9, Rach 91.1, Chopin 94.7; shallow 127 -> 45, sideways 82; tremor 0.32 (lower than any repair run).
**Conclusion:** overshoot alone is worth almost as much as repair alone (E24 93.2) and costs no tremor; the two
overlap (E26 = both: 94.7, +1.5 over either). Contact ladder complete: E05 82.2 -> E20 87.5 -> E23 93.0 / E24 93.2 ->
E26 94.7. Dev branch merged (REACH_SPLIT, ROLL_STAGGER, default off); E29 running.

### K20 (K-d): K1 with the gauntlet geometry's canny: steel gauntlets on the plate's hands (hand_r 0.24, frame r 0.79)
Grey plate, chrome/jazz prompt, control = canny of the normal pass with joint rings and a wrist cuff. H3 now draws
**steel gauntlets** on the hands (lab/plots/k20cmp.jpg: gauntlet chrome plate | K1 | K20), following the plate
better than any chrome run (hand_r 0.24 vs 0.13-0.19), with the best whole-frame match so far (0.79) and flow_cos 0.84.
The forearms stay skin (no geometry there). **Conclusion:** K-d supported: the *control's geometry* sets the category
(gauntlet vs hand) while still pinning the shape; the look is carried by the edges, not the plate pixels. Next: K21 =
the same with vambrace bands along the forearm (lab/plots/gauntlet_pass.jpg), K19 = full gauntlet chrome plate.

### K19: gauntlet chrome plate (0.70) + its frame 0 pinned + gauntlet canny: back to hand_r 0.12
The render takes the chrome plate's look (polished liquid chrome, dark room) and drifts (hand_r 0.12, frame r 0.70,
flow_cos 0.88; lab/plots/k19cmp.jpg: gauntlet plate | K20 | K19). Same control as K20 (0.24). **Conclusion:** a chrome
plate in the video stream costs geometry whatever the control; K20's recipe (grey plate + gauntlet-geometry control +
prompt) is the one to build on. K21 (K20 + vambrace bands) running.

### E29 (H5 reach): E26 + REACH_SPLIT: 95.5% (new best, +0.7)
Debussy 92.9, Rach 94.4, Chopin 96.8 (+1.4); sideways 71 -> 57, shallow 25, short 5; motion unchanged (wrist dev
11.6 deg, shake 1.55 mm, tremor 0.35). **Conclusion:** redistributing over-wide chords between hands removes 14
sideways misses at no motion cost; it becomes part of the default stack. Remaining 87 misses: sideways 57 (the
four-octave chords and fast leaps), shallow 25, short 5. E30 (real-size hand + REACH_SPLIT) running.

### K21 (K-d): K20 + vambrace bands: full armour on the plate's arms; the edge metric stops being fair
Steel gauntlets and banded vambraces on both arms, in the plate's positions (lab/plots/k21cmp.jpg: plate | K20 | K21,
two moments). Against the grey plate: hand_r 0.13 (K20 0.24), flow_cos 0.79, frame r 0.74. Cross-scored against the
three Blender normal passes (plain / + gauntlet / + gauntlet and vambrace) as references:
| render | vs plain: hand_r / flow_cos / flow_err | vs gauntlet | vs gauntlet+vambrace |
|---|---|---|---|
| K1 (skin) | 0.26 / 0.68 / 0.71 | 0.24 / 0.80 / 0.68 | 0.14 / 0.90 / 0.68 |
| K20 | 0.13 / 0.80 / 0.68 | 0.19 / 0.88 / 0.66 | 0.04 / 0.92 / 0.67 |
| K21 | 0.01 / 0.79 / 0.72 | 0.07 / 0.85 / 0.66 | 0.04 / **0.94 / 0.54** |
**Conclusion:** the edge-structure score (hand_r) ranks the smooth skin hand first against *every* reference,
including the armoured ones: it rewards smooth outlines and cannot rank armour detail, so it is retired as the lead
metric for armoured renders. Motion agreement against the matching geometry is fair, and there K21 is the best
render of the whole leg (flow_cos 0.94, flow_err 0.54). Next metric: a geometry-matched score (render vs the
normal pass of the same armour, silhouette IoU of the hand mask, per-finger flow).

### K22: K21 with a grey plate of the armoured geometry (plate and control agree)
Visually = K21 (lab/plots/k22cmp.jpg: armoured grey plate | K21 | K22). Against the matching geometry (gauntlet +
vambrace normal pass) flow_cos 0.95, flow_err 0.51 (K21 0.94 / 0.54), the best motion match of leg K; against the bare
grey plate hand_r 0.10, flow_cos 0.75 (expected: the plate it is scored on lacks the armour). **Conclusion:** with a
grey plate, matching the plate's geometry to the control is a small gain; the recipe (grey plate + armour geometry in
the Blender hand + its normal-pass canny + prompt) is stable. Leg K recipe for the film: K22.

### E30 (H5 reach): real-size hand + REACH_SPLIT: 90.5% (+0.8 vs E28)
Debussy 100.0, Rach 84.2 (unchanged), Chopin 95.6 (+1.7); sideways 152 -> 135; motion unchanged. **Conclusion:** the
redistribution helps the small hand in the Chopin, but its Rachmaninoff losses are octaves it cannot open to (class a2),
not hand assignment. Queued ahead of the ablations: E32 = E30 + anatomical thumb/pinky spread (CMC 45, pinky 30 deg),
E33 = E32 + home spread x1.15.

### K23: the K22 recipe on a second shot (ov125, overhead): it transfers
Full knight gauntlets with chainmail over a candle-lit cafe, hands in the plate's positions (lab/plots/k23cmp.jpg and
k23_ov125.mp4: plate | armoured grey plate | K23). Against the matching geometry: flow_cos 0.92, flow_err 0.56
(kh111 K22: 0.95 / 0.51). Whole-frame r vs the grey plate 0.60 (the cafe background replaces the grey room).
**Conclusion:** the recipe generalises across camera angles. Geometry bug seen from above: the thumb's first ring sits
on the CMC bone inside the palm and shows as a stray arc; fix: no ring on finger1-1.

### E31 (H5 reach): E29 + ROLL_STAGGER: 95.4% (-0.1): no gain
Same per section (Chopin 96.6 vs 96.8); sideways 57 -> 48 but 13 "unfingered" appear (the staggered copies of the
four-octave chords, missed either way, relabelled because the scorer matches fingers by onset); motion unchanged.
**Conclusion:** ROLL_STAGGER does not add to REACH_SPLIT on the whole suite (as in the pilot); it stays off. The
remaining misses after E29: ~57 sideways (four-octave chords, fast leaps), 25 shallow, 5 short.

### E32 (H5 reach, real-size hand): + thumb CMC 45 / pinky 30 deg abduction: 92.5% (+2.0 vs E30)
Debussy 100.0, Rach 87.1 (+2.9), Chopin 96.9 (+1.3); sideways 135 -> 97; wrist dev 10.7 deg, shake 1.59 mm, tremor
0.33; joint-frames outside the QA anatomy table 1578 -> 2977 per minute (the table caps finger 2-5 abduction at 25
deg, so the pinky at up to 30 deg counts as out). **Conclusion:** reach is the small hand's limit; opening the thumb
and pinky toward their anatomical maximum buys 38 notes. Whether 30 deg pinky abduction is acceptable is a judgement
call for the animation (pianists stretch toward it in octaves); the thumb part alone is uncontroversial. E33 (+ home
spread x1.15) next.

### E33 (H5 reach, real-size hand): E32 + home spread x1.15: 91.9% (-0.6 vs E32)
Debussy 99.0 (-1.0), Rach 86.4 (-0.7), Chopin 96.3 (-0.6); sideways 97 -> 116, shallow 24, short 16; motion and
anatomy unchanged; 49 min. **Conclusion:** planning the hand wider than its rest spacing hurts slightly; the reach gain
comes from the joint limits, not from the plan. HOME_S stays 1.0. Best real-size hand: E32 (92.5%).
(Corrected entry: the first version carried numbers typed before the results were read.)

### E21 (H4 contact): E20 + pressing pad may sink 3 mm into its own key: 88.4% (+0.9 vs E20)
Debussy 86.7, Rach 83.5, Chopin 93.2; shallow 127 -> 117, sideways 102, short 4; tremor 0.32, shake 1.59 mm.
**Conclusion:** the own-key collision holds the pad up a little (10 shallow misses), a small effect next to overshoot
(E23, -82 shallow) and repair (E24, -86); OWN_KEY_GIVE 3 mm is a candidate addition to the E29 stack.
(Run paused 09:36 for the greyscale v18 render and re-run from scratch at 09:59.)

### E07 (ablation): E05 with SPLIT_WIDE off (old playable() note dropping): 82.7% (E05 82.2)
Debussy 74.5, Rach 74.4, Chopin 91.6; unfingered 25, sideways 83 (E05 113), shallow 219, short 4.
**Conclusion:** on the E05 rig, fingering every note was slightly *negative* (-0.6): of the 25 notes it newly
fingered, few were reached, and the rolled sub-chords pulled the hand off ~30 other notes (sideways 83 -> 113).
Fingering every note is still the right policy for the film (no key goes down without a finger), but its cost should
be re-measured on the current stack: queued idea E34 = E29 with SPLIT_WIDE off.

### E02 (ablation): E06 (real-size hand, E05 code) without the FürElise wrist model: 74.9% (E06 74.6)
Debussy 93.9 (+2.1), Rach 57.4 (-0.7), Chopin 89.7 (+1.1); sideways 201 -> 174, shallow 271 -> 289; wrist dev 11.3
deg (E06 10.6), tremor 0.23 (0.26). **Conclusion:** the learned wrist placement does not raise key sync (+-0.3, within
noise); it trades a few shallow presses for sideways ones and keeps the wrist ~0.7 deg straighter. It earns its place
as a motion-naturalness term, not an accuracy term. (Measured on E05-era code; not re-measured on the E29 stack.)

### E22 (H4 contact): E05 + hard press mask (across/depth weight 3,1.5 -> 20,20; black 20,20,20): 95.3%
Debussy 96.9 (best big-hand Debussy), Rach 94.7, Chopin 95.7; sideways 59, shallow 21, short 10. But finger tremor
0.31 -> 0.66 (x2.1) and shake 1.54 -> 1.75 mm. **Conclusion:** pinning the pad hard on its key during the press is,
alone, as strong as the whole E29 stack (95.5), and costs motion quality (as the one-window pilot predicted). Queued
next, ahead of the remaining ablations: E36 = E29 + hard mask, E37 = E29 + middle mask [8,1,8].

### E11 (H1 reverse ladder 1): ragdoll + finger smoothing back on: 75.6% (E10 76.4)
Real-size hand, every wrist constraint still off, finger joint smoothing restored (current code: every note fingered).
Finger tremor 1.67 -> 0.60 (x0.36), key sync -0.8, wrist shake unchanged (10.6 mm). **Conclusion:** finger smoothing
removes most of the ragdoll's finger tremor at almost no accuracy cost; the wrist still wanders (sideways 340). The
ladder continues with E12 (wrist smoothing) and E14 (wrist joint term) after the contact runs.

### E36 (H4 contact): E29 + hard press mask: 97.0% (new best, +1.5); Debussy 100%
Debussy **100.0**, Rach 96.5, Chopin 97.1; misses 58: sideways 41, shallow 11, short 6. Tremor 0.67 (E29 0.35, x1.9),
shake 1.76 mm (1.55), wrist dev 11.3 deg. **Conclusion:** the hard mask and the E29 stack add up (E22 95.3, E29 95.5
-> 97.0): the stack fixes misses the mask leaves and vice versa. The price is finger tremor (x1.9); E37 (middle mask
[8,1,8]) measures how much of the gain survives at lower tremor. The 100% target is now 58 notes away.

### Glitch at 248-252 s (v18/v19 cut): investigation 2026-10-10 evening — NOT _close_roles
Frame read of the v19 cut confirms the visual: L hand swings off the front of the keys fully splayed (~250.4 s),
arms crossed at the forearms, and open-ended tube segments with gaps on the forearms (sleeve/cuff artifacts).
But in the performed timeline at 247-253 s: gap (xhR-xhL) never below +0.20 m, `near` = 0 everywhere,
zero role swaps. **_close_roles is exonerated for this window** — the hands never layer, so its output is inert.
Measured in-window (E36 w_249.9): keysync 53/56, no misses at 248-252; but anatomy_out = 286 — the glitch is an
anatomy/pose violation, invisible to the key-sync metric. Arm IK is exact (upper 0.290, fore 0.313 m to 1e-16),
so the tube gaps are not IK overstretch; suspects: (a) `Seg.set` orientation flip when a segment vector passes
near vertical/degenerate in `edit` view, (b) `pose_mpfb` lowerarm chain aim — mpfb forearm ends short by design
(`wr - 0.05` swallow), so a wrist yaw that exposes the cut edge shows the sleeve. L-hand xh jumps -0.154 -> -0.024
(130 mm, hand_track slice boundary) with pad span to 195 mm — check slice assignment/split for L hand there.
Next: dump `frames[j]["hands"][*]["root"]["rot"]` for j in 250-252 s (rot discontinuity = flip), and check Seg.set
for a normals/roll flip; also re-check with `--hands capsule` to separate rig from renderer.

### Probe: no rotation/position discontinuity in the animation at 248-252 s
`lab/probe_glitch_248.py` (2026-10-10 20:34): max per-frame hand-root rotation 8.4 deg (L) / 3.7 deg (R),
no wrist jump > 50 mm in the window. So the animation data itself is smooth — the glitch (crossed arms,
splayed hand off the keys, open tube ends) is introduced at the RENDER side (blender_piano.py: Seg placement /
mpfb lowerarm aim) or is an accumulation of the L-hand wander (xh -0.154 -> -0.024 jumps, 195 mm pad span)
that reads as splay at camera. Next: render the window with `--hands capsule` and `--hands mpfb` from the same
frames JSON and diff; inspect `Seg.set` roll and the mpfb `lowerarm01/02` aim when the wrist yaw is extreme.

### E37 (H4): E29 + middle press mask [8,1,8]: 96.56% (E36 96.98)
Debussy 93.9, Rach 96.3, Chopin 97.1. The middle mask keeps 97% of the hard mask's gain; check tremor column
in results.csv — if it is back near E29's 0.35, E37 is the better trade (E36's cost was tremor x1.9).
E37 trade measured: tremor does NOT come back with the middle mask — jitter 1.70 mm / tremor 0.7 stays at E36's
level (1.76 / 0.67), while keysync drops 0.4 pt (96.98 -> 96.56; Debussy 100 -> 93.9). So E36 (hard mask) stays
best: its only cost, tremor x1.9 vs E29, is identical in E37. Anatomy_out is the same 11-12/min for both —
the tremor is not what causes the 248-252 s glitch either.

## 2026-10-16: pose-to-pose rewrite (rejected by eye), then back to small steps on v18

### Pose-to-pose rig, branch `piano-grace` (worktree ~/h3b_grace): REJECTED
**Verdict (user, by eye):** "a big regression", "very jarring and jumpy"; "v18 was so close". Parked; do not resume
without the user asking. Videos and tables: band site section `pose-rig` (marked rejected).
- What it was: `WRIST_MODE=landing` (wrist follows the FürElise pianists' path; one whole-hand solve per chord onset,
  eased between landings, 0.12 s dwell) and `FINGER_MODE=pose` (`blockout/fingers.py`: each finger keyed in joint space,
  lift / key bed / held contact, minimum-jerk blends; idle fingers in one curl family on a soft spring).
- Its numbers all said "better" (suite keys in time 95.6 vs 93.2%, pops 4.8 vs 21/min, frozen 6.8 vs 12.4%, crossed
  2.9 vs 11.9%, broken 0.05 vs 0.46%; never-tuned Brahms 98.5%, Alla Turca 97.2%). The eye said worse.
  **Lesson: pops/min (any joint > 15 rad/s) and frozen % do not measure visible jerkiness.** Arm stop-and-go between
  landings and minimum-jerk finger blends read as jumpy even with no single-frame jump. Judge on footage first;
  one change at a time against v18.
- An early bug in that branch also folded the idle thumbs up over the backs of the hands (see the thumb axis note).

### Findings from that work that still hold for v18
1. **Rachmaninoff leap lag.** In the agitato both hands leap ~30 cm every 0.5 s. v18's pass-A wrist (per-frame solves
   + Whittaker) arrives late: up to **9 cm off the chord at its onset** (R wrist x 0.021 where the chord needs -0.07 at
   100.91 s); fingers then stretch to their abduction limits. Most v18 pops there are free fingers (thumb IP, pinky MCP
   flipping 40-50 deg in one frame: per-frame LM restarts from q_rest pick another pose family).
2. **Thumb hinge axis (rig bug, not changed: the user decides on rigging).** `rig/skeleton.py` derives the thumb
   MCP/IP flex axis from the rest bend between thumb bones: it comes out as the bone's local **Z** (0.94-0.96), while
   fingers 2-5 bend about local **X** (0.96-0.99, MPFB's curl axis). Positive thumb "flex" therefore swings the tip
   toward the fingers and *up* (dorsal); bending about local +X moves it across and under the palm (tip z -26 -> -45
   -> -55 mm for MCP=IP=0/20/40 deg). With LIMITS tmcp (-10, 55), tip (-20, 80) the thumb can barely curl palmar
   (10/20 deg) and comfort.json's thumb sits pinned at those limits (MCP -10.0, IP -17.4). Likely cause of "the thumb
   sticks straight out". Fix = use local X with the palmar sign for tmcp/tip; old takes would then need the old axis
   to render (store the convention in the anim JSON).
3. **Splay vs real pianists** (human_score keypoint measures; FürElise pieces 81, 32, 5, 12, 40 at 1/20 frames):
   | finger spread, median (p90) | v18 suite | pianists |
   |---|---|---|
   | index-middle | 9 (28) | 8 (16) |
   | middle-ring | -1 (14) | -5 (3) |
   | ring-pinky | 23 (44) | 18 (30) |
   Worst when the pinky presses (ring-pinky 31 / 48), but idle fingers fan too (20 / 39; v18's EXT_OUT fans idle
   fingers outward on purpose). Hand yaw is NOT the problem: v18 median -2 deg (slightly in), pianists +5.
   This is the user's "turned its hands out a bit too much".
4. **Press depth (v18 suite):** 86% of notes reach >= 0.95 key depth within 0.25 s of onset, 9% stay under 0.8,
   4% under 0.5. The user's "didn't press down enough sometimes".
5. **Real-pianist calibration of the review metrics** (FürElise 81 + 32, keypoint-derived joint angles at 24 fps):
   finger speed 31 deg/s, frozen 5.0%, pops 3.4/min, PIP 27 deg. v18 (same measure): 45, 12.4%, 27.5, 39.
6. **Hand split for one-track MIDI** (`piano.split_by_path` on piano-grace, commit 527dd59; portable): Viterbi over a
   moving left/right boundary. Agreement with the FürElise pianists' own hands: 95.5% on five pieces it was not set
   on (20, 60, 100, 120, 140) vs 83.2% for middle C.
7. Part-boundary artefact (any part-wise solve): a part only sees notes from 0.5 s before its start, so a hand that
   rests across the boundary is planned from its *next* chord (the 287.3 s part put the idle R hand in the L hand's
   register). Planning from ~8 s of earlier notes fixed it on piano-grace.

### v18 tuning, branch `piano-v18-tune` (worktree ~/h3b_v18t), 12 s windows
New knob: `RIG_W["comfort_abd"]` (rig/hand.py): multiplies the comfort term of fingers 2-5 MCP abduction so the hand
moves over instead of a finger fanning out. Default 1 = v18 byte-for-byte. Windows solved with v18 code + PIANO_OVR
(score s: Rach 100-112, Debussy 50-62, Chopin 282-294, Brahms 0-12 held out). Anims, logs and scripts:
`~/h3/hands_loop/v18tune_20261016/` (`windows/`, `v18check.py`, `score.py`).
- **T-A**: `PRESS_OVERSHOOT` 0.003 -> 0.006.
- **T-B**: T-A + `EXT_OUT` all 0 (no idle fan) + `comfort_abd` 4.
- **T-C**: T-A + `EXT_OUT` all 0 + `comfort_abd` 10.

| window | keys fully down (>=0.95) v18 / A / B / C | index-middle p90 v18 / C | ring-pinky p90 v18 / C | keys in time v18 / C | pops/min v18 / C |
|---|---|---|---|---|---|
| Brahms | 94 / 99 / 99 / 99 % | 17 / 11 | 33 / 33 | 97.0 / 97.0 | 7.5 / 10 |
| Chopin 282-294 | 92 / 94 / 96 / 96 % | 35 / 19 | 46 / 46 | 96.2 / 96.2 | 12.5 / 12.5 |
| Debussy 50-62 | 86 / 90 / 90 / 100 % | 15 / 11 | 35 / 26 | 90.5 / 100 | 10 / 2.5 |
| Rach 100-112 | 89 / 94 / 91 / 93 % | 52 / 40 | 57 / 52 | 90.9 / 92.1 | 152 / 175 |
Reading: deeper press works everywhere; comfort_abd 10 pulls the index in hard but barely moves the pinky (it flares
while pressing: the pinky's key is reached by abduction because the other fingers pin the hand). Rach got a little
busier on paper (pops 175 vs 152, broken 1.7 vs 1.0%). Next: footage (v18 | A | C side by side), then a pinky-
specific step if the user agrees the flare is the remaining "turned out" look.
Footage (v18 | T-A | T-C, same camera and moment, piano audio): band site section `v18-tune`; files in
`~/h3/hands_loop/v18tune_20261016/*.mp4` (Rach close-up + wide, Chopin close-up + wide, Debussy close-up, Brahms
close-up). Stills: T-C reads like v18 with the fingers held closer together; no new look. Awaiting the user's eye.

### 2026-10-16 evening: user picked T-C; next = idle fingers keep formation
User on the A/C footage: "c is good, only thing i notice is the middle finger, or center two fingers kinda pop out
sometimes when moving or not being used, and its usually more natural to just kinda stay in formation with the other
fingers as you go across". **T-C is the new base** (PRESS_OVERSHOOT 0.006, EXT_OUT 0, RIG_W comfort_abd 10).
- Frames at the worst moments (T-C, L hand): Debussy 53.7-54.2 s (L crossing over R) idle index and middle stand
  straight up; Rach 100.3-101.3 s one idle finger points straight forward while the others curl. Cause: v18 drives each
  idle finger to its own world-space float target (home spot + lift EXT_UP 10 mm + lengthen EXT_FWD 8 mm, glide to the
  next key up to GLIDE_MAX 0.8 s early, OVER_FREE_UP when over the other hand) plus collision pushes; nothing ties it
  to its neighbours.
- Measure (`v18tune_20261016/formation2.py`): per idle finger 2-5, |total bend (MCP+PIP+DIP) - mean of neighbours'|;
  "pointing" = idle finger bend < 30 deg while its neighbours average > 70.
  | window | v18 p50/p90, pointing | T-C p50/p90, pointing |
  |---|---|---|
  | Debussy 50-62 | 14 / 50 deg, 5.1% | 11 / 48, 4.1% |
  | Rach 100-112 | 34 / 94, 14.6% | 39 / 110, 15.5% |
  | Chopin 282-294 | 12 / 42, 2.5% | 12 / 45, 2.1% |
  | Brahms 0-12 | 13 / 47, 7.9% | 12 / 53, 8.3% |
- New knob `RIG_W["formation"]` (rig/hand.py `finger_residuals`, opt-in, 0 = v18): an idle finger's MCP/PIP/DIP are
  pulled toward the mean of its neighbours' (fingers 2-5), scaled by the human joint spread like comfort.
- Running on top of T-C (exact PIANO_OVR in `v18tune_20261016/variants.txt`): **F1** formation 3e-5; **F2** 1e-4;
  **F3** F2 + EXT_FWD 0 + EXT_UP 0.004 + GLIDE_MAX 0.4.
Results (12 s windows; `formation2.py`, `score.py`):
| window | out of formation p90 C / F1 / F2 / F3 | pointing C / F2 | keys in time C / F1 / F2 / F3 | pops/min C / F2 |
|---|---|---|---|---|
| Debussy 50-62 | 48 / 31 / 28 / 33 | 4.1 / 0.0% | 100 / 100 / 100 / **90.5** | 2.5 / 0 |
| Rach 100-112 | 110 / 90 / 78 / 74 | 15.5 / 5.4% | 92.1 / 92.1 / 92.1 / 93.9 | 175 / 165 |
| Chopin 282-294 | 45 / 23 / 18 / 17 | 2.1 / 0.0% | 96.2 all | 12.5 / 12.5 |
| Brahms 0-12 | 53 / 36 / 26 / 25 | 8.3 / 1.0% | 97.0 all | 10 / 10 |
F2 also lowers crossed tips (Rach 25 -> 19%, Chopin 23 -> 22%), raises idle-finger motion (frozen ~0-4%; speed +10%)
and curl (PIP +4 deg: idle fingers now curl with working neighbours); Debussy hand overlap 92 -> 109 frames. F3's idle
lift/lengthen/glide changes cost Debussy 10% keys: dropped. **Candidate: F2.** Frame check: Debussy 54.0 s (L over
R) idle fingers curl in formation instead of standing up; Rach 101.3 s unchanged, and NOT a reach issue (corrected after checking): no L finger presses
at 101.33 s; ring (-11 deg) and pinky (-19) just released L4/L5 and stay straight while the middle is at 114. F2's
formation pull loses to the release blend / pass-B path prior (FINGER_PRIOR 5e-3) during fast leaps. Next step there. Footage C | F2: band site section
`v18-formation`, files `v18tune_20261016/form_*.mp4`.

### 2026-10-16 night: user review of F2 — Debussy hands pile into each other; Chopin close-up broken (not reviewed)
User: "debussy has extreme overlap with right and left, use more fingers on one hand to cover those, not crossing
hands, and then the chopin final close up he is not playing the piano at all ... very very broken shot, i dont think
you reviewed it." Correct: I published after looking at two stills. **Rule from now: every video gets a 1 fps
contact sheet (`v18tune_20261016/review_sheets.sh`) that I look at in full before it goes on the site.**
- Chopin 282-294 s, R-hand close-up: 286-292 s is left hand only. v18's `hand_track` glides a resting hand toward its
  next note across the whole gap, so the idle R hand drifts toward the L hand, `_close_roles` marks it "over", and it
  takes the over-hand pose (lifted ~6 cm, OVER_YAW turned in, OVER_ROLL, free fingers curled): pinched, off the keys,
  once hanging in front of them (287.6-292.6 s). Same in v18, T-C and F2.
- Debussy opening: the L hand holds a third (F4-Ab4, ~5 s) right under the R hand's moving thirds (Bb4 and up); with the
  full-size hand at both spots the hands overlap and the role system layers one over the other.
- New switches (piano.py, default off = v18): `CONSOLIDATE` (`_consolidate`, after the staff split and
  `_reach_split`): a note sounding within CONS_NEAR = 0.12 m of keys the other hand is sounding goes to that hand when
  that hand is the busier one while it sounds and can take it (at every onset during the note, its keys plus this one
  are <= 5 within HAND_SPAN; legato tails < 80 ms ignored). Suite: 38 of 2,046 notes move (Debussy 24, Rach 4, Chopin
  10); all six L notes of Debussy 47-62 s go to the R hand (it holds the third with thumb and index while 3-5 play the
  thirds). `REST_HOLD = (1.0, 0.6)`: a hand with more than 1 s to its next note stays where it was and travels in the
  last 0.6 s (v18 drifted across the whole gap).
- Running **G** = F2 + CONSOLIDATE + REST_HOLD on Debussy 50-62 and 62-74, Chopin 282-294 and 266-278 (R hand
  playing), Rach 100-112, Brahms 0-12 (F2 also on the two new windows).
Results G (F2 + CONSOLIDATE + REST_HOLD) and **H** (G + `IDLE_SEP` 0.24 m + hands decided once on the whole piece when
CONSOLIDATE is on; window splits had seen too little context and left the L hand 14 presses in Debussy 62-74):
| window | hand overlap frames F2 / G / H | keys in time F2 / G / H | out of formation p90 F2 / H |
|---|---|---|---|
| Debussy 50-62 | 109 / 141 / **0** | 100 / 100 / 100 | 28 / 23 |
| Debussy 62-74 | 106 / 99 / **0** | 92.9 / 96.4 / **100** | 35 / 25 |
| Chopin 282-294 | 20 / 2 / 3 | 96.2 / 98.1 / 98.1 | 18 / 16 |
| Chopin 266-278 | 14 / 14 / 14 | 100 all | 32 / 32 |
| Rach 100-112, Brahms 0-12 | 0 | unchanged | unchanged |
G alone made Debussy 50-62 worse (141): with all its notes given to R, the idle L hand sat at v18's default rest spot
(piece centre - 15 cm, x ~ 0) right under R. IDLE_SEP (a hand with no note sounding and none within 0.6 s keeps >= 0.24 m
from the other, eased over +-0.25 s) fixes it: L rests at x -0.10..-0.21 while R plays. Chopin: REST_HOLD keeps the
idle R hand at x 0.15 through 286-292 s (v18 drifted it into the L hand's register and it took the over pose).
Review (1 fps contact sheets, every video, before posting): Debussy wide 50-62 / 62-74, Debussy R close-up, Chopin
282-294 wide / L / R close-ups, Chopin 266-278 R close-up — H hands apart, resting hand calm on its keys, nothing
broken at 1 fps. Band site section `v18-hands`; files `v18tune_20261016/h/`. Current candidate: **H** (PIANO_OVR in
`v18tune_20261016/variants.txt`).

### 2026-10-16 night: user "its really good" on H -> H settings become this branch's defaults; next: keys hit "a bit too high", fingertips through the piano case
- **H as defaults** (piano.py): PRESS_OVERSHOOT 0.006, EXT_OUT 0, RIG_W comfort_abd 10 + formation 1e-4, CONSOLIDATE True,
  REST_HOLD (1.0, 0.6), IDLE_SEP 0.24. Checked byte-for-byte against the PIANO_OVR H windows (Brahms 0-12, Debussy 50-62)
  before pushing. v18 itself stays on piano-v18-base.
- **Fingertips through the case** (`v18tune_20261016/case_strike.py`; tip = distal bone end + 6 mm): every hit is an IDLE
  finger, nearly straight (total bend p50 -5..22 deg), tip inside the fallboard (y >= 0.153, z < 0.055) while the hand is
  5-9 cm into the keys; mostly the middle finger, also index/ring: 106 / 19 / 51 / 115 / 81 / 8 finger-frames per 12 s in
  Debussy 50-62 / 62-74, Chopin 282-294 / 266-278, Rach, Brahms. Cause: `_key_env` drops every skin sample whose lowest
  point is above 2 cm (`d[low > 0.02] = 0`, "well above every key top") -- but the fallboard stands 5.5 cm and the lid body
  7 cm, so a fingertip in the fallboard's face is never tested. A human curls that finger slightly.
- **Strike point is not the "too high"**: pressing pads land 21-25 mm from the key front for index-ring (median), 6-13 mm
  for thumb and pinky; deeper (45-60 mm, p75) only in some Rach/Brahms chords. FürElise pianists' long fingers strike
  40-80 mm in. Pre-strike lift p50 2-5 mm, p90 12-26 mm; knuckle height 57-63 mm (pianists 42 mm x 24.3/17.7 = 58).
  Side views: pressing tips are on their keys; some long fingers lie nearly flat along the keys. Asking the user what
  "too high" refers to before changing anything.
- User on "too high": **"some long fingers lie almost flat along the keys and press with the pad from very low. A real hand
  would come down more from above with a curled finger"** -- that is the point.
  Measured (H windows, `finger_angle.py`, world frame): pressing fingertips are mostly steep (distal 47-70 deg below
  horizontal at the median, flat < 25 deg in 0-11%, mostly the index); the flat long fingers are mostly IDLE ones lying
  straight along the keys (the same fingers whose tips enter the fallboard). Inputs that make them straight: the
  collision skip above 2 cm; both calibrated rest poses are flat (comfort.json MCP -7..-2, PIP 10-22, DIP -5..0;
  comfort_mediapipe.json MCP 8-16, PIP 7-21); DIP coupling ~0 for index/ring/pinky (FürElise kdip).
- New switches: `CASE_COLLIDE` (piano.py; skin above 2 cm still meets the fallboard), `RIG_CURL` env (rig/hand.py;
  rounded rest pose for fingers 2-5 {"mcp", "pip", "kdip"}). Running on H: **I1** CASE_COLLIDE; **I2** CASE_COLLIDE +
  RIG_CURL {"mcp": 20, "pip": 45, "kdip": 0.67} (the classic rounded hand; DIP = 2/3 PIP).
Results I1 / I2 (vs H; `score.py`, `idle_straight.py`):
| window | idle fingers straight H / I1 / I2 | tips in fallboard H / I1 / I2 | keys in time H / I1 / I2 |
|---|---|---|---|
| Debussy 50-62 | 89 / 89 / 17% | 106 / 6 / 0 | 100 / **85.7** / 100 |
| Debussy 62-74 | 85 / 87 / 0% | 19 / 0 / 0 | 100 / 100 / 100 |
| Chopin 282-294 | 80 / 78 / 4% | 51 / 10 / 0 | 98.1 / 98.1 / 96.2 |
| Chopin 266-278 | 64 / 61 / 14% | 115 / 15 / 2 | 100 / 98.6 / 98.6 |
| Rach 100-112 | 41 / 38 / 19% | 81 / 20 / 5 | 92.1 / 89.0 / 89.6 |
| Brahms 0-12 | 70 / 70 / 25% | 8 / 0 / 0 | 97.0 / 95.5 / 97.0 |
I1 (case collision alone) stops the fallboard hits but leaves idle fingers straight and costs keys: dropped. I2 curls the
hand (idle bend p50 21-48 -> 64-89 deg) and empties the fallboard, but hooked tips (DIP > PIP + 30) rise in the Rach
(40 -> 99 finger-frames, ~80% idle): the rest pose gives the DIP 30 deg while its coupling to the PIP is weak (couple
5e-5), so a PIP that straightens leaves the tip bent. Frames (H | I2, Chopin 270.6 s, Debussy 56.2 s): H's long fingers
lie flat along the keys; I2's curl down onto them, perhaps a little far ("curl slightly"). Running **I3** = I2 + couple
1e-3, and **I4** = gentler curl (MCP 15, PIP 35) + couple 1e-3, both with CASE_COLLIDE.
Results I3 / I4 (vs H): tight DIP coupling removes the hooks (Rach broken 1.7% -> 0.5%). I4 (MCP 15, PIP 35) is the
gentler curl: idle fingers straight 10-35% (H 41-89%), tips in fallboard 0-3 (H up to 115), keys in time 100% on
Debussy x2, Chopin 266-278 and Brahms (H 97), but Chopin 282-294 96.2 (H 98.1) and **Rach 87.2 (H 92.1)**: 11 Rach notes
lost, 3 gained, scattered, mostly never pressed -- reach, not the fallboard (a curled rest resists the stretch). I5 =
I4 + RIG_W comfort_press 0.3 (pressing fingers pulled less to the rest pose) running on Rach/Chopin/Brahms/Debussy.
**Review of the H | I4 sheets caught a new defect, in H as well:** Debussy 58.6 s (and ~52.6, 56.6 s), with the thumb and
pinky playing, the idle index and middle stick straight out and up (total bend -9..-21 deg, tip level with its
knuckle). Root cause of both this and the user's "long fingers flat, from very low": **the hand rides at a small hand's
height.** `furelise_cal` knuckle_z 0.042 m is the FürElise pianists' (17.7 cm hands) and `_learned_wrist` scales the
pianists' wrist offsets by 24.3/17.7 in x and y but NOT z. Knuckles ~1.5 cm too low for this hand: a long finger can
only reach its key flat, an idle finger cannot hang curled without entering the keys, so the collision straightens it.
New switch `WRIST_Z_SCALE` (piano.py). Running **I6** = I4 + KNUCKLE_Z 0.0577 (0.042 x 24.3/17.7) + WRIST_Z_SCALE.
Results I5 / I6: I5 (I4 + comfort_press 0.3) wins back part of the Rach keys (87.2 -> 89.6) but loses Brahms (100 -> 97).
I6 (I4 + KNUCKLE_Z 0.0577 + WRIST_Z_SCALE) does not raise the hand much (pressing knuckle heights 41-71 mm vs H 39-62:
the whole-hand solve pulls the root back to reach), makes the Debussy pressing middle finger flatter (tip p50 28 deg, 46%
flat vs H 53 / 8%) and costs Debussy keys (95.2): dropped; the hand-height hypothesis is not confirmed. **I4 posted** (band
site `v18-curl`, files `v18tune_20261016/i4/`), all six videos reviewed; open in H and I4: idle finger sticking up while
thumb and pinky play (Debussy ~52.6, 56.6, 58.6 s); I4 costs Rach keys (92.1 -> 87.2) and Chopin 282-294 (98.1 -> 96.2).
- User: "is the problem then finger anticipation?" Measured (`v18tune_20261016/approach.py`; fingers 2-5, the pad's motion
  over the ~170 ms before its key goes down): in H **81-94% of strikes slide in at key level** (descent angle p50 2-11 deg,
  90 = straight down; pad at key-top height 170 ms before; sideways travel p50 21-27 mm, Rach 170 mm); I4 the same. v18's
  approach is a 0.10-0.18 s straight blend to a point only h = 2-5 mm over the key for soft notes, so all the travel
  happens at key level. New switch `APPROACH_ARC` (piano.finger_tip): the finger keeps "clear" over the keys while it
  travels (scaled by travel, full at 2 cm), finishes the sideways move with "lead" of the approach left, then drops; the
  strike starts from >= hmin; the approach takes >= amin. Running **J** = I4 + APPROACH_ARC {clear 0.010, lead 0.4,
  hmin 0.006, amin 0.16}.

### 2026-10-17: key placement and posture — "curl the fingers and tilt the knuckles up"
User on the Clair de Lune R close-up at 8 s (score 58.2): "touching the key literally at the joint where it starts, the
key is almost impossible to even press down there ... not considering leverage on the key press at all ... the black key
up should be off limits for the white keys", then: **"curl the fingers and tilt the knuckles up"**. At 58.2 s H's pinky
presses C5 (white) 51 mm in, right at the narrowing where the black keys begin (55 mm); the hand is low and flat (I4
lower still, palm almost on the keys). Placement never respected leverage: the calibrated FürElise strike bands put
long fingers 40-80 mm in (90 mm near black keys), shifted 12 mm forward; the press target's y weight was 1 (x, z: 3), so
pads drift deeper; key depth = pad drop / (travel x lever), so a key goes fully down from anywhere along it.
New switch `WHITE_FRONT_MAX` (piano.hitpad: white keys struck at most this far in). Running **K** = I4 base +
WHITE_FRONT_MAX 0.045 + PRESS_AXES [3, 3, 3] + PITCH0 -0.20 rad (palm up ~11 deg from wrist to knuckles) + RIG_W
root_rot 2e-3 (the hand holds the planned tilt instead of flattening freely; was 1e-4) + RIG_CURL {mcp 20, pip 45,
kdip 0.67}. Measure: `v18tune_20261016/posture.py` (knuckle tilt, knuckle height, white presses past 45 mm, fingertip
angle). J (I4 + anticipation) running in parallel.
**Correction (2026-10-17):** the earlier note "strike point is not the 'too high' ... pressing pads land 21-25 mm from the
key front for index-ring" was WRONG: `case_strike.py` pooled white keys (measured from the white front) with black keys
(measured from the black key's own front, which pulls the median down). White keys alone (`posture.py`), H: depth p50
43-47 mm, p90 51-74 mm, **44-55% of white presses deeper than 45 mm** (the black keys start at 55); I4: p50 30 mm, 26%
past 45 mm. The user's observation was right. Knuckle tilt in H while playing: +1..+8 deg (knuckles barely above the
wrist), knuckle height 56-63 mm.
Results J / K (6 windows): **K** puts white-key presses in front (past 45 mm: H 35-55% -> K 0-11%; depth p50 29-37 mm) and
curls the hand (idle straight 0-28%; PIP 38-54), keys in time ~I4 (Brahms 98.5, Chopin 98.6/96.2, Debussy 95.2/100, Rach
89.6). But **the knuckles do not tilt up**: tilt +2..+10 deg (H +1..+8): PITCH0 and root_rot 2e-3 lose to the finger
targets, the solver still sets the hand's attitude freely. **J's anticipation did not work** as measured: strikes still
slide in (74-97%), because most of the sideways travel is the hand carrying the finger at key level; APPROACH_ARC only
lifts the finger's own travel relative to its home spot. New: RIG_W `tilt` (rig/hand.py residuals: holds only the hand's
pitch -- forward axis elevation -- to the plan's; turn and roll stay free). Running **L** = K + PITCH0 -0.25 + tilt 0.1.
L (K + PITCH0 -0.25 + RIG_W tilt 0.1): the plan asks +8..+22 deg (checked with `_plan_root`), the solved R hand stays at
+2 (L hand drops to +3 from H's +10): the wrist height is held to the pianists' path (root_pos priors 0.3 / EASE 2.0),
so tilting the knuckles up would leave the fingertips short of the keys and the solver flattens the hand; tilt 0.1 loses.
**K posted** (band site `v18-front`, files `v18tune_20261016/k/`, five videos reviewed; H's raised idle finger at
Debussy 58.6 s is gone in K). Running **M** = K + PITCH0 -0.25 + tilt 1.0 (outweighs the wrist-height priors: the wrist
may drop as the knuckles rise).
**Bug found (2026-10-17): PIANO_OVR overrides of PITCH0, KNUCKLE_Z, Y_KN and the calibrated hit pads were silently
undone in every CLI run.** `__main__` calls `set_hand_model()` after the module-level override block, and it reloads
the FürElise calibration ("place", "hitpad"). Found because one pass-A frame solved directly held the tilt (+21.5 deg)
and a full `_pass3_rig` on a cache kept +16 deg through every stage, while the CLI windows ended at +2. So **I6 (hand
height) and L/M (knuckle tilt) never had their PITCH0/KNUCKLE_Z** (L/M held the hand to the calibrated near-flat plan;
that is why the tilt went *down*). Any earlier lab experiment that overrode those names was affected the same way.
Fix: `_apply_overrides()`; `set_hand_model` re-applies the plain overrides after loading. With no PIANO_OVR nothing
changes (H defaults unchanged). Re-running **M2** = K + PITCH0 -0.25 + tilt 1.0 and **KH** = K + PITCH0 -0.20 +
KNUCKLE_Z 0.0577 + WRIST_Z_SCALE (no tilt hold).
Results M2 / KH (after the fix): **M2 tilts the knuckles up +14..+16 deg in every window** (K +2..+10), knuckles 3-5 mm
higher, white presses stay at the front (0-12% past 45 mm), fingertips a little steeper in Chopin/Brahms, broken lowest
yet (Rach 0.09%). Keys vs K: Brahms 98.5 -> 95.5, Debussy 62-74 100 -> 96.4, Chopin 282-294 96.2 -> 94.3, Rach 89.6 ->
89.0, Chopin 266-278 98.6 -> 100, Debussy 50-62 95.2 =. KH (hand raised to this hand's size, no tilt hold): tilt
unchanged (+2..+9), idle hands frozen 38-45%: dropped. Side views (new `rside` camera) show the bridge: H flat with long
fingers along the keys, K curled but flat-knuckled, M2 arched with fingers coming down. Review (M2 panel cropped, 1 fps,
all six videos): nothing broken; the thumb hangs below the key fronts at Chopin ~274.6 / 277.6 s (thumb axis, unchanged),
the pinky reaches out flat in the Rach leaps (~103.6 s). Posted: band site `v18-knuckles`, files `v18tune_20261016/m2/`.
**User on M2 (2026-10-17): "its looking great"; bring the centre of the palm toward the player by 1/5 of the keys, to
balance the intense back/down curl from the pinky.** 1/5 of a white key (150 mm) = 30 mm. Measure `palm.py` (palm centre
= midpoint of wrist and knuckle line, y from the white key fronts; PIP+DIP bend of pressing fingers). M2 while playing:
palm -13..-46 mm, knuckles +13..+47 mm in from the key fronts (Debussy/Rach +40..+47: pressing pads at ~30 mm are *behind*
the knuckles, the fingers curl back under the hand), pressing curl index 63-119, middle 99-131, ring 71-103, pinky 39-54
deg. New: `HAND_BACK` (m; `_pass3_rig` shifts the wrist reference toward the player before pass A; free fingers ride
along through FREE_FOLLOW; 0 = H). Running **N** = M2 + HAND_BACK 0.030, and **N45** (0.045) as a hedge in case the
whole-hand solve gives part of the shift back.
Result N / N45 (shift before pass A): **the solve gives the shift back.** Palm moved only 2-3 mm in Debussy/Rach and
9-15 mm in Chopin/Brahms (N45 barely more than N): the whole-hand passes hold the curled finger shape (RIG_CURL comfort +
tilt hold) and pull the wrist forward again against its 0.3 prior. Keys: Debussy 50-62 95.2 -> 90.5, others ~M2. So
HAND_BACK moved to the **settled** wrist path (after pass A and the ease, before FINGER_REFINE / pass B, where the wrist
is fixed and only the fingers solve): the palm lands exactly 30 mm back and the fingers have to reach out. Running **N2**.
Result N2 (shift on the settled path, wrist fixed in pass B): the palm sits exactly 30 mm back (palm y -43..-73 mm,
knuckles -14..+17) and the curl opens right up (pressing pinky 39-54 -> 15 deg, ring 71-103 -> 13-88), knuckle tilt
holds +14..+16, but **keys in time collapse to 40-71%**: from a fixed wrist 30 mm further back the short fingers
(pinky first) cannot reach their keys; the repair loop cannot move the wrist. A pianist's wrist gives toward the hard
key. So HAND_BACK moved again, to **after pass A and before the ease** (the ragdoll ease lets pressing fingers pull the
wrist up to 25 mm per pass on a firm spring, eased over time): the palm sits back and comes in only where a key needs it.
Running **N3** (EASE_PRIOR 2.0 as H) and **N3E** (EASE_PRIOR 1.0, a softer give).
Result N3 / N3E (shift after pass A, ease gives back): **keys held** (N3 vs M2: Brahms 95.5 =, Chopin 266-278 100 ->
98.6, Chopin 282-294 94.3 -> 90.6, Debussy 95.2 / 96.4 =, Rach 89.0 -> 92.7), but the palm nets only part of the 30 mm:
Chopin/Brahms 15-16 mm back (knuckles -3..+4 mm from the key fronts; pressing pinky curl 42-54 -> 15-18 deg, ring 71-95 ->
34-95), Rach 9 mm, Debussy 50-62 10 mm, Debussy 62-74 3 mm (black-key passages: the ease pulls the hand back in to reach
them; curl there unchanged). N3E (EASE_PRIOR 1.0) ~ N3. Rendering H | M2 | N3 (`v18tune_20261016/n3/`, adds a Brahms side
view); running **N4** = N3 placement with HAND_BACK 0.050 to see if Chopin/Brahms reach the full 30 mm with keys held.
N3 review (H | M2 | N3, `v18tune_20261016/n3/`, N3 panel cropped at 1 fps, seven videos incl. a new Brahms side view):
nothing broken; long fingers reach out instead of tucking under; the thumb hanging at the key fronts (Debussy 52.6 /
56.6 s, Chopin 274.6 / 277.6 s) is the same in H, M2 and N3 (thumb axis, unchanged).
Result **N4** (N3 placement, HAND_BACK 0.050): palm nets more -- Chopin/Brahms 22-26 mm back of M2 (knuckles -9..-4 mm),
Rach 17, Debussy 6-12 -- curl opens further (Chopin 282-294 pressing middle 109 -> 67, Rach pinky 45 -> 25); keys: Rach
93.9 (best yet), Debussy 95.2 / 96.4 =, Chopin 266-278 97.1, but Brahms 91.0 (M2/N3 95.5) and Chopin 282-294 86.8 (N3
90.6, M2 94.3). Rendering M2 | N3 | N4 (`v18tune_20261016/n4/`) for the user to judge the amount by eye.
N4 review (N4 panel cropped at 1 fps, seven videos): nothing broken; thumb at Chopin 274.6 / 277.6 s as in M2; Rach pinky
still reaches out straight in the leaps (100.6, 104.6 s). **Posted M2 | N3 | N4**: band site `v18-palm`, files
`v18tune_20261016/n4/` (N3 vs H in `n3/`). N2 not shown (keys 40-71%). Waiting on the user's eye: N3 (keys held, palm
3-19 mm back) or N4 (palm 6-26 mm back, a few keys lost in Brahms / Chopin 282-294).
**User (2026-10-17): "its really close... add a 'reach' and test it out; careful not to lose ground; if it doesn't work we
recollect our thoughts."** New: `REACH` (with HAND_BACK). The palm sits the full 30 mm back on the *settled* path (N2's
placement, after pass A and the ease), and the hand **reaches in to the settled path (= M2's wrist) around any note a
finger misses from there**: detection passes run pass B from the back path and mark each note whose key is not half
down within `t` s of its onset, from `lead` s before the onset to its end; `_reach_curve` bridges gaps under `hold` s
(no back and forth), is fully in over every marked frame and eases in before / out after over `ease` s; then
FINGER_REFINE and the repair loop run as before on that path. Worst case per note = M2's wrist, so it can't lose keys
to the palm shift. Running **P** (lead 0.1, ease 0.3, hold 1.0, t 0.1) and **P2** (t 0.06).
Result **P / P2 (reach): keys held, but the palm stays in.** Keys = M2 in every window (Rach 89.6 P / 89.0 P2), nothing
lost, but the hand reaches in 65-100% of the frames (detection marked 5-44 notes per window and the 1 s bridge joins
them), so the palm nets only 0-6 mm back of M2 and the curl is unchanged. P2 ~ P. **Why** (N2's misses from the full
30 mm back, by the finger M2 used): index / middle / ring miss 0-24% (only some index black keys), but the **thumb misses
74-85%** and the **pinky 33-69%**. The short digits set where the hand must be; the long fingers curl to match. The
thumb is also the digit with the known rig problem (its flex bends about the wrong axis, it hangs below the key fronts)
-- a thumb that could reach forward along its key would let the palm sit further back. Stopped here to recollect with
the user (as agreed). REACH stays in the code, off by default.

### 2026-10-17: thumb axis (T), pinky roll (R), on N3
**User: "love all 3"** (the three options after P: fix the thumb's rig axis, roll the hand toward the pinky on pinky
notes, keep N3) -- the go-ahead for the thumb rig change. N3 is the base for all of these.
- **`RIG_THUMB_AXIS=x`** (rig/skeleton.py, default `rest` = v18/H and every earlier take): the thumb MCP/IP flex about
  the bones' own curl axis (local X, as for fingers 2-5), signed palmar, instead of the rest-bend hinge (local Z). Flexion
  now sends the tip across toward the fingers and down. The CMC is unchanged (a +-30 deg cone either way). The planner's
  thumb model (`THUMB["axis"]`, piano.py) follows the same switch (checked: planner and rig axes agree in both modes).
  The anim JSON stores `thumb_axis`; `blender_piano.pose_rig` builds its Skeleton with it, so old takes still render
  with the old hinge (checked: stored thumb joints = FK to 0.01 mm under each take's own axis, ~10 mm off under the
  other). Relaxed thumb recalibrated on the new axis (`RIG_THUMB_AXIS=x python -m blockout.rig.calibrate` ->
  `rig/comfort_thumbx.json`, only its thumb is used): it now hits the pianists' thumb medians (IP 10.6, tip pitch 11.8,
  abduction 34.7 deg); on the old axis the fit sat pinned at the limits (MCP -10, IP -17.4).
- Before: a pressing thumb sat **at its MCP limit 70-84%** and IP limit 46-82% of the time (it could only reach down by
  extending, which pulls the tip back toward the player).
- Result **T** (N3 + thumb axis): pressing thumb at its MCP limit 6-31%, IP 2-21%. Keys in time vs N3: Chopin 282-294
  90.6 -> **96.2**, Chopin 266-278 98.6 -> **100**, Brahms 95.5 -> 94.0 (1 note), Debussy x2 and Rach equal. Pops/min:
  Rach 137 -> 85, Brahms 7.5 -> 2.5, Chopin 266-278 7.5 -> 0. Palm 3-7 mm *less* back than N3 (knuckles forward
  ~4 mm); pinky curl unchanged. **The thumb fix does not let the palm sit further back:** the thumb tip still lands only
  ~12-16 mm ahead of the knuckle line -- the forward reach is set by the CMC cone around MPFB's rest metacarpal (which
  points steeply down and out), not by the hinge axis.
- **T4** (N4 + thumb): vs N4 Brahms 91.0 -> 94.0, Chopin 266-278 97.1 -> 98.6, Chopin 282-294 86.8 -> 88.7, Debussy
  62-74 96.4 -> 92.9 (1 note), Rach 93.9 -> 91.5; palm 3-7 mm less back than N4. **TP** (P reach + thumb): reach still
  marks 6-42 notes and reaches in 60-99% of frames; palm 0-10 mm back. The thumb does not rescue REACH.
- T review (N3 | T, `v18tune_20261016/t/`, T panel at 1 fps, seven videos): nothing broken. **The Chopin thumb still
  lies low in front of the keys at 274.6 / 277.6 s** (it presses middle C there: tip on the key at y 15 mm, IP joint
  y -18 z 3 mm, MCP joint y -60 z 14 mm -- angles are now pianist-like, distal ~10 deg down, but the whole thumb is
  so low that its underside sits below the key-top line in front of the key fronts).
- **Correction to the N3/N4 reviews:** the digit "hanging" straight down at Debussy 52.6 / 56.6 s is the **pinky**, not
  the thumb (rside looks from the pinky side). There the thumb holds a black key (B-flat / A-flat, y 67-76 mm), the long
  fingers idle deep (y 118-145), and the pinky strikes a white key at its very front (tip y 13-28 mm, z -9) -- reaching
  back and down from knuckles at y 35-45. Same pattern in Brahms rside (2.9, 3.9, 5.9 s). Cause: the white pad "near
  the black keys" (FürElise: pinky 29-62 mm, capped at 45) is chosen per onset slice from the notes *struck* in it, so
  a pinky note struck alone under a *held* black thumb key gets the front pad (15-36 mm).
- New switches, all off by default: **`PINKY_ROLL`** (rad; on a pinky note with the thumb not playing, the plan's roll
  turns further toward the pinky, eased by the roll spring) -- running **R** 0.15 and **R3** 0.30 on T; **`NB_HELD`**
  (held black keys count as near-black, for the pad and the hand's y offset) -- running **TB** on T; **`THUMB_CLEAR`**
  ((m, w): in front of the key fronts the thumb's skin -- metacarpal, proximal, IP -- stays m above the white key tops,
  weight w) -- running **TC** (0.004, 0.5) on T.
- Result **R / R3** (T + PINKY_ROLL 0.15 / 0.30 rad): **safe, small.** Keys = T except Brahms 94.0 -> 95.5 (both). On
  pinky notes the hand turns toward the pinky: Chopin 266-278 4.9 -> 6.7 (R) / 12.3 deg (R3), Chopin 282-294 3.8 -> 7.8 /
  11.0, Brahms 5.8 -> 6.2 / 7.0; Rach and Debussy unchanged (the thumb plays with the pinky 84-91% of those frames, and
  the roll stays out of thumb+pinky shapes). Palm and pinky curl ~unchanged (curl there was already 16-23 deg).
- Result **TB** (T + NB_HELD): rejected. It does move the Debussy 52.6 s pinky in (F5 under a held B-flat thumb: tip y
  13 -> 29 mm) but the pinky still points back (knuckles at y ~43, held deep by the thumb's black key; the white pad is
  capped at 45 mm by WHITE_FRONT_MAX = the user's K "white keys pressed at the front"), and Debussy 62-74 loses 2 notes
  (96.4 -> 89.3).
- Result **TC** (T + THUMB_CLEAR 0.004 / 0.5): rejected. The thumb does arch (pressing IP +12..+20 deg, not flat) but the
  solver gets there by bringing the hand **forward** 7-13 mm in Chopin / Brahms (palm -54 -> -47, -53 -> -40, -59 -> -50:
  most of N3's palm-back gain gone); Brahms 94.0 -> 92.5, Rach 92.7 -> 90.9. **Why:** from N3's palm-back placement the
  thumb is at nearly full stretch to its key (CMC ~113 mm from the tip), so it can only lie flat; an arched thumb
  needs the hand further in. Thumb shape and palm-back placement trade against each other on this hand.
- Rendering **N3 | T | R3** (`v18tune_20261016/r3/`) for the band site.
- R3 review (N3 | T | R3, `v18tune_20261016/r3/`, R3 panel at 1 fps, seven videos): nothing broken; R3 looks like T
  with a slight turn toward the pinky on Chopin / Brahms pinky notes. Unchanged: Chopin thumb flat at 274.6 / 277.6 s,
  pinky stab at Debussy 52.6 / 56.6 / 58.6 s and Brahms 3.9 / 5.9 s, Rach pinky straight in the leaps (101.6, 104.6 s).
  **Posted N3 | T | R3**: band site section `v18-thumb` (media `v18r3_*.mp4`); corrected the `v18-palm` Debussy note
  (the hanging digit is the pinky). Backup `manifest.backup_20261017_thumb.json`. Not default yet: T and R3 wait on
  the user's eye. Open: the Debussy/Brahms pinky stab (hand held deep by a thumb black key vs the 45 mm white-key cap);
  the flat thumb is the cost of palm-back on this hand's thumb reach.

### 2026-10-17 (evening): R3 is the default; full-suite performance render with moving cameras

- The user: "amazing work. we love it. pick what your favorite is, t looked great to me", then render the whole Knight's
  Suite with several camera modes ("approved to have fun with camera now, but our goal is watching the fingers, the
  steady cam that tracks the fingers is always good"), commit and push. Mid-render: "keep everything high contrast /
  different colors to go into h3 well when its done ... dont worry about the H3 render work right now, you're just doing
  the blender performance".
- **Pick: R3** (T + PINKY_ROLL 0.30): T, which the user liked, plus the pinky roll, which was safe on every window
  (Brahms +1 note) and only ever turns the hand a little toward the pinky.
- **R3 is now the code default** (no PIANO_OVR or env needed): WHITE_FRONT_MAX 0.045, PRESS_AXES (3, 3, 3), new
  `PALM_PITCH` -0.25 rad (replaces the FürElise-calibrated PITCH0 after the calibration loads, as PIANO_OVR PITCH0 did),
  RIG_W root_rot 0.002 + couple 0.001 + tilt 1.0, CASE_COLLIDE, HAND_BACK 0.030, PINKY_ROLL 0.30, rig `RIG_CURL`
  default `{"mcp": 20, "pip": 45, "kdip": 0.67}` (env RIG_CURL=none gives the old rest pose), `RIG_THUMB_AXIS` default
  `x` (old takes still render with their stored axis; RIG_THUMB_AXIS=rest for v18/H). v18/H again: RIG_CURL=none
  RIG_THUMB_AXIS=rest PIANO_OVR with the H values. **Check:** ks_deb and chop2 re-solved with no overrides are
  byte-identical to the reviewed R3 windows (anim and notes).
- **Full suite, R3** (`tools/v18/solve.sh`, 24 parts, ~60 min; `~/h3/hands_loop/suite_r3_20261017/solve/`), scored with
  `tools/v18/review_metrics.py` against the v18 take (same notes, byte-identical notes.json):

  | suite | v18 | R3 |
  |---|---|---|
  | keys in time | 93.2% | **94.1%** (Debussy 90.7 = 90.7, Rach 92.0 -> 93.6, Chopin 94.0 -> 94.6) |
  | frozen joints | 12.4% | **7.2%** |
  | broken frames | 0.46% | **0.07%** |
  | pops / min | 21.0 | **14.1** (Rach 82.8 -> 63.0) |
  | crossed tips | 11.9% | **2.3%** |
  | PIP curl | 31 deg | 32 deg (Debussy 26 -> 44) |
- **Camera edit, `blockout/cameras.py --style fingers`** (new): cuts on downbeats, ~8.5 s soft / ~5.5 s loud, a camera
  that keeps being drawn runs at most 14 s. Every camera follows its hand through a spring (an operator a beat behind;
  stiffer when the music is busy) with a 2.5 mm hand-held float, and backs off until the hand it watches stays inside
  92% of the frame. Shots: steadicam (front three-quarter, high, the review handcam smoothed; on the busier hand),
  steady_low (low from the hand's outer end: strikes in profile), both hands, overhead (one hand, turning slowly),
  orbit (round one hand), across (from inside the piano, case hidden, looking back at the fingers), pov. Opens
  top-down descending onto the hands, closes rising off them (height set by the hands' spread). The body is hidden
  throughout. **Arm check:** no camera may come within 8 cm of either arm's axis (forearm, upper arm); a shot that
  would is swung 18 deg further to the hand's outer side, then falls back to the steadicam (0, 25 deg), straight down,
  both hands. The first plan hit this in the Debussy (the steadicam near an upper arm) and fell back to "both" twice
  in a row (a jump cut, 35 s of one framing); fixed by the swing and by re-merging a fallback that lands on the camera
  before it. Suite edit: 40 shots, steadicam 31% + low steadicam 14% of the runtime, closest arm 11.6 cm.
- **Look, `blender_piano.py --look show`** (new; EEVEE, 1920x1080, 32 samples): high contrast, every element its own
  colour for H3 later: orange hands and arms, pure white / black keys (black keys rough so they stay black), navy case,
  grey room; soft key light, cool rim, fill, contact shadows, AO. No depth of field (`FINGER_DOF` off: every edge
  sharp). Tried first: AgX with silver/clay hands on ivory keys -- hands and white keys too close in tone (the user's
  note). The default grey Workbench look is unchanged.
- `tools/v18/render_show.sh OUT [JOBS]`: edit + Salamander audio + parallel EEVEE render to `OUT/frames_show/` (clean, no
  text: the H3 input) + `OUT/suite_show.mp4` (piece titles burned in for viewing: Debussy at 1.0 s, Rachmaninoff 37.6 s,
  Chopin 120.6 s of video time). ffmpeg 8 drawtext drops one trailing character per multi-byte character (an em dash
  cost "2" of "No. 2"), so the titles are ASCII.
- Render fixes found on the first frames: screen-space reflections left grey blotches on the white keys and keybed in
  the low shots (SSR off fixed it; AO off or rougher keys did not), so `--look show` has no SSR. The top-down opener /
  closer showed the gap between the skinned forearm's end and the elbow piece once the camera was high (closing rise,
  339-350 s): those shots now frame toward the keys as they rise (+10 cm), so the elbows stay below frame; 527 frames
  re-rendered. Render: ~4 frames/s with 6 Blender processes on the GB10 (8,415 frames in ~35 min).
- **Review** (`suite_show.mp4`, every second of the 350 s at 1 fps, 30 sheets): no camera clipping, every tracked hand
  in frame, fingers readable in every shot type; titles right. Notes: the upper arms' joint spheres show in the both-
  hands and across shots (geometry, as built); the left forearm narrows in a fast Rachmaninoff leap at video 73 s (an
  extreme wrist turn, the known ~0.8% mid-leap bends); the steadicam trails a Rachmaninoff leap for a moment at 106 s.
  The old v18 left-hand swing-off at video 248-252 s is not there in R3.
- **Posted:** band site section `suite-r3-show` (`media/suite_r3_show.mp4`, 1080p, 5 min 50 s, 225 MB). Backup
  `manifest.backup_20261017_suite_show.json`. Clean frames for H3: `~/h3/hands_loop/suite_r3_20261017/solve/frames_show/`.

### 2026-10-17 (late): user review of the suite render -> V3 (left hand, crossings, holds, strikes, wrist lift)
User: "the left hand is bugged the entire first half, and the hand crossover/interactions are still really rough.
also we aren't pressing the keys long enough/with force so it doesn't really feel like we are playing it", then
"wrists lift up in y space on breaks and motions normally". Tools and every number below: `~/h3/hands_loop/lh_fix_20261017/`
(`winrun.sh` suite test windows, `score.py`, `sbs_show.sh` show-look side by sides, `clips/`, `scores_round*.txt`).
- **Left hand, Debussy (root cause):** the FürElise wrist model (`WRIST_MODEL`, on since v18) replaces our planned wrist
  position, and it only knows the hand's own notes: after its last one it holds there ("long gaps hold"). The left hand
  plays one third at 40.5 s and nothing until 75.7 s, so it sat under the right hand (wrist x ~0, thumb at +0.15 inside
  the right hand) instead of where `IDLE_SEP` put it (-0.09..-0.12). Parts that started in the rest (part 1 on) began at
  the plan, which is why it looked fine from 14 s. Same model also dropped the over/under layering of close hands
  (`OVER_DY/DZ`, `UNDER_*`, `LEAP_DZ`, the `MIN_GAP` push): the over hand kept only its turn and roll, and turned into the
  other hand at its height. And `_consolidate` moved notes one at a time: Clair de Lune's opening F4-Ab4 third was split,
  one note to each hand. Fixes: `MODEL_IDLE_PLAN` (an off-duty hand follows the plan, eased over 0.25 s), `MODEL_ROLES`
  (role offsets added to the model's path, through the hand spring), `CONS_CHORDS` (a chord moves whole or not at all).
- **Left hand, Rachmaninoff:** the hand jumps 30-40 cm every half beat (bass octave <-> middle chord). A free finger
  glided toward its next key where the key *is* (`finger_tip` glide), so before every leap the pinky/thumb pointed out
  straight along the keys, at its spread and hyperextension stops (L pinky at the 20 deg stop 68% of frames, MCP at
  -20 deg 24%). `GLIDE_LOCAL`: the finger pre-shapes for its key as it will sit against the hand when struck.
- **Holds:** over each note's written length the key was fully down only 44-68% of the time (suite, R3). The solver
  keeps 90-98% of the holds it *plans*; the loss was the wrist path travelling to the next chord while keys were still
  held (fingers dragged off sideways), plus `release_far`. `MODEL_HOLD`: the model's path stays on a chord until its keys
  are let go; `LEAP_TIME` (0.12 s + 0.6 s/m): a hand that must move lets go just early enough for the distance, also when
  the far chord comes right after the note's written end; `WRIST_HOLD_W` 0.6 -> 3.0: held frames pin the wrist path.
- **Strikes:** Debussy/Chopin touch is 0.08-0.13, so fingers started 4-6 mm above the key and pressed slowly (ease-out):
  nothing read as a strike (pad stroke p50 4 mm in Debussy). `FORCE` {touch >= 0.35, lift >= 10 mm} for the motion only
  (velocities and audio unchanged), `ARM_DROP` 2 (the arm's dip into each onset 2-10 mm).
- **Wrist lift** (the user's note): the model's path stayed low between chords (wrist z flat, ~23 mm, pressing or not).
  `WRIST_LIFT`: in every break with no key held the wrist rises and lands on the next note: up to 35 mm in a rest (full at
  0.12 + 0.5 s), 0.35 x the distance travelled on a leap (at most 50 mm); free fingers ride up with it.
- Tried, not taken: `HOLD_REACH` 0.17 m (keep a held note while one hand spans it and the next chord, the model placing
  the hand over both): held +4-6 points in Debussy/Chopin, but keys in time Debussy end 92 -> 85, Chopin 99 -> 96 and
  crossed / over-spread fingers back in Chopin. Off.

Test windows (suite seconds; 12 s each; base = R3; held = share of written note length with the key fully down; stroke =
pad travel into a key, p50; hand-hand = frames with the hands' skins > 3 mm into each other):
| window | keys in time R3 / V3 | pops/min | held % | L pinky at stop % | stroke mm | hand-hand |
|---|---|---|---|---|---|---|
| Debussy 39.9 (opening) | 100 / 100 | 2.5 / 0 | 74 / 81 | 0 / 0 | 3 / 10 | 158 / 11 |
| Debussy 50 | 95.2 / 95.2 | 2.5 / 0 | 62 / 61 | 0 / 0 | 4 / 11 | 0 / 0 |
| Debussy 66 (end) | 80.8 / 92.3 | 7.5 / 0 | 66 / 71 | 10 / 1 | 1 / 9 | 188 / 2 |
| Rach 100 | 92.7 / 89.0 | 85 / 20 | 45 / 28 | 68 / 45 | 15 / 37 | 0 / 0 |
| Rach 122 | 92.7 / 92.0 | 75 / 20 | 51 / 43 | 72 / 62 | 13 / 28 | 7 / 37 |
| Rach 136 | 78.8 / 93.9 | 12.5 / 2.5 | 46 / 50 | 64 / 63 | 12 / 13 | 46 / 10 |
| Chopin 282 | 96.2 / 94.3 | 2.5 / 0 | 19 / 28 | 6 / 6 | 12 / 14 | 11 / 3 |
| Chopin 266 | 100 / 98.6 | 2.5 / 2.5 | 58 / 60 | 5 / 6 | 10 / 13 | 3 / 2 |
Also: crossed fingertips Rach 100 9.5 -> 0%, L MCP at its hyperextension stop Rach 24 -> 11%, hand turning > 300 deg/s
L 5.2 -> 2.8%. Rach held % goes *down* by design: on the leaps the hand now lets go, lifts and travels (the stroke into
the keys doubles). Wrist z in Rach 100: R3 ~23 mm pressing or not; V3 21-27 mm pressing, 41 mm between chords, max 63.
By eye (`clips/`): Debussy opening -- R3 hands glued together for 11 s, V3 the left hand holds its third beside the
right, then rests 12 cm to its left; Rach left hand -- R3 a finger pointing out along the keys on every bass leap
(67.1, 68.1, 69.1 s), V3 a compact hand on its chord.
- **V3 = the code defaults** (`CONS_CHORDS, MODEL_IDLE_PLAN, MODEL_ROLES, GLIDE_LOCAL, MODEL_HOLD` True; `WRIST_HOLD_W` 3.0;
  `FORCE`, `ARM_DROP` 2, `LEAP_TIME` (0.12, 0.6), `WRIST_LIFT` as above). R3 = all of those off / back to their R3 values
  (noted on each constant).
- **Full suite, V3** (`~/h3/hands_loop/suite_v3_20261017/solve/`; `tools/v18/solve.sh`, defaults; a fresh window solve with
  the defaults is byte-identical to the V3 window): `review_metrics.py` + `lh_fix_20261017/score.py`, R3 -> V3:
  | | R3 | V3 |
  |---|---|---|
  | keys in time | 94.1% | 93.7% (Debussy 90.7 -> **95.9**, Rach 93.6 -> 92.0, Chopin 94.6 -> 94.7) |
  | pops / min | 14.1 | **5.0** (Rach 63 -> 20) |
  | crossed tips | 2.3% | **0.6%** |
  | frozen joints | 7.2% | 6.1% |
  | broken frames | 0.07% | 0.05% |
  | hands overlapping > 2 cm | Debussy 39%, suite 11% | Debussy **1%**, suite 7% |
  | L pinky at its spread stop / L MCP at hyperextension | 19 / 4.4% | 16 / 2.4% (Rach 60/14 -> 48/8) |
  | pad stroke into a key, p50 | 12 mm (Debussy 4) | **16 mm** (Debussy 9, Rach 23) |
  | key fully down over the written length | 51% | 48% (Debussy 71 -> 75, Rach 51 -> 42, Chopin 50 = 50) |
  Held does not improve: the Rach leaps now let go to travel (by design), and in the Chopin 90% of the lost time is the
  finger already gone to another key (`release_far`/fingering: pedalled arpeggios spread wider than one hand), not the
  pad slipping. Holding those would need the pedal model or a different fingering; left for later.
- **Render fix:** the steadicam on the left hand looks along the raised right forearm in Rachmaninoff (video 99-107 s)
  and showed the open end of the skinned forearm mesh (a hollow cone; the 6 cm primitive sleeve from the elbow didn't
  reach it). `blender_piano.py` `SLEEVE` (env `RIG_SLEEVE`) 0.06 -> 0.14 m covers it (tested 0.06 / 0.10 / 0.14 on the
  three worst frames); the 3,381 suite frames with an elbow end in view were re-rendered.
- **Posted:** band site section `suite-v3-show` (`media/suite_v3_show.mp4`, 1080p, 5:50; plus Debussy opening and Rach left-hand R3 | V3 clips); backup `manifest.backup_20261017_suite_v3.json`. The hands cross in the score at video 331-332 s (left arpeggio to C#4 over the right's F#3): the right hand lifts over (wrist 105-110 mm), closest 18 mm.
- **Review** (`suite_show.mp4`, V3, all 351 s at 1 fps): the left hand never parks under the right in the first half;
  Rach left hand compact on its bass jumps; wrists visibly up between chords. Still seen: the right index lifted high
  for a beat at 60 s and 64 s (same in R3), joint spheres on the upper arms in the across shots (as before).

## 2026-10-17 (night): the working piano (`blockout/action.py`) -- what V3 actually plays

The user: "make sure our final piano playing is perfected, holds the notes, etc, instead of just visual simulations,
lets build out a working piano with pressures, and transition to song vs what it would sound like with our current
model playing"; then `/loop until the hands actually play the song perfectly`. Renders held (the GPU is busy with the
user's benchmarks); everything below is CPU.

- **Why:** `fr["keys"]` comes from the finger the plan assigns to each note only (and a key "half down" counted as
  played), and the audio has always been the written MIDI (`sampler.py` on `notes.json`). Nothing listened to the hands.
- **The piano** (`blockout/action.py`, module docstring has the model): skin rebuilt per frame from the take's joint
  angles with the rig (pads match the take to 0.006 mm), PCHIP between frames at 1 kHz; keys are weighted levers on a
  balance rail 25 cm back (DW 50 g, UW 25 g, felt at 10 mm, free rise <= 6 m/s^2); fingertip pulp 1 N/mm, the bed
  stops a finger past 3 mm of squash (it falls back <= 0.3 m/s); a finger presses a key only after coming down onto it
  from above (sliding in below the top = "through key side", counted, not played); hammer rides at 5:1, lets off 2.5 mm
  before the string, flies under gravity, sounds only if it gets there; repetition resets below 60% depth; damper
  45%/35%. Velocity = one log curve on hammer speed fit over the matched notes (r reported). `sampler.render` got
  `pedal=` (array / "off") and `norm=False` so song and model go through the same piano, pedal and gain.
- **Bugs met on the way (fixed):** the sim loop span 0 steps forever when the window left < 1 ms (full suite never
  ended); first contact model used capsules on the bone line (too fat, wrong); the bed lift teleported fingers 1-2 mm
  when crossing a black/white bed edge (loud fake blows); the renderer's back-of-key pivot made any touch near the
  fallboard throw the key front down.
- **V3 on the working piano** (`~/h3/hands_loop/action_20261017/v3/`, 37 s for the suite without audio):
  | | Debussy | Rach | Chopin | suite |
  |---|---|---|---|---|
  | written notes sounded | 81.4% | 93.5% | 89.0% | **90.1%** |
  | never sounded | 18 | 45 | 124 | 189 |
  | wrong keys sounded | 128 | 772 | 2718 | **3627 (621/min)** |
  | struck twice | 29 | 125 | 330 | 487 |
  | timing p50 / p90 ms | 13 / 37 | 27 / 48 | 14 / 38 | 18 / 43 |
  | held >= 90% of written length | 52% | 24% | 34% | 31% |
  | stops audibly early with the pedal | 28% | 30% | 56% | 45% |
  Velocity curve r = 0.37 (vel = 77.8 + 16.3 ln(hammer m/s)).
- **Faults by cause** (`tools/faults.py`): wrong keys: free finger (idle / travelling) skin in the key tops 2002, free
  thumb 522, pressing thumb also on a neighbour 731, pressing finger also on a neighbour 365, same key re-struck 487
  (mostly an arriving finger tapping its key 0.1-0.3 s before the real strike). Never sounded: blow > 120 ms early 93,
  short of let-off 24, never touched 23, late 16, repeat not re-armed 13, key already down 10, too slow 10.
  Checked by eye on the render: video 10.4 s the right thumb lies along the key fronts and slides 9 cm over G4-B4,
  its skin 2-6 mm inside the key tops (the solver's key collision is a soft residual, env weight 20: a few mm of
  sinking costs it almost nothing).
- **Posted:** band site section `working-piano` (A/B player: song / hands / both without pedal, synced to the V3
  video; table; three piano rolls). `band_site.py` got an `ab` item kind. Manifest backup
  `manifest.backup_20261017_action_v3.json`.
- **Next (C1/C2, running):** `FREE_CLEAR` (m of clearance for skin of a finger with nothing to press) and
  `FREE_ENV_GAIN` (its key-collision weight) in `_key_env`; C2 adds `THUMB_PRESS_GIVE`/`OWN_KEY_GIVE` 0.5 mm. Windows
  ks_deb, ks_rach, ks_chop, chop2 vs base, scored on the working piano. Tools: `~/h3/hands_loop/action_20261017/tools/`.
- **C/E rounds** (windows, `~/h3/hands_loop/action_20261017/clear/`, `tools/score_win.py`; 12 s windows, counts for the
  window): C1 = `FREE_CLEAR` 1.5 mm + `FREE_ENV_GAIN` 25; C2 = C1 + `THUMB_PRESS_GIVE`/`OWN_KEY_GIVE` 0.5 mm; E1/E2 = C2 +
  `FREE_BLACK` 5 mm (a free finger rides over the black tops anywhere in the black-key zone: a leap can skip a whole black
  key between frames) + `ENV_GAIN` 5 / 25; E3 = C2 + `FREE_BLACK`.
  | window | base wrong / sounded | C2 | E3 | E1 | E2 |
  |---|---|---|---|---|---|
  | ks_rach | 202 / 93.8% | 157 / 89.6 | 118 / 84.0 | 97 / 71.5 | 95 / 65.3 |
  | ks_chop | 134 / 95.7% | 107 / 97.9 | 75 / 91.5 | 75 / 83.0 | 66 / 70.2 |
  | chop2 | 116 / 91.2% | 77 / 89.5 | 56 / 78.9 | 46 / 63.2 | 50 / 66.7 |
  Stronger collisions halve the wrong keys and lose a third of the written notes: a tug of war. Cause: the solver's skin is
  flat discs (`surface_under_soft_v`, any overlap counts full depth) and a pressing finger's *own* key counts as well (only
  `OWN_KEY_GIVE` of slack), so a pad on a black key "crushes" both white neighbours; a heavy weight then stops the press.
  Frame traces (Rach 100-112): pads landing 8.5 mm off a key centre catch the neighbour (both sound, vel ~115); a pressing
  finger's middle of the distal bone reaching into the black-key zone; a pressing thumb across the next white key; free
  fingers resting among the black keys. In Chopin the left hand's leaps travel 0.5-1 m/s with the lowest skin 5-13 mm up
  over the black-key zone: it brushes black tops (a glissando) -- all physically real.
- **R round (running):** `KEY_ROUND` = the solver collides rounded skin against each key top with the working piano's own
  contact model (`key_pen_round_v`), the pressing finger's own key left out (rail and fallboard as before); then strong
  weights only block what the piano would sound.
- **Holds** (`tools/holds.py`, V3 suite): 770 of 1727 notes (>= 80 ms) stop audibly early with the pedal: the plan's
  fingering gives the finger to another note first 486 (654 s lost), the hand travels on (`release_far`/`LEAP_TIME`) 259,
  the solved finger leaves its key early 17. By piece and hand: Chopin L 348 (written notes run past the cutting bass change
  by p50 0.54 s: in our transcription of a recording these are mostly the pianist's pedal blur, an arpeggio no hand holds),
  Chopin R 209, Rach L 99, Rach R 92, Debussy 22. So holds are a fingering / pedal question more than a motion one.
- **R round** (`KEY_ROUND` + `ENV_GAIN`): same tug of war. Rach wrong 202 -> 94-111 but sounded 93.8 -> 67-79%: the
  octave chords' pinky lands 15-36 mm short. Any whole-hand collision rule costs the stretch (a thumb or inner finger
  may no longer lie on a neighbour, so the hand rises / turns).
- **K round** (`CLEAN` in `_solve_frame`: violating fingers re-solved alone, wrist fixed): wrong -30..-50%, pads move
  ~2 mm, but finger pops double (Rach 20 -> 40-71/min): per-frame fixes flicker.
- **Post-pass cleanup `blockout/clean.py`** (on a finished take; module docstring): per frame and finger, rounded skin
  vs key tops (the working piano's contact model), own keys (held / landing within 0.3 s / leaving within 0.15 s, from the
  take's fingering) excluded, free fingers also kept over the black tops in the black-key zone and checked half-way
  along their path to the neighbouring frames (84 of the 106 wrong keys left after a frame-only cleanup happened
  *between* frames: fast low travel across black keys); violating fingers re-solved alone (pressing / landing pads pinned
  to the take's own pads, depth 1000 / along 300 / across 10 toward the key centre line), the joint corrections blended
  per finger in time (Whittaker). Lessons: a pressing finger has no spare joints (any fix lifts it: P1 Rach sounded
  61.8%); pins must be to the *original* pads (re-pinning each pass crept presses up 3-4 mm); the time blend must not
  bleed a free finger's lift into its strike; the sweep check must skip pressing / landing fingers (their path is the
  strike). Windows (wrong / sounded / pops-per-min): Rach base 202 / 93.8 / 20 -> S1 134 / 90.3 / 14; Chopin 282
  134 / 95.7 / 0 -> 48 / 95.7 / 0; Chopin 266 116 / 91.2 / 0 -> 44 / 91.2 / 2.9.
- **Whole suite with S1** (`~/h3/hands_loop/action_20261017/suite_clean/`, 3.7 min on 16 cores): wrong 3627 -> 1224,
  struck twice 487 -> 202, but sounded 90.1 -> 83.7% and pops 5 -> 10.6/min, pads moved 10 mm mean. Not shippable: 108 of
  the 178 notes lost had the pressing pad ~7 mm higher (the strong time blend pulled free-finger lifts into the strikes).
  Hard-pinning strikes (S4, `W_PRESS` 1e4) keeps the notes and pops jump (Rach 63/min): a free finger lifted high just
  before a pinned landing has to drop fast. Conclusion: the big corrections are free fingers V3 *plans* to float 3-15 mm
  over the keys and low among the black keys; that belongs in the plan, not in a patch.
- **H round (running):** plan knobs `FREE_HOVER_ADD` (m added to the free float), `FREE_BLACK_PLAN` (free tips over the
  black-key zone float over the black tops), `THUMB_HOVER` (idle thumb's share of the float; V3 0.4 = rests low on the key
  edge). H1 = +6 mm, black zone 5 mm, thumb 1.0; H2 = H1 + `FORCE` hmin 14 mm; H3 = H1 + `FREE_ENV_GAIN` 5 + `FREE_CLEAR` 1 mm.
- **H round** (windows; `plan/`): H3 (plan: free float +6 mm, over the black tops in the black-key zone, idle thumb floats
  like the fingers; solver: free-finger key collisions x5 with 1 mm clearance) alone adds no pops (Rach 20 = base, Chopin
  0) and moves pads 4-5 mm (Debussy 20 mm: idle fingers and thumb float higher). With the cleanup on top (`clean.py`
  gentle pins `W_PRESS` 50, corrections capped at `MAX_DQ` 0.3 rad per joint = H3m):
  | window | base wrong / sounded / pops | H3 | H3m |
  |---|---|---|---|
  | ks_deb | 23 / 94.4 / 0 | 28 / 94.4 / 2.9 | 18 / 88.9 / 5.7 |
  | ks_rach | 202 / 93.8 / 20 | 178 / 91.0 / 20 | 126 / 89.6 / 11.4 |
  | ks_chop | 134 / 95.7 / 0 | 93 / **100** / 0 | 39 / 97.9 / 2.9 |
  | chop2 | 116 / 91.2 / 0 | 76 / 91.2 / 0 | 33 / 91.2 / 0 |
  H3 is being solved for the whole suite (`suite_h3/`). Rach residue (H3m): pressing thumbs catching the next white key 55,
  pressing long fingers' distal ends catching the black key behind 17 (W->B) / the next white 23, free fingers 10.
  The thumb's contact point is the corner of its tip facing the fingers (rig `eff_local` side 0.45 w), aimed at the key
  centre: the 2 cm thumb overhangs its outside neighbour by ~4.6 mm every time. `THUMB_AIM` (m, toward the fingers)
  added; T1 = H3 + THUMB_AIM 5 mm, T2 = T1 + `HITPAD_SHIFT` -18 mm (strikes 6 mm nearer the key fronts) running.
- **T round** (windows): T1 = H3 + `THUMB_AIM` 5 mm: Rach sounded 94.4% (base 93.8, H3 91.0), wrong 159; Chopin 282
  97.9%, wrong 94. T2 (+ `HITPAD_SHIFT` -18 mm) no better: dropped. Cleanup on T1: with pressing pads slid to the key
  centre (x weight 10) it pushed *thumbs* off their keys (L1 on B4 17-26 mm off, lifted: lost); `CENTRE` 0 and
  `PRESS_W` (100, 300, 1000) (= T1p: pressing pads stay exactly where the solve put them; the cleanup only moves free
  fingers) -> Rach wrong 132 / 91.7%, Chopin 282 44 / 97.9%, no new pops. The 4 Rach notes T1p "loses" were never played:
  in T1 they sounded by accident (a free finger brushing the right key); their assigned fingers never pressed them.
- **Rach misses are octave chords**: every remaining never-sounded Rach note is in a 4-note chord spanning 12 semitones
  (165 mm); the pinky (or middle) lands 15-38 mm short. `RIG_W` comfort_abd 10 (H: hands fanned too much) also keeps
  octave chords from spreading. `WIDE_ABD` (m, gain) relaxes it only while a hand presses keys >= m apart; W1 = T1 +
  (0.14, 1.0) running on ks_rach + rach2.
- **Ops:** ~30 solves at once with llama-server + ComfyUI resident ran toymaker out of memory at 01:27 (hydra-head
  SIGKILLed, systemd restarted it; one suite part died silently, the stitch came out 8,089 frames). Now <= 12 at once
  (`tools/solve_lim.sh`); part 16 of the H3 suite re-run alone; T1 suite solving at 11 parts at a time (`suite_t1/`).
- **Cleanup, final shape** (`clean.py` defaults now): pressing fingers are left exactly as solved (`FIX_PRESSING` False:
  applying their own re-solves made thumbs roll frame to frame, Rach pops 20 -> 49/min, broken 1.3%); press / landing
  frames pinned hard (`W_PRESS` 1e4); a free finger's correction tapers to zero over 3 frames before a landing and after a
  release (`TAPER`): no lift that then has to snap onto a strike. Windows (T1u / H3u vs base): Debussy 94.4 = 94.4%, wrong
  23 -> 19; Rach 93.8 -> 93.1%, 202 -> 140, pops 20 = 20; Chopin 282 95.7 -> 97.9%, 134 -> 67; Chopin 266 91.2 = 91.2%,
  116 -> 46; pops ~ base.
- **H3 suite + cleanup (`suite_h3/cand2/`)** vs V3: wrong keys 3627 -> **1707**, struck twice 487 -> 227, sounded 90.1 ->
  88.7% (Debussy 81.4 -> 85.6, Rach 93.5 -> 91.1, Chopin 89.0 -> 87.5), timing p90 43 -> 44 ms, pops 5.0 -> 5.8/min,
  broken 0.05 -> 0.06%, crossed 0.57 -> 1.11%. (H3 suite before the cleanup: 89.7% / 2587 wrong / 316 twice.)
  The first cleanup on this suite (before the taper) lost 124 notes: soft press pins let free-finger lifts bleed in.
- **Left (cand2):** pressing thumbs 833 and pressing fingers 609 of the wrong keys; the thumb's shaft lies across the next
  key toward the fingers (W->W, and B->W from a black key) in every piece, plus W->W outside. V3's solver lets a pressing
  thumb sink `THUMB_PRESS_GIVE` 4 mm into other keys; G1 = T1 + give 0 running (Rach). Octave chords: rig span ~130-136 mm
  thumb-to-pinky vs 165 needed (joint limits hit) -- the user's call.
- **C1 = T1 suite + cleanup** (`suite_t1/cand/`): wrong keys 3627 -> **1440** (-60%), struck twice 487 -> 220, sounded
  90.1 -> 88.8% (Debussy 81.4 -> 85.6, Rach 93.5 -> 92.0, Chopin 89.0 -> 87.2), timing p90 43 -> 44 ms, pops 5.0 -> 5.7,
  broken 0.05 -> 0.08%, crossed 0.57 -> 0.94%. Posted: band site section `working-piano-c1` (A/B: song | V3 hands | C1
  hands over the V3 picture, one gain; table; rolls). G1 (pressing thumb give 0) on Rach: -15 wrong, -2 notes: not taken.
- **"Blow early" is mostly repeats:** 94 of the never-sounded notes (V3 93) have a blow of their pitch 0.12-0.6 s early;
  ~73 are a repeated note whose earlier blow is the previous note's: the key never rose to re-arm. V3 frees only the
  *same* finger before a re-strike (`finger_events`); a finger change on a repeated note (common in the Chopin) leaves the
  old finger holding the key. `REPEAT_GAP` (s): any earlier hold on a pitch (any finger, either hand) ends strike_time +
  gap before its next onset. RG1 = T1 + 0.09 s running (4 windows), plus T1 on chop2 / ks_deb for a like-for-like base.
- **Stitch bug (fixed, `tools/v18/stitch.py`):** the stitched suite copied part 0's header, so its `fingering` covered
  only 37-55 s. Every whole-suite tool that read it was blind after that: `clean.py` didn't know which key a finger was
  about to land on (outside a press) and lifted landing fingers -- that is why each whole-suite cleanup lost more notes
  than the windows did. Now every part's fingering is merged, each over its own stretch (cut mid-overlap). V3's own
  suite.json is unchanged (its frames are what they are); re-stitched suite_h3 and suite_t1.
- **C1f** (T1 suite, re-stitched, + cleanup; `suite_t1/cand_f/`) vs V3: sounded 90.1 -> **89.7%** (Debussy 81.4 -> 84.5,
  Rach 93.5 -> 93.3, Chopin 89.0 -> 88.0; raw T1 89.8: the cleanup costs nothing now), wrong keys 3627 -> **1635** (-55%),
  struck twice 487 -> 256, pops 5.0 -> 5.9, broken 0.05 -> 0.07%, crossed tips 0.57 -> 1.47% (raw T1 0.97: the plan
  raises it in the Chopin, the cleanup adds ~0.5). Band site post corrected to these (the first C1 numbers, 1,440 /
  88.8%, came from the blind cleanup), with a dated correction note.
- **RG1** (T1 + `REPEAT_GAP` 0.09 s, windows): Rach 95.1% (T1 94.4), struck twice 12 -> 8; Chopin 282 100% (T1 97.9);
  cleaned: Rach 93.8% / 128 wrong, Chopin 282 100% / 72, Chopin 266 89.5% / 41, Debussy 94.4% / 12. C2 = T1 +
  REPEAT_GAP suite solving (`suite_c2/`).
- **`CROSS_GUARD`** (clean.py, on): a finger's fix is undone if it crosses index..pinky tips that weren't crossed. No change
  on the four windows (the crossings appeared on the whole suite).
- **PX1 = RG1 + `PRESS_AXES` x 6 (was 3, both colours)**: pressing pads pulled twice as hard to the key centre line across
  the keyboard. Rach 93.8 -> **96.5%** sounded, wrong (cleaned) 202 -> 104, pops 20 -> 14; Chopin 266 91.2 -> 94.7% raw /
  93.0 cleaned, wrong 116 -> 39. Costs: Rach broken 0.05 -> 0.35% (9 frames of a tip hooked > 30 deg past its PIP, index
  and pinky), Rach timing p90 55 -> 69 ms. PX2 (x 4.5) and PX1 on ks_chop / ks_deb running. (Debussy's timing p90 on T1 is
  one note: A#4 sounded 77 ms late by the right thumb's shaft, not by its finger.)
- **PX1 on all four windows (cleaned)** vs base: Debussy 94.4 = 94.4%, wrong 23 -> 22, timing p90 41 -> 38; Rach 93.8 ->
  96.5%, 202 -> 104; Chopin 282 95.7 -> 97.9%, 134 -> 67; Chopin 266 91.2 -> 93.0%, 116 -> 39. PX2 (x 4.5): Rach 94.4%,
  Chopin 266 94.7%: not better overall. **C3 = C2 + `PRESS_AXES` x 6** suite solving (`suite_c3/`).
- **C2 = T1 + `REPEAT_GAP` 0.09, suite + cleanup** (`suite_c2/cand/`) vs V3: sounded 90.1 -> **90.3%** (first whole-suite
  candidate above V3; Debussy 81.4 -> 89.7, Rach 93.5 -> 90.8, Chopin 89.0 -> 90.1), wrong 3627 -> **1593**, struck twice
  487 -> 235, pops 5.0 -> 6.1, broken 0.05 -> 0.06%, crossed 0.57 -> 1.30%. Raw C2 (no cleanup): 90.5% / 2344 wrong / 271.
- **C3 = C2 + `PRESS_AXES` x 6, suite + cleanup** (`suite_c3/cand_g/`, cleanup with the crossing guard applied after the
  time blend) vs V3: sounded 90.1 -> **92.1%** (Debussy 81.4 -> 88.7, Rach 93.5 -> 93.3, Chopin 89.0 -> 91.5), wrong keys
  3627 -> **1542** (-57%), struck twice 487 -> 240, timing p90 43 -> 49 ms, pops 5.0 -> 6.7/min, broken 0.05 -> 0.15%
  (tips hooked past the PIP in Rach chords), crossed tips 0.57 -> 1.17% (the raw C3 solve has 1.20: the plan's, mostly
  Chopin). Before moving the crossing guard after the blend the cleanup added +0.6 (1.80%): the per-frame guard can't
  see crossings made by easing corrections into neighbouring frames; now a finger's fix is pinned to zero for +-2 frames
  wherever the blended result crosses tips the original didn't, and the blend is redone.
- Posted: band site section `working-piano-c3` (A/B song | V3 | C3 over the V3 picture; V3 / C1 / C3 table; rolls), backup
  `manifest.backup_20261018_action_c3.json`. `action.py --write-anim OUT` writes a take whose key depths come from the
  working piano (for renders, once the GPU is ours again).
- **C3g faults:** wrong keys by phase: pressing thumbs 518, pressing fingers 500, landing fingers 224 (31 + 9 + 8 black
  target -> another black key; 22 + 7 white target -> the black key beside it; mostly the distal end), free 200, leaving
  101. Never sounded 152: 48 look like repeats but are mostly Rach right-pinky top notes of fff chords that are fingered
  and never pressed (the octave-span limit again), ~23 are genuinely fast repeated chords (0.17 s apart).
- **AA1** = C3 + `APPROACH_ARC` {clear 20 mm, lead 0.4, hmin 12 mm, amin 0.10 s} (travel high, over the key early, then
  drop): running on ks_chop / chop2 / ks_rach against PX1 (= C3 settings).
- **AA1** (C3 + `APPROACH_ARC`) vs PX1 (= C3 settings), cleaned, three windows: notes +2 (Rach 96.5 -> 98.6%, Chopin 282
  97.9 -> 100, Chopin 266 94.7 -> 91.2), wrong +20 (210 -> 230), struck twice 23 -> 14. Not a clear win: off.
- **Loop stopped (2026-10-18 07:05)** at a decision point, not at perfect: what's left needs the user's calls (octave
  reach = rig joint limits; arched thumb, rejected before as TC; holds = fingering / pedal) and their eye on a render (all
  C-round motion changes are metric-judged only). C3 render input ready: `~/h3/hands_loop/action_20261017/suite_c3/render/`
  (physics key depths via `action.py --write-anim`; ks.wav = 39.9 s silence + C3 hands). Handoff:
  `~/h3/HANDOFF_20261018_working_piano.md`.
- **Loop resumed 2026-10-18 09:45** (the user: stopping at a decision point was wrong; the job isn't done). Open
  decisions become side candidates; GPU renders stay held for their benchmarks.

## 2026-10-18 (morning): the pedal, and the judge's skin

- **Holds were mostly the pedal model.** `sampler.auto_pedal` changes the pedal at every new lowest note of the lower
  track; in a broken-chord left hand every arpeggio note is "new", so it changed 3-5 times inside one written note
  (120 changes/min). The Chopin is a transcription of a pedalled recording (no pedal data, README): its written lengths
  are sound lengths, and its note ends cluster (68% have another end within 25 ms, 30% shuffled) about 0.1 s after a
  bass onset: the player's own pedal lifts. **`sampler.harmony_pedal`** (now the working piano's pedal, `action.py
  --pedal harmony`, default; `auto_pedal` stays the default of `render()`, the H3 audio paths use it): changes where the
  bass really moves (the new lowest lower-track note at or below every lower-track note still written to sound,
  ignoring ones ending within 0.10 s; broken-chord bass notes don't count) 0.12 s after the onset (syncopated: legato
  releases fall first), plus clusters of >= 2 written ends within 30 ms; changes closer than 80 ms merge. Grid-searched
  (`~/h3/hands_loop/holds_20261018/grid.py`) for cuts with the song's blur held at or under auto's:

  | pedal | changes/min | song blur mean (s past written end) | notes sounding at once | C3 cut | V3 cut |
  |---|---|---|---|---|---|
  | auto (before) | 120 | 0.48 | 8.3 | 47.6% | 44.6% |
  | harmony | 59 | **0.26** | 7.1 | **13.0%** | 13.0% |

  The song is *less* blurred, not more. Rejected: a held-bass veto (cuts 2.6% but blur 3.9 s, 27 notes at once: mush);
  splitting score / transcription sources (Rach cuts 8 -> 24%). What's left (229 C3 notes): the plan's fingering
  (`holds_20261018/plan_why.py`: Chopin release_far 127, repair let go 76; Rach repair 52 = the four-stave climax, a
  2 s bass chord under leaping hands, held by the pedal in any performance). `PEDAL_HOLD` (piano.py, off: the
  fingering only protects a note until just after the last pedal change under it) and `HOLD_REACH` didn't help
  (plan-level prediction 14.7 -> 15.0 / 15.8%).
- **The judge's skin was wrong.** `action.spheres` put one sphere of radius w (the section's half-width) on each skin
  section's lowest point. At the 2:1 flat fingertip that ball is too narrow near the bottom for a level finger, and once
  the finger rolls the lowest point slides sideways and the ball reaches a key the real skin is clear of (R2 rolled 25
  deg on E5: ball 3 mm into F5, skin 2.5 mm short). Tried in turn (C3 suite): capsule per section (radius h along the
  width; too wide at the bottom, 2261 wrong), ellipse points (537 wrong, 85.0% sounded), ellipse + measured fingertip
  cap (the rig holds the u 0.9 section to the bone's end; the mesh's index tip is 7.7 / 5.7 / 3.7 mm half-width at u 0.9
  / 0.95 / 1.0, the middle finger's skin ends at u 0.96): 542 wrong, 83.1%. Then the real thing:
- **`CONTACT = "mesh"` (action.py default now):** the MPFB hands mesh (`hand_model/mpfb_skin.json`,
  `rig/export_mesh.py`: 1596 vertices per hand, weights, triangles) posed every frame by linear blend skinning with
  `Skeleton.fk` -- the mesh has only an Armature modifier, so this is what the renderer draws: checked against Blender's
  evaluated mesh on a C3 frame, mean 0.001 mm, max 0.68 mm (vertices with no forearm weight; the 55-61 wrist vertices
  with some forearm weight sit far above the keys). Vertices + triangle-edge midpoints, 0.5 mm contact spheres, points
  that never come within 8 mm of the black key tops dropped; PCHIP built per 4 s block (`_BlockPchip`); contacts in
  20 ms chunks so a run start knows the keys' current depth (a point arriving over a key above its *current* surface came
  from above). **C3 on the rendered skin: 84.8% sounded (sphere judge said 92.1), wrong 845 (1542), struck twice 135
  (240), cut 12.4%.** The sphere judge's 92.1% was flattered by fat contacts pressing keys the skin never reaches, and
  its wrong keys inflated (the "thumb shaft" fault was mostly the ball: pressing-thumb wrong 415 -> 51 with the ellipse,
  279 on the mesh, nearly all toward the fingers with the pad 5-6 mm off centre: `THUMB_AIM` 5 mm overshoots).
  All numbers from here on are on the mesh judge (`score_win.py` writes `action_mesh/`).
- C3 on the mesh judge, faults: never sounded 291 = plan pressed but the key never went down 94 (side entries: a finger
  sliding from a pressed white key straight onto a black key without rising over it, e.g. R3 onto A#4 at video 10.17,
  25 mm in 83 ms, never above 12 mm), key down but silent 87 (53 slow: the strike profile eases out into the bed, 10 ->
  80% in 83 ms p50 / 167 p90; 34 off by > 120 ms), the plan never pressed it 110 (Rach agitato 0.1 s notes the solve
  doesn't get down: plan depth 0-0.6; octave chords). Wrong 845: pressing fingers 304, pressing thumbs 279, free 262.
- **Pedal, transcribed spans:** in the Chopin (a transcription: written lengths are sound lengths) a bass note that keeps
  sounding through a broken chord wider than a hand shows the player kept the pedal *down* there; the bass-move rule
  changed it anyway and cut what the recording held. `harmony_pedal(transcribed=...)`: in those spans only the end
  clusters (the recording's own lifts) change the pedal; `action.TRANSCRIBED = ("Chopin",)`. Cuts (C3, mesh judge)
  12.4 -> **8.2%** (Chopin 16 -> 9%), song blur 0.26 -> 0.32 s (auto pedal: 0.48). k = 3 clusters would cut 6.5% but
  blur the Chopin past the auto pedal: no.
- **Key gap:** the physics' white keys were full pitch (23.5 mm); the renderer's are 22.3 mm with a 1.2 mm gap
  (`action.KEY_GAP` 0.6 mm a side now). Barely matters (C3 wrong 845 -> 847).
- **Final-judge baselines** (rendered mesh skin, key gap, transcribed pedal; `action_mesh2/`):

  | suite | V3 | C3 raw | C3 cleaned |
  |---|---|---|---|
  | sounded | **86.5%** | 85.5 | 85.0 |
  | missed | 259 | 278 | 288 |
  | wrong | 1334 | 1164 | **847** |
  | struck twice | 216 | 153 | 133 |
  | cut (pedal) | 8.6% | -- | 8.3 |

  C3's plan loses to V3 on sounded notes: all of it in the Rach agitato (video 40-50 s: V3 misses 10, C3 46).
  **Cause: `REPEAT_GAP` 0.09** shortens the 0.1 s repeated chords to ~50 ms holds (57 of 236 notes < 60 ms vs 15),
  too short to get a key past let-off. **`REPEAT_MIN_HOLD`** 0.08 (the gap never cuts a hold below 80 ms or its planned
  length): back to 15. The first C5 suite (started with the 50 ms bug) was killed and restarted with it.
- **Window results on the mesh judge** (raw solves; C3 = PX1): `APPROACH_ARC` (AA1) missed 20 -> 13 on four windows,
  wrong -9 (it was a wash on the sphere judge, which hid side entries); AA2 (stronger) worse than AA1. `REPAIR_DEPTH` 0.9
  (verify presses past let-off, not half way) missed 61 -> 49 on five windows, wrong +11. `THUMB_AIM` 2.5 mm (TA25):
  thumb-pressing catches 50 -> 39, wrong 187 -> 172, notes equal; 0 mm loses notes. `STRIKE_LINEAR` (a press never eases
  out into the bed): missed 52 -> 47, wrong -8. Wider joint limits (RL1, real-hand ranges via `RIG_LIMITS`): the
  agitato window 41 -> 40 missed (its misses are speed, not reach). Cleanup: the mesh-skin cleanup (`clean.MESH`, M1-M5)
  and pad centring (S4-S6) don't beat the soft section cleanup S2 (`FIX_PRESSING="soft"; GAIN_PRESS=400`, wrong 164 ->
  120 on four windows, notes -1): pressing catches can't be fixed with the pad pinned; they need the plan.
- **Why pressing fingers catch black keys:** in C3 the knuckles sit 18-50 mm into the keys, the PIP up to 86 mm over the
  black keys, and the end phalanx hooks down and *back* to a pad at y ~40 (the bone's end at ~38): the DIP hangs at
  y 55-68, z 15-25 over the black keys and its skin dips onto them. Pads land deeper than their targets (36-48 mm vs
  13-38): `PRESS_AXES` y 3 vs x 6. PY1 (`PRESS_AXES` y 8) running. Finger yaw is not it (catch / clean presses both
  ~15-19 deg). Holds (plan prediction, final pedal): 10.2%; per-finger `REACH_D` (0.07 / 0.045 / 0.04 / 0.045 / 0.06,
  thumb..pinky; a 5-tuple now allowed) 9.8%; `HOLD_AWARE` fingering (path-carried holds in the Viterbi) no gain: the
  remaining cuts are the Rach four-stave climax and Chopin hand moves no fingering of one hand avoids.
- **Agitato isolation (rach0, mesh judge, raw):** V3 17 missed; H1 (hover +6 mm, black-zone float, idle thumb float)
  20; H3 (+ free collisions) 20; T1 (+ `THUMB_AIM` 5 mm) 25; RG1 (+ `REPEAT_GAP` 0.09) 42; C6 (thumb aim 2.5 mm,
  `REPEAT_MIN_HOLD`) 25; **C6f = C6 + `FREE_HOVER_FAST` (0.12, 0.4)** (the extra hover fades out when the finger's next
  note is within 0.12-0.4 s) 21. C6f on the other four windows running. PY1 (`PRESS_AXES` y 8): worse (50 missed / 215
  wrong vs C4g 47 / 182): dropped. `DYN_STRIKE` (each note's strike from its written velocity; the hands' dynamics were
  flat, written vs hammer r = 0.02 in the Chopin) and a gentle `THUMB_CLEAR` (0.003, 0.3) on windows (c6/C6d, c6/C6tc).
- **12:00 the user re-prioritised:** H3 matching our video with five hand looks (skeleton, monster, robot, paw, real)
  goes first (a subagent with Codex); the piano work runs as long-form background runners
  (`~/h3/hands_loop/action_20261017/tools/after_suite.sh`, `score_loop.sh`; results in `suite_c6/SUMMARY.txt`,
  `c6/RESULTS.txt`).
- **C6d (`DYN_STRIKE`, strike time from velocity 30-140 ms, lift 3-16 mm):** Chopin 266 missed 4 -> 20 (slow soft
  presses don't sound), timing p90 29 -> 46 ms, velocity r still ~0: dropped. The judge reads hammer speed off a
  24 fps take, where a 20-60 ms strike is under two frames: dynamics need the plan's sub-frame strike shape in the
  judge, or a lift-height-only version a video can show (C6d2: lift 3-16 mm by velocity, strike 40-70 ms, running).
- **C6tc (gentle `THUMB_CLEAR` 0.003 / 0.3):** Chopin 282 100% (C6 95.7), wrong 32 -> 25; Chopin 266 91.2% (93.0),
  wrong 17 -> 15; pressing-thumb catches 7 -> 4. The hand comes forward only ~2 mm (knuckles -3.5 -> -1.3 mm, TC's
  full-strength run moved it 7-13 mm), so the palm-back placement mostly stays. Rach window pending.
- **C6 suite (final judge), cleaned with S2:** sounded **90.4%** (V3 86.5, C3 85.0; raw C6 90.1), missed 184, wrong
  **680** (V3 1,334), struck twice 120 (216), timing p90 52 ms (49), holds cut 8.4% (8.6). Debussy 77.3 (= V3), Rach
  92.3 (91.6), Chopin 90.4 (84.1). Debussy misses: a resting left-hand finger lies on a right-hand key in the opening
  and holds it (e.g. L2 on F#4 from video 4.61 to past the R1 onset at 4.79; traced with `ACTION_TRACE`, a new per-ms
  key log in action.py). **Pops 11.9/min (V3 5.0, C3 6.7; Rach 37.6).** Source: `REPAIR_DEPTH` 0.9 (RD1 windows:
  ks_rach 14 -> 32, rach0 37 -> 57 pops/min): the repair boost switches a missed note's finger weight on as a step.
  `REPAIR_RAMP` (the boost eases in and out over 0.12 s) on rach0 / ks_rach running (c6/C6r), with C6 minus
  REPAIR_DEPTH as the control (c6/C6n). Posted: band site `working-piano-real-skin` (C6 track + table).
- C6f (hover fade): 5 windows missed 40 -> 34, wrong +12; C6tc (gentle thumb arch): 3 windows wrong -9, missed +1;
  both in **C7** (suite solving). C6d2 (lift height by velocity only, 40-70 ms strikes): Chopin 266 missed 4 -> 12:
  dynamics shelved until the judge sees the plan's sub-frame strike.
- **Pops vs deep repair (rach0 / ks_rach, pops/min):** C6 60 / 40; `REPAIR_RAMP` 0.12 s 63 / 52 (no help: the pops
  are not the weight step); C6 without `REPAIR_DEPTH` 0.9 29 / 32 but missed 25 -> 35 and 7 -> 9. A pop is any finger
  joint over 15 rad/s (36 deg in one frame); in the agitato's 0.1 s chords, fully pressing every key takes about that.
  Middle settings running: `REPAIR_W` 2.5 (c6/C6w25), `REPAIR_DEPTH` 0.75 (c6/C6d75).
- **Repair strength:** `REPAIR_W` 2.5 (was 4) keeps the deep repair's notes (rach0 25 missed = C6, ks_rach 6 vs 7)
  and halves the Rach pops (rach0 60 -> 32, ks_rach 40 -> 29 /min, = no deep repair at all). `REPAIR_DEPTH` 0.75:
  in between (27 / 8 missed, 46 / 32 pops). **C8 = C7 + `REPAIR_W` 2.5**, suite solving.
- **C7 = C6 + `FREE_HOVER_FAST` + gentle `THUMB_CLEAR` (suite, cleaned): 88.3%** (C6 90.4), missed 224 (184), wrong
  692 (680); every piece down (Debussy 75.3, Rach 91.0, Chopin 87.8). The window gains (C6f +6 notes / 5 windows,
  C6tc -9 wrong / 3 windows) did not carry to the suite. **Lesson: a 12 s window moves by several notes on a
  re-solve; small window gains need a suite run before they count.** C6 stays the best candidate.
- `WHITE_FRONT_MAX` 0.035 (C6front, 5 windows): missed 40 -> 36, wrong 186 -> 190, pressing-finger catches 44 -> 49:
  the pads don't reach their targets anyway (white-key black-side catches: pads at 38-49 mm in C6). Not adopted.
- **C8 (relaunched) = C6 + `REPAIR_W` 2.5** only (the first C8 carried C7's knobs; killed at 2/24 parts).

## 2026-10-18: H3 hand looks (the leg-K recipe with five hand looks)

**Why.** The team reported that H3 does not match our blockout video. Leg K (K22/K23, 10 Oct) had shown that it does when the
job is set up right, but only with one look (chrome armour).
**Hypothesis.** The ControlNet control's *geometry* sets the category of hand H3 draws (K20/K21). So any look carried by
bone-parented geometry on the stock MPFB2 hand (scale 1.0, rig and motion untouched), with a grey plate of the same
geometry, the Canny of its normal pass as the control and a matching prompt, will follow the plate as well as the armour did.
**Setup.** Codex wrote `blockout/hand_styles.py` (`blender_piano.py --hand-style skeleton|monster|robot|paw|real`): bones
with the skin hidden; warts, ridges, talons, scales and spines; bevelled shells, hinge pins, panels and pistons; fur volumes,
toes, tufts, small claws and beans; nothing. Codex also wrote the skeleton/monster/robot/paw prompts
(`h3_tools/hand_look_prompts.json`). I wrote the real prompt (rendered while Codex was still working, to use a free GPU window).
Pipeline `h3_tools/hand_looks.py`. Shots kh111 (frames 2664-2787) and ov125 (3000-3123) of the v17 still edit, rebuilt
from `/tmp/claude-1000/suite_v17/edit_c.json` with `cameras.direct_still`. The original edit.json is gone; the rebuild matches
the K22 plates to a mean 0.1/255. kh111's first 7 frames are pinned to the shot's camera: the K22 plate had a camera cut at
frame 7. RIG_SLEEVE=0.06 as in the v17 plates. Job = K22 (fl2va, mode B, grey plate 0.88, Fun ControlNet 1.0 on the Canny
20/60 of the normal pass, 20 steps, res_multistep/beta, seed 7, audio frozen); only the plate, control and hand sentence change.
**Runs.** 10 H3 renders, one per look per shot, same seed; ~16 min each (28 min for the first, under memory contention).
**Results** (motionmatch.py inside each look's own Blender mask; flow vs the look's normal pass / vs its grey plate; fidelity.py
whole-frame edge r vs the plate):

| shot | look | H3 drew | flow_cos / err vs normal | flow_cos / err vs plate | frame r |
|---|---|---|---|---|---|
| kh111 | skeleton | skeleton hands | 0.87 / 0.66 | 0.83 / 0.68 | 0.67 |
| kh111 | monster | scaly clawed creature hands | 0.95 / 0.48 | 0.95 / 0.55 | 0.58 |
| kh111 | robot | metal robot hands | 0.94 / 0.53 | 0.93 / 0.57 | 0.60 |
| kh111 | paw | grey fluffy cat paws | 0.96 / 0.45 | 0.96 / 0.46 | 0.59 |
| kh111 | real | human hands | 0.70 / 0.70 | 0.80 / 0.68 | 0.64 |
| ov125 | skeleton | skeleton hands | 0.80 / 0.70 | 0.76 / 0.72 | 0.66 |
| ov125 | monster | scaly clawed creature hands | 0.87 / 0.64 | 0.87 / 0.64 | 0.61 |
| ov125 | robot | metal robot hands | 0.92 / 0.54 | 0.91 / 0.58 | 0.66 |
| ov125 | paw | cat paws, pink pads toward the camera | 0.91 / 0.58 | 0.92 / 0.56 | 0.58 |
| ov125 | real | human hands, dimmer, softer fingers | 0.61 / 0.79 | 0.78 / 0.77 | 0.66 |
| ref | K22 / K23 armour | steel gauntlets | 0.95 / 0.51, 0.92 / 0.56 | 0.95 / 0.49, 0.93 / 0.52 | 0.76, 0.60 |

All 10 drew the requested look on the plate's hands, with no cuts or drift (reviewed at 1 fps; no frame shifted >8 px).
None needed a second iteration. Weak spots: H3 adds motion blur on the fastest moves. On ov125 the paws turn their pink pads
toward the camera: the prompt names the beans, and the Blender beans are on the palm side, out of view. Flow agreement
tracks the geometry's texture: smooth skin (real) and thin bones (skeleton) give Farneback less to lock onto, so they score
lower although they sit on the plate in the review.
**Conclusion.** Hypothesis supported: the recipe is look-agnostic. The look comes from the control geometry + the prompt,
the motion comes from the Blender hand. Posted as the band-site section `h3-hand-looks` (cycle videos, 5-up grids, plate |
control | H3 trios, scores, recipe). Renders: `~/comfyui-h3-audio/output/ks_lipsync/hl_<shot>_<look>_00001_.mp4`; passes,
jobs, scores and videos under `~/h3/results/hand_looks/`.
- **C6 misses, what's left (cleaned suite, 184):** key down but silent 69 (45 stall at 76-91% depth, short of let-off
  at 89%: the finger is stopped because its skin runs under a black key's body; 15 blows > 120 ms off; 9 keys held
  by other skin), plan pressed but the key never went down 55, plan never pressed 60.
- **Why fingers hook:** with the MPFB hand the plan's knuckle line is `Y_KN` 7 mm into the keys (+18 mm `YOFF_NB`
  near black keys) while white-key pads aim at 20-33 mm: the knuckles sit almost over the pads, so a 9-10 cm finger
  folds into a claw: PIP out over the black keys (y 50-86 mm), the end phalanx hooked back toward the player, the pad
  landing at ~40 mm with the tip under the black keys. A pianist's knuckles sit ~4-6 cm behind the tips. Side
  candidate **C9 = C8 + `Y_KN` -0.015** (knuckles 22 mm further toward the player) on the whole suite. It changes the
  look the user chose (M2 curled, palm back): not adoptable without their eye.
- **C8 = C6 + `REPAIR_W` 2.5 (suite, cleaned): 90.3%** (C6 90.4), missed 186, wrong 665 (680), struck twice 109
  (120), timing p90 52 ms, holds cut 8.4%; pops 10.0/min (C6 11.9; Rach 33.1 vs 37.6): the windows' halving didn't
  carry. Rendering C8 for the user's eye (GPU lease piano-c8, `tools/render_after.sh`).
- **Where the Rach pops are:** nearly all *free* fingers: the idle thumb's IP joint (31 of ~70 pop frames), the free
  pinky's spread (13). **`blockout/depop.py`:** a free finger's joint over 12 rad/s gets its path Whittaker-smoothed
  over +-4 frames; press frames (and 3 before) pinned. On C8 cleaned: pops 10.0 -> **5.05/min** (V3 5.0; Rach
  33 -> 15), broken / crossed unchanged; but wrong 665 -> 739, sounded 90.3 -> 89.9 (smoothed free fingers dip into
  keys). Order solve -> depop -> cleanup -> physics running.
- **Order solve -> depop -> cleanup S2 (C8):** LAND 3 frames: sounded 89.9% (C8 90.3), wrong 679 (665), struck
  twice 111, holds cut 8.1%, **pops 5.48/min** (C8 10.0, V3 5.0; Rach 15.9). LAND 5: 90.1% / 673 / pops 8.0 (Rach 29):
  many pops are in the strike's fast approach. **C8d = C8 + depop (LAND 3) + S2**: vs V3 more notes (89.9 vs 86.5%),
  half the wrong keys (679 vs 1,334), about as smooth (5.5 vs 5.0 pops/min). Rendering C8 and C8d for the user.
- **C8 render** (`suite_c8/render/suite_show.mp4`, 350.6 s, physics key depths, C8 hands' audio), reviewed at 1 fps
  (12 contact sheets): plays through, no broken geometry, no missing frames, cuts only at the edit's camera changes.
  Seen: a hooked right-hand finger (0:42, the claw from the knuckle line, C9 tests it); fanned pinky / thumb on the
  right hand (1:07-1:14, 4:17-4:19); the hands crowd each other in one register (2:36-2:44, 5:32). C8d render queued
  (lease piano-c8d).
- **C9 = C8 + `Y_KN` -0.015 (suite, cleaned): 90.8%** (C8 90.3), wrong 690 (665), struck twice 126 (109), pops 9.7,
  crossed 1.06% (0.80). But it did not move the knuckles back: pressing fingers' MCP y went 16 -> 26 mm (deeper),
  PIP 45 -> 55: the solve's reach pulls the hand in whatever `_centre`'s target says. Inconclusive for the hooked-finger
  idea (it needs a direct knuckle-depth control); not adopted.
- **C8d render** (`suite_c8d/render/suite_show.mp4`), reviewed at 1 fps: identical to C8 at that rate (the de-pop
  pass only eases idle-finger snaps); same notes (0:42 hook, fanned right pinky/thumb, 5:32 crossing). Posted: band
  site top section `piano-c8-render` (C8d whole suite, C8 | C8d Rach 0:40-0:56 side by side, C8 whole suite,
  V3 / C6 / C8 / C8d table). Not the default yet: the user's eye decides.
- **C8d faults (wrong 679 + struck twice 111):** pressing fingers 225 (white key catching a black key 110, pad ~1 mm
  off centre; white key catching the next white 94, pad ~5 mm off), pressing thumbs 152 (next white key toward the
  fingers 115: the flat thumb's shaft), free travel/idle 126 (C8 108: the de-pop's smoothing adds ~18), landing /
  lifting ~110. Suites running (the C8d recipe, `after_suite.sh` DEPOP=1): **C10 = C8 + `BLACK_SHY` 3 mm** (a white
  key with a black key on one side only, C E F B, is struck 3 mm toward its open side); **C11 = C8 + gentle
  `THUMB_CLEAR`** (alone; C7 had it mixed with the hover fade).
- **C10 = C8 + `BLACK_SHY` 3 mm (C8d recipe): 90.1%** (C8d 89.9; Debussy 76.3 -> 79.4), wrong 692 (679), struck
  twice 126 (111), pops 4.5/min, but crossed tips 1.54% and white-key-on-black catches 116 (110): the shift didn't
  reach the catches it aimed at (the solve doesn't land pads to a few mm) and it crowds neighbouring fingers. Not
  adopted. C8d stays the candidate.
- **C11 = C8 + gentle `THUMB_CLEAR` (C8d recipe): 89.4%** (C8d 89.9), wrong 650 (679), struck twice 122 (111), pops
  4.45. Pressing-thumb catches 154 (C8d 152): the arch does not reach them (the drop is free fingers, noise). The
  MPFB thumb's end phalanx is ~27 mm wide near its base (half-width 13.5 mm at u 0.6), wider than a 22.3 mm key top:
  a flat pressing thumb overlaps both neighbours. Fixing it means pressing with the narrower tip (more IP flex, a
  thumb effector nearer the tip): a pose / rig decision (the hand stays the stock MPFB at scale 1.0). Not adopted.
- **C12 = C8 + `KEY_ROUND` (solver collisions with the judge's rounded skin, normal weight; C8d recipe): 90.3%**,
  wrong **632** (C8d 679), struck twice 131 (111), pops 6.4 (5.5; Rach 21 vs 16), crossed 1.33% (0.77). Fewer wrong
  keys at the cost of visibly worse motion (crossings, pops): not adopted. C8d stays the candidate.
- **Early touches (C8d, 70):** another finger brushing the key before its striker 32; the striker itself grazing its
  key 0.12-0.37 s early on the way in 25; same finger / same key quick repeats 13. No single knob.
- **Thumb at the tip:** the rig's pad is the soft-min lowest skin over u 0.55-1.0 of the end bone (the `eff_local`
  "corner" is unused). `rig/ik.py RIG_THUMB_EFF_US` (env) restricts the thumb's to the tip end; **C13 = C8 +
  `RIG_THUMB_EFF_US=0.85,1.0`** (C8d recipe) solving: aimed at the 152 pressing-thumb catches (a flat thumb's ~27 mm
  base over both neighbours). A rig / pose change: the user's call even if it wins.
- **C13 = C8 + thumb pad at the tip (`RIG_THUMB_EFF_US=0.85,1.0`; C8d recipe): 90.8%** (C8d 89.9; Debussy 80.4,
  Rach 92.7), wrong 713 (679), struck twice 121, pops 5.1 (5.5), crossed 0.69% (0.77), broken 0.21 (0.13). Pressing-
  thumb catches 184 (152): worse, so the flat wide base was not the whole story (the tilted thumb still lands on its
  neighbours). More notes and smoother but more wrong keys. **C14 = C13 + `KEY_ROUND`** (C12 cut wrong keys 47 but
  crossed more; C13 crosses less) on the suite.
- **Stronger cleanup on C8d's de-popped take** (S2 + `CLEAR` 2 mm, `GAIN_FREE` 1000, 6 passes): 90.0%, wrong 688
  (C8d 679), struck twice 120 (111), pops 6.4 (5.5). The cleanup is saturated: bigger corrections add pops and don't
  remove catches.
- **C14 = C8 + `KEY_ROUND` + thumb pad at the tip (`RIG_THUMB_EFF_US=0.85,1.0`), C8d recipe: 91.1%** (best; C8d
  89.9, V3 86.5), missed 170, wrong **602** (best; C8d 679, V3 1,334), struck twice 110, timing p90 50 ms, holds cut
  8.6%, broken 0.09%, pops 6.1 (C8d 5.5), crossed 0.99% (0.77). The two changes complement: KEY_ROUND's fewer
  catches, the tip press's notes and fewer crossings. New best candidate; rendering it (lease piano-c14). The thumb
  change is a rig decision for the user.
- **C14 render** (`suite_c14/render/suite_show.mp4`) reviewed at 1 fps: plays through, nothing broken or missing; the
  0:42 hook is milder than C8d's; 5:32 crossing and the 4:17 fan unchanged. Posted: band site top section
  `piano-c14-render` (whole suite, C8d | C14 Rach side by side, V3 / C8d / C14 table). Awaiting the user's eye; the
  thumb-tip press (`RIG_THUMB_EFF_US`) is a rig decision.
- **Stricter de-pop on C14** (`LIMIT` 10 rad/s, `HALF` 5 frames; was 12 / 4): 91.1%, wrong 604 (602), struck twice
  113, pops 6.07 -> **5.65**, crossed 1.01%. Free smoothness: now the depop defaults.
- **C14 fault picture** (de-popped, `tools/faults.py`, `wrong_detail.py`, `wrong_geom.py`, new `pad_bias.py`): wrong
  601 = pressing fingers 214 (thumb-side 149 vs pinky-side 61), free fingers 225 (travel/idle 113, landing 58, lifting
  54), pressing thumbs 118 (91 the next white key toward the fingers), struck twice 113 (early touch 44, timing 36).
  558 of the wrong blows have skin > 1 mm in the key at the take's own frames (not between frames). Pressing thumbs'
  pads sit 1.3-2.8 mm toward the fingers (p50; black keys 2.3-2.8): `THUMB_AIM` 2.5 mm was tuned for the old
  corner pad; with the tip pad (`RIG_THUMB_EFF_US`) it moves the whole thumb toward the fingers.
- New plan switch `RELEASE_ARC` {"lead", "hold"} (piano.finger_tip): a released finger rises to its hover height in
  the first `lead` share of the release and starts sideways only after `hold` (default None = one straight line from
  the key bed to the hover spot, which drags a low finger across its neighbours when the hand shifts). Suites
  running (6 parts at once each, C8d recipe + new depop): **C15 = C14 + `THUMB_AIM` 0**; **C16 = C14 +
  `RELEASE_ARC` {lead 0.5, hold 0.25}**.
- **The cleanup is saturated (C14's de-popped take, whole suite, `tools/cl_suite.sh`; base 604 wrong, pops 5.65,
  crossed 1.01%):** `CROSS_GUARD` off 599 / 5.73 / **1.68%**; `TAPER` 1 frame 602 / 5.73 / 1.01; `W_HI` 500 (blend
  holds the fix) 602 / **9.07** / 0.99; `SMOOTH_LAM` 40 596 / 7.36 / 1.02. ~6,000 of 16,830 hand-frames still have
  skin in a key after 4 passes whatever the blend. `tools/fix_probe.py`: the per-frame re-solve clears 43/49 free-
  finger and 116/176 pressing-thumb violations but only 41/90 pressing-finger ones; and `tools/model_vs_mesh.py`: at
  the wrong blows the rig's skin model sees the same penetration as the mesh (free fingers p50 4.2 vs 4.7 mm), so the
  model is not blind. What stays is pinned: pressing / landing fingers (the cleanup keeps their pad paths to keep
  the strikes).
- **Landing catches are leaps (`tools/landing.py`):** a free finger's catch < 0.15 s before its own note has its pad
  26 mm (p50) off its key and 7.6 mm below the key-top line 80 ms before the onset: the hand is still travelling
  low. 95 of 263 free-finger catches happen with the wrist moving > 0.3 m/s. Two finds: (1) **`LEAP_V` is defined
  twice in piano.py** (0.80 m/s for the hand roles, then 1e9 for the wrist smoothing), so `leap` is always 0 and the
  over-hand leap arc `LEAP_DZ` / `LEAP_BUNCH` have never fired; left as is (V3/C14 are tuned on it), noted in the code.
  (2) New plan switch **`LEAP_LIFT`** (m; `LEAP_LIFT_MIN` 4 cm, `LEAP_LIFT_FULL` 20 cm): across each gap between a
  hand's last let-go (as played, release_far's early lets-go) and its next note, a hand that travels far arcs up
  (sine, peak mid-gap), free fingers with it; probe on the 154.4 s left-hand leap: +25 mm mid-way, back by the
  landing. **C17 = C14 + `LEAP_LIFT` 25 mm** queued after C15/C16.
- Rach climax (~120.5 s, ~128.4 s): a few notes go to a thumb 20-36 cm from its key (missed anyway, their
  neighbours caught on the way): fingering, not motion.
- Pad guard variants on C14's take (whole suite): `PAD_GUARD` (0.5, 4 mm) 91.1%, wrong 599 (604), struck twice 111
  (113), pops 5.56 (5.65), crossed 0.99%, pressing-thumb catches 118 (120): a hair better, inside what a recipe change
  is worth; `PAD_GUARD_DIR` (slides toward the centre line free) wrong 618: worse. The cleanup stays S2.
- **Missed notes on C14 (171, `tools/missed.py`):** key fully down but no matched blow 59 (Chopin 46), plan pressed
  but the physics key not down 46, no finger assigned 41 (Rach climax), the solve never pressed 25. Traces
  (`ACTION_TRACE`): B4 at 131.54 (written vel 88) goes down over ~140 ms decelerating, blows 144 ms late (unmatched:
  missed + a wrong blow); A4 at 44.70 (pp) passes let-off at 31 mm/s, too slow to throw the hammer to the string.
  **Cause: the solve slows every strike.** The plan strikes from 12-25 mm in ~60 ms (~0.5 m/s at the bed); the tip
  spring (`SPRING_TIP`), the joint-path smoothing (`FINGER_REFINE`: strike frames weighted `FINGER_PRESS_W` 8 against
  `FINGER_LAMBDA` 6) and the second solve's prior to that path stretch it over 4-5 frames: key speeds 0.04-0.12 m/s,
  hammer vs written velocity r 0.07 (Chopin) / 0.10 (Rach) within a piece. (So the C6d note that 24 fps can't show
  a strike was wrong: one frame can, the solve smooths it away.)
- New plan switch **`STRIKE_KEEP`** {"raw", "pin"}: frames of a strike (lift to key bed, +-1 frame) take the plan's
  unsprung fingertip height and are pinned at "pin" in the joint-path smoothing. Window 130.6-132.2 with {1.0, 100}:
  R5's B4 fully down one frame after its onset (C14: 140 ms later), blow at -10 ms, hammer 2.0 m/s. A crisper
  strike on screen: a look call for the user. **C19 = C14 + `STRIKE_KEEP` {1.0, 100}** queued with C17 (6 parts
  each), then C18 = C14 + `PRESS_AXES` x 12 (pressing fingers lean toward the thumb side: 149 vs 61 catches)
  (`tools/queue_c17_19_18.sh`).
- **C15 = C14 + `THUMB_AIM` 0 (C8d recipe, new depop): 90.8%** (C14 91.1), missed 176 (171), wrong **548** (C14
  604 de-popped), struck twice 119 (113), timing p90 54 ms, holds cut 8.2%, pops **5.22** (5.65), crossed 1.03%.
  Pressing-thumb catches 120 -> 75, free thumb 42 -> 29: with the tip pad the 2.5 mm aim toward the fingers only
  pushed the thumb onto the next key. Best wrong-key count yet; notes level (Debussy +3, Rach -7: re-solve noise
  scale).

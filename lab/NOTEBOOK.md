# Piano-hand lab notebook

## 2026-10-11: no-crossing ownership and wrist planning

**Requirement and hypothesis.** The user's minute contains unwanted hand
crossing. Its saved roots invert in five intervals at playback 47.917–52.917 s:
40/720 samples, 3.333 seconds, worst R−L gap −251.803 mm. Original v19 part 17
has a lone right-hand bass note at 293.193 s; the learned reference attracts
that otherwise idle hand throughout the earlier rest. Window-dependent staff
splitting is a second instability. Persistent held-note ownership, restrained
first-note anticipation and hard bimanual root bounds should remove these
inversions while retaining all original events, dimensions and budgets.

**Change.** `NO_HAND_CROSS` is on by default in this isolated test branch.
Allocate the full performed score before extracting a window. An event keeps
its hand until its original release; new notes use an ordered register cut.
Retain all 2,046 events and report 318 redistributions and 610 conservative
capacity/span conflicts; zero held-register inversions over every score
transition. The initial learned approach begins 350 ms before the first note,
rather than several seconds early. Whole-hand solves have hard root-x bounds.
After smoothing, project both wrist roots into their ordered halfspace and
exact physical arm workspaces before solving fingers. Minimum root gap is
60 mm. This is a style/planning constraint, not a skin or force certificate.
Other curl/wrist-carry switches remain off in this matched experiment.

**Matched result.** Complete-budget 287–295-second stress runs, size 1, 24 Hz,
same E36 settings, original note ledger and frozen input/code hashes:

| Policy | Crossed samples | Minimum root gap | Onsets | Original held-key coverage |
|---|---:|---:|---:|---:|
| Legacy |72/192|−31.526 mm|21/31|18.55%|
| Ordered |0/192|166.605 mm|27/31|45.70%|

Both retain exact arm lengths and the 12 rad/s joint-speed bound. Matched
ordered maximum wrist speed falls 1.310→0.711 m/s. This is a measured speed,
not a newly enforced wrist-velocity cap. No onset windows are censored.

**Fresh minute and limitation.** Eight independently solved 7.5-second chunks,
239.9–299.9 s, unaltered poses, original size 1 and audio: zero inversions in
1,440 samples; minimum wrist gap 97.897 mm. Mean long-finger MCP heads also
remain ordered, minimum gap 59.516 mm; these are skeleton landmarks. Exact
arm-length error is floating-point noise, 1.11e−13 mm. However all seven
chunk joins exceed the joint-speed gate, maximum 40.366 rad/s; one wrist
jumps 111.643 mm at a join. Onsets 250/283, held-key coverage 45.67%. The older
minute reports 263/283 and 50.83%, but differs in controller/bounds/chunking
and is not a causal matched baseline. Full rendered motion is not accepted.

**Actual skin.** Five independent production Blender level-2 snapshots,
both quad diagonals and CPU parity below 0.0012 mm. No sampled interhand
surface intersections, but self folds and instrument/key-support failures
remain. Ordered stress frame 36 has 117 self pairs and R penetration 0.186 mm;
frame 72 has 61 self pairs and R/L penetration 0.285/0.340 mm. The fresh
minute's closest-hands frame 1079 has 392 L self pairs and L penetration
6.533 mm (1,831 instrument triangle intersections). No complete geometry,
positive-clearance, body/costume, force or continuous-time acceptance.

**Reviewable artifacts.** Fresh grayscale minute, 768×432/12 fps/720 frames,
and matched eight-second comparison, 1536×432/12 fps/96 frames, original audio,
full decode checks and manual fixed-view samples. `hand_order_20261011.json`
retains every controller outcome, original ledger metrics, join failures,
actual surfaces, visual notes and input hashes. 150 tests pass. A report-only
NumPy boolean serialization failure was fixed without changing any solved
pose; its failed protocol log is retained. Next: one persistent whole-piece
plan/state to eliminate chunk resets, with full skin/contact constraints and
restrained free posture. The no-crossing gate is not full rig realism.

## 2026-10-11: wrist-carried curl and isolated reach, all four outcomes retained

**Hypothesis.** Unused digits should stay gently curled and controlled near
the keys; a solo stretch should be carried by the wrist before lateral finger
bending. The old controller deliberately lifted, lengthened and fanned idle
digits on an under-damped spring, and considered the first chord release to
free the hand. Implemented two independent opt-in factors in the actual
controller: FK rest targets with critically damped preparation and phase-faded
free posture gain; isolated lateral wrist alignment with all-slice held-note
tracking and removal of reach-based early release. Original score, dimensions,
joint bounds, arm lengths and 12 rad/s joint speed cap retained.

**Experiment.** Four complete-controller runs on the same performed Knights
Suite 247–253-second passage, 24 Hz, hand scale 0.7284, unchanged E36 budgets.
Identical input/code fingerprints verified after all runs. Every original
note is retained in a separate ledger; existing fingering/event truncations
are exposed rather than counted against the modified schedule.

| Controller | Onsets (key ≥50% within 250 ms) | Original held-key sample coverage | Shortened/missing scheduled holds | R unused-middle rest deviation RMS |
|---|---:|---:|---:|---:|
| Baseline |32/38|42.52%|28|63.00°|
| Free control |33/38|40.13%|28|8.17°|
| Wrist carry |28/38|48.73%|20|15.58°|
| Both |29/38|50.00%|20|8.04°|

The unused-middle comparison uses the same 98 R/109 L inactive frames in all
four runs. L deviation is 9.70→5.49° for baseline→both. On identical strongly
active isolated-note frames (18 R, 8 L), mean horizontal pad excursion from
the relaxed pad at the achieved wrist falls 17.28→4.20 mm R and 12.21→8.65 mm L.
These are limited kinematic measurements, not muscle/force validation.
No joint-bound or speed-cap violations; maximum arm-length error is floating
point noise (1.11e-13 mm). The velocity cap is an engineering bound and does
not by itself establish natural movement. Pad control-point spacing is also
reported separately from actual skin clearance.

**Independent surface result.** Eight actual Blender level 2 snapshots,
unchanged LBS weights/deformation, verified CPU/topology parity, both quad
diagonals at identical frames 48/108. Self-crossing pairs: baseline 212/456,
free 68/477, wrist 135/568, both 69/526. Both introduces L instrument penetration
3.427 mm at frame 48 and retains R 4.110 mm/L 1.283 mm at 108. None accepted.
Improved free posture can coexist with worse playing-pose folds and note
onset regression. Complete trajectory/body/costume, positive clearance,
containment and continuous collision checks remain open.

**Conclusion.** The user's composure correction has a measurable effect on
achieved digits and isolated reach. It does not yet produce acceptable
playing. Keep both switches off by default; next work must couple held-note
support and wrist planning with the actual-surface constraints. Do not
promote the better curl as a complete rig fix.

141 tests pass (38.923 s). Native research graph completed all five roles,
episode 39148ff9245c48bc6f6163abb5db398e44caa098622ecb837c933f05ff1f4db8,
methodology-only PASS. Retained sandbox DNS failure and successful approved
rerun. Full results: `lab/realism_results/composure_20261011.json`; protocol:
`lab/COMPOSURE_PROTOCOL.md`. Fresh six-second grayscale side-by-side render,
1536×432 at 12 fps, audio matched to identical performed notes, 72 frames,
complete decode validated, SHA256
deb8b3f95b5ee37ed2d43669dd4f3018c70bc9e60c2f6a039f416f0c9c2f4ac6.
Three visual checkpoints per controller reviewed. Existing band site now
contains this comparison, all outcomes, the older 60-second performance and
the unchanged drummer diagnostics. Busy drummer stress patterns remain 0/12
within their proxy scope; no new drummer fix is claimed in this experiment.

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

### Rendered-skin contact research (2026-10-11): poses improve, motion fails

**Hypothesis:** finger-tube and vertex-only checks miss the rendered skin and
triangle crossings. Keep anatomy and engineering speed bounds fixed, and
compare clipped-point witnesses with whole-triangle constraints on the same
three adjacent frames at hand scale 1.

**Result:** accepted hand poses improve from 4/6 to 6/6; independent Blender
surface/topology parity passes within 0.0012 mm. However, 18/21 interpolated
observations at 240 Hz collide, up to 5.5905 mm. The original saved overlap,
Debussy and wide-chord probes have actual-skin instrument intersections on
77/86, 86/86 and 80/86 frames, respectively. These are different geometry
metrics from the older key-sync/tube checks and are not directly comparable.
92 unit tests pass. Drummer tip-radius and antipodal-rotation regressions are
covered, but its complete rendered body/grip trajectory is not validated.

**Conclusion:** the diagnostic pose solver cannot yet produce an approved
performance. Key 45 starts returning while its scored note is still held;
requiring its actual sustained skin support exposes a rejected numerical
failure. Free fingers also look too straight and the posture looks strained.
Coupled contact/key/wrist planning, full self/interhand/body clearance,
continuous collision checking and human form review remain necessary.
The native research graph reviewed the experiment protocol, not the rig:
its proposed 258-input paired ablation has not yet run. Full CDL/KS musical
validation and the calibrated 0.728 hand scale have not been rerun for this
skin solver. Details and source/result fingerprints: `REALISM_20261011.md`
and `realism_results/skin_contact_motion_20261011.json`.

### Held skin contact (2026-10-11): finite contact improves support; motion still fails

**Hypothesis:** bounded feasibility initialization improves restoration, and a
contact must remain on the finite exposed top, not merely near its height.
In the controlled frozen L25 pair, only the initialization budget changes;
input/source/score hashes match. The first seeded attempt failed and is retained.

**Result:** the corrected conditioned pair accepts the bounded-start candidate
and rejects the original-start candidate (5.714 mm penetration). Independent
support checks reveal that the earlier six-pose window allowed contact points
to slide off key fronts. The new finite-top gate accepts six poses and all 14
requested anchors; all exported instants independently retain required holds.
Between those instants, 18/21 observations still collide (3.3281 mm maximum),
and 10/74 required held-note observations lack support (earlier loose window:
43/74). The six-pose comparisons include conditioning/retention changes and do
not isolate footprint constraints alone. Blender parity passes within
0.001238 mm. 97 tests pass.

**Conclusion:** contact validity is stronger, but the performance remains
unacceptable. The right hand looks overcurled near the key fronts and the left
free fingers remain too straight. Bounded joint/wrist speeds do not ensure a
clear path or stable held notes. Replan contact placement and connected motion
using the human calibration; complete geometry/CCD and full CDL/KS are pending.
See `REALISM_20261011.md` and `realism_results/held_contact_20261011.json`.
# 2026-10-11: mechanical/path and drummer-grip research checkpoint

Hypothesis: coupled neighboring contact solves improve sampled feasibility;
frozen original poses may still prevent a valid path. Separately, moving piano
solids and cross-hand drummer grip capsules need full accounting.

Exploratory path: 18/21 colliding observations become four, worst depth
1.263942 mm; four required held-pitch observations remain unsupported. 64/72
local corrections pass, but the complete path is rejected with no exported
candidate. Assigned held-key commands and neighboring poses both change in
this treatment, so no single-factor causal claim follows.

Mechanical ablation: identical key configurations show up to 6.25 mm key/key
and 8.952609 mm key/keybed interference in the old rectangular model. An opt-in
notched-key/bed prototype passes these sampled checks and 63 varied stroke
configurations, with Blender parity within 0.001237 mm. It retains the old
short hinge; calibrated action/forces are still open. Shape-only hand motion
is unchanged: 18/21 collisions and 10/74 unsupported held observations.

Drummer: complete cross-hand finger/stick proxy checks retain 10/10 demo and
16/16 roll acceptance. All twelve stress patterns remain rejected, including
eight with newly recorded interhand finger collisions and seven with fingers
against the opposite stick. Retained grip has 6 mm minimum within-hand spacing.
These remain kinematic proxies; full renderer/body/cloth/shell/feet/force and
continuous-time checks are pending.

Native graph `skin-contact-trajectory` completed all five nodes and approved
the proposed experiment method (episode `276a908d…`, complete receipt in
`lab/realism_results/skin_trajectory_review_episode.json`). A matched frozen
versus movable-interior-pose experiment is currently running. Additional
movable-node solves are accounted for; no result is asserted before completion.
Numeric solver checkpoints are unvalidated and cannot be exported as passing
animations. The path gate now includes support, spacing, rigid arms, root
consistency, key travel, mechanical interference and all engineering rates.

Measured scale 0.7284 is opt-in; the existing planner intentionally defaults
to scale 1. Earlier references to a calibrated production default were
incorrect. Fresh measured-size generation remains required. Visual right-hand
form still fails. 105 tests pass in 8.612 s. Code checkpoint `9a95cec`; complete
detail in `lab/REALISM_20261011.md` and the existing band site. Goal active.

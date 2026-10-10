# E24 sideways misses: diagnosis (83 misses; 66 black)

Method: for each miss, rebuilt the hand's notes as animate() does (window notes, split_hands, slices, fingering, finger_events, release_far, _place). The planned finger matches the run's finger for 83/83 misses. Per miss I computed: cluster span (notes of that hand within 50 ms), hand-centre travel and speed from the previous slice, held keys still alive after release_far and their offset from the new hand position, the other hand's idle time and distance, the best free alternative finger, and the in-slice residual `resid` = contact_x - (hand_x + sgn*HOME[f]).

Rule order (first match wins): a1, a2, d, c, b, e, e2, x. Thresholds are in the "How I tested each class" section below.

## Counts

| class | total | Debussy (6) | Rach (30) | Chopin (47) |
|---|---|---|---|---|
| a1 impossible: chord wider than HAND_SPAN, or hand leap > 1.5 m/s | 32 | 0 | 12 | 20 |
| a2 within HAND_SPAN but wider than the rig's finger layout (octave chords) | 13 | 0 | 10 | 3 |
| b fingering choice | 9 | 1 | 1 | 7 |
| c hand assignment (other hand free and nearer) | 4 | 0 | 0 | 4 |
| d held-key pin (held key 20-33 mm off, below REACH_D 40 mm) | 4 | 0 | 0 | 4 |
| e black-key aim (black, plan fine, abs dx < 20 mm) | 7 | 1 | 4 | 2 |
| e2 marginal (abs dx <= 26 mm) | 6 | 1 | 0 | 5 |
| x unexplained | 8 | 3 | 3 | 2 |

Overlaps: 14 of the 32 class-a1 notes have the other hand within 190 mm (reassignable, so also class c). 23 of 32 a1 misses have abs dx >= 60 mm (the big errors); all other classes are mostly 14-55 mm.

## Findings

### 1. Octave chords: the hand rig cannot open as wide as the planner thinks (a2, 13 misses, plus most of e and x in Rach)
The planner accepts an octave (thumb-pinky = 165 mm, below HAND_SPAN 0.19), but `HOME` places the fingers only 120 mm apart (-0.062 to 0.058).
- `_place` centres the hand on the mean of (contact - HOME), so the thumb and pinky each fall about 22 mm short. The residual is -22 to +27 mm.
- The measured dx is almost exactly -resid (mean |dx+resid| = 7.8 mm).
- That explains the sign pattern: R pinky negative, R thumb or index positive, i.e. inward.
- 12 of the 13 a2 misses are on a black key (the pinky on C#4, F#3, A#3 and similar). A black key tolerates only +/-8.9 mm.
- It is worst in the Rach 95-135 s block (4-note octave-plus chords, span 164-165 mm).

### 2. Chords and leaps no hand can do (a1, 32 misses; 23 of them with abs dx >= 60 mm)
- 30 of 32 are clusters wider than 0.19 m, where SPLIT_WIDE splits into sub-chords with identical onsets. dt between sub-chords is 0.00-0.02 s, so the hand would have to jump 150-370 mm in under 20 ms.
- Examples: Rach 120.55 and 128.50 are 6-note bass chords per hand, C#1..E4 (529 mm) in L and C#3..G#5 (494 mm) in R. 381.86 has R on C#7 and C#3 (658 mm) within 20 ms.
- 14 of the 32 sit within 190 mm of the other hand. The staff split puts low notes (MIDI 49-65) in R although L is right there, e.g. 381.88 C#3 in R with L 10 mm away, and 274.13 D#3 in R with L 36 mm away.
- 2 are pure fast leaps: 190.76 C#4 f2 at 2.4 m/s over 169 mm in 70 ms, and 197.66 C#4 at 3.8 m/s over 228 mm in 60 ms.
- The scorer allows 0.25 s from onset to a key at 50% depth, so staggering sub-chord motion by 0.1-0.2 s is legal for scoring.

### 3. Black keys with a tiny error (e + e2, 13 misses)
Planned offset is 0-26 mm; the pad centre lands 9-26 mm off the black key (tolerance 8.9 mm). This is mostly fixed by item 1, since 4 of the 7 e misses are 4-note octave voicings.
Singles: 245.01 C#5 f5 dx 9, and 291.61 D#3 f2 dx 9 with plan offset 0. For these, the pad centre sits on the lateral edge of a narrow black key.

### 4. Fingering (b, 9) and held keys (d, 4), both Chopin-heavy
- b: pinky or ring on a black key when a free, nearer finger exists, e.g. 75.70 R C#4 f5 (23 mm off, f4 would be 5 mm); 223.29 and 321.59 R G#5 f2 after A5 f1 (38 mm vs f1 12 mm); 197.60 R B5 f4 after C#6 f1 (127 vs 35 mm). The Viterbi hand-travel term `abs(hb-ha)/WHITE_W*(0.3+0.05/dt)` is weak compared with the SPAN costs.
- d: a held finger is up to 33 mm off the new hand position but release_far only releases beyond REACH_D = 0.04 m. Examples: 264.60 L G#3 f5 with C4 f4 held at 31 mm; 254.33 L A2 f5 with G#3 f1 held at 33 mm; 263.27 L G#2 f5 with F#3 f1 held at 21 mm; 352.49 R E5 f5 with C#5 f3 and D#5 f4 held at 23 and 16 mm.

### 5. Unexplained (x, 8)
Debussy 54.34/54.35, 67.41 (x2), 89.03/89.04 and Chopin 253.15: planned residual <= 15 mm and travel about 0, yet dx is 33-124 mm. The other hand is close (71-175 mm, active). This is probably hand-hand avoidance or the under/over role logic in `_close_roles`, or IK/reach limits. I did not verify the mechanism.

## Examples (time, hand, note, finger)

**a1**
- 120.55 L C#1 f5 (dx 365): a 529 mm chord, 377 mm hand jump from the previous slice.
- 120.56 R C#3 f2 (dx 420): the 494 mm R chord, with L 115 mm away. A pianist would roll or split between the hands.
- 128.50 L C#2 f1 (dx 363); 128.52 R C#3 f1 (dx 364).
- 381.88 R C#3 f2 (dx 198) next to C#7 in the same hand. The L hand is 10 mm from C#3 and should take it.
- 274.13 R D#3 f2 (dx 185) with D#5 in the same hand and L 36 mm away. L takes D#3.
- 190.76 R C#4 f2 (dx 75) after G#5 in 70 ms.
- 353.61 R A4 f1 (dx 150) after D#7, 319 mm in 40 ms.
- 222.07 R F4 f2 (dx 171) after B6, 363 mm in 240 ms.

**a2**
- 97.03 and 104.80 R C#4 f5 (dx -26, -28): octave chord C#3 f1 ... C#4 f5, resid +23.
- 95.01 and 107.23 L F#3 f5 (dx +19, +14): resid -23.
- 112.13 L A#3 f5 (dx 27); 127.01 L F#3 f5 (dx 13); 113.62 L C#4 f5 (dx 12).
- 129.51 L F#4 f1 (dx -14) and R D4 f1 (dx 17) in simultaneous octave voicings.

**b**
- 75.70 R C#4 f5 where f4 is 5 mm off.
- 197.60 R B5 f4 where f1 is 35 mm off versus 127 mm.
- 223.29 and 321.59 R G#5 f2; 352.59 R F#5 f2 (alt f5, 35 vs 129 mm).
- 369.54 L F#2 f3 (dx 51).
- 109.69 L D#4 f4 (alt f2 16 vs 50 mm).

**c**
- 259.32 L G#4 f2 (dx -104), cluster B3/G#4 with R 12 mm from G#4 and idle.
- 263.47 L F#4 f3 with R 35 mm away and idle 1.2 s.
- 290.58 L D#2 f1 (dx 29), 327 mm hand move, other hand idle 2.6 s.
- 369.93 L C#4 f1 (dx -20), 197 mm move, R 19 mm away.

**d**: 264.60, 254.33, 263.27, 352.49 as above.

**e**
- 77.26 L D#3 f5 (dx -18)
- 83.30 R F#5 f5 (dx -12)
- 132.57 and 133.09 L C#4 f1 (dx -9, -9)
- 109.70 L D#5 f1 (dx -14)
- 245.01 R C#5 f5 (dx 9)
- 291.61 L D#3 f2 (dx 9)

## Ranked, switchable changes to piano.py (all default off)

1. **ROLL_STAGGER / reach-based roll timing.** In `slices()` (SPLIT_WIDE branch), or in `hand_track`/`finger_events`, give later sub-chords of a split chord a motion-time offset: `ROLL_V = 1.5` m/s, set offset = hand travel / ROLL_V, capped at about 0.2 s (inside the scorer's 0.25 s window). Apply the same minimum-dt rule for single-note leaps faster than 1.5 m/s. Targets a1: up to 32 misses (about 39%), including nearly all the 100-400 mm ones.
2. **REBALANCE_HANDS in `split_hands()`.** Param `reach_split=False`. After the staff split, for each onset cluster whose span in one hand exceeds HAND_SPAN, or a lone outlier closer to the other hand's centre by more than 60 mm while that hand is free (>= 0.25 s idle or span-compatible), move the note to the other hand. Mirror `playable()`'s anchor logic. 14 of the 32 a1 notes and 4 c notes have the other hand within 190 mm.
3. **HAND_OPEN (finger-spread scaling) for octaves.** Params `HAND_OPEN=False`, `OPEN_MAX=1.45`. In `hand_track` compute per-slice s = clip(key span / (HOME[4]-HOME[0]), 1, OPEN_MAX), then use `HOME[f]*s` in `_place`, `release_far`, and the `loc = sgn*HOME[f]` line in animate (the finger-home offset). Fixes a2 (13) plus a share of e/e2 (about 8 more). Alternative or complement: make `_place` weight black-key contacts (`PLACE_BLACK_W=1.0`, try 3.0) so a wide white key absorbs the unavoidable error. A hard `OCTAVE_SPAN_MAX` in `fingering` would not help, since the octave is in the score.
4. **Smaller fixes.**
   - d: `release_far` D (REACH_D 0.04 -> 0.02, env var already exists) frees the 4 pinned cases.
   - b: add `TRAVEL_W` (default 1.0, try 3.0) to the Viterbi term `abs(hb-ha)/WHITE_W*(...)`, and reduce the f1 black-key penalty from 3.0 (`BLACK_THUMB_PEN`) so the nearest free finger wins.
   - x: investigate hand-hand avoidance (COLL_ON is False by default, so check `_close_roles`).

## How I tested each class
- a1: cluster span > 0.19 m, or previous-slice hand speed > 1.5 m/s within 0.3 s.
- a2: |resid| >= 15 mm, |dx + resid| <= 14 mm, cluster spread > 110 mm.
- d: held key (still alive after release_far) 20+ mm off the new hand position.
- c: other hand idle > 0.25 s and nearer to the note by > 60 mm.
- b: a free finger would be > 15 mm nearer to the previous hand position.
- e: black key with abs dx < 20 mm.
- e2: abs dx <= 26 mm (marginal).
- Analysis script and row data: scratchpad (an.py, rows.pkl); no project file was modified.

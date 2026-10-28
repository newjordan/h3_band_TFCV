# Piano-hand animation: literature review aimed at ~100% key sync

Date: 2026-10-10. Scope: what published work does to make every key press land, and what we can lift into our MPFB2 hand rig.
Conventions: [V] = read in the paper text this session. [S] = from search-result summaries only. [M] = from my memory, not re-checked; verify before relying on it.

Our situation (for reference): MIDI -> Viterbi fingering (Parncutt-style span costs) -> learned wrist path (Whittaker-smoothed) -> per-finger LM IK to pad targets with soft key collision -> key depth = how far the pad pushes it. Metric: key >= 50% down within 0.25 s of onset. Baseline 82%; misses are 'shallow' (~226) and 'sideways' (~113). Ragdoll wrist gave 76%, so the fault is in targets, fingering and solve, not in constraints.

---------------------------------------------------------------------
## 0. Executive summary (read this first)

1. Every published system that gets high press accuracy treats the fingertip as the primary quantity and the wrist as derived. Tipiano (F1 0.910 vs 0.121 for a diffusion baseline) fixes fingertip Y (which key) and Z (press depth) by hard masking during presses and lets only the front-back coordinate X float. The wrist is then computed as fingertip centroid plus a per-region offset. We do the opposite (wrist first, then finger IK). This is the biggest structural difference and the most promising lever. [V]
2. Our 'shallow' class is most likely a target-definition problem. RoboPianist and FürElise reward the key reaching ~full travel (RoboPianist drives key-state toward 1 with a tight tolerance; FürElise counts a key as sounded at 90% travel). If our pad target sits at "key surface" or "just touching", a soft contact model will leave the key part-way down. Aim the pad target below the fully-pressed position (overshoot) so the contact model has to bottom the key out. [V for what they reward; the diagnosis of our shallow class is a hypothesis]
3. 'Sideways' is a lateral-weight / black-key / reachability problem. Tipiano's std of pressing-fingertip positions is 11.4 mm along the keyboard vs a 23.5 mm white key pitch, i.e. real pianists only have about +/-1 sigma of slack; black keys have far less. Use anisotropic IK weights (lateral stiff, depth-along-key loose) and a reachability check. [V numbers; application is ours]
4. Fingering agreement with humans is capped: human-human match rate is about 71% and the best models are about 64-68% [V]. So "more human-like fingering" is not a route to 100% sync. The route is playable-by-construction fingering: costs computed from our own rig's reach, not from a generic Parncutt table, because our 17.7 cm hand is smaller than the table's reference hand.
5. The only approach that can actually guarantee ~100% is closed loop: solve, measure the exact metric per note, and re-solve failures with targeted corrections (wrist nudge, finger swap, deeper target). Nothing in the piano literature does this with a kinematic rig, but it is the standard "verify and repair" pattern and our metric is cheap to evaluate.

---------------------------------------------------------------------
## 1. FürElise (Wang, Xu, Shi, Schumann, Liu; SIGGRAPH Asia 2024)

Sources: https://arxiv.org/abs/2410.05791 , https://arxiv.org/html/2410.05791v1 , DOI 10.1145/3680528.3687703.

What it is [V]: dataset of markerless-captured two-hand MANO motion with a Yamaha Disklavier (MIDI ground truth); a diffusion model (EDGE-style transformer) makes 2 s windows (120 frames at 59.94 fps) of hand joint reference conditioned on a sheet-music matrix; retrieval of similar dataset motion by note-matrix L2 over length-30 windows; an RL policy (PPO, per-hand discriminators) imitates the merged reference in a physics sim and is rewarded for key presses. The abstract/intro says ~10 h, 15 pianists, 153 pieces; the conclusion says 8 h, 11 pianists, 98 pieces (internal inconsistency in the paper).

Fingering: not planned explicitly. The finger that is nearest the target key in the diffusion output is taken as the target fingertip. So fingering is implicit and inherited from the generative model. [V]

Key-press reward (their Eqs. 6-9) [V]:
- key counts as sounded when p_k/d_k > 0.9 (p = press distance, d = full travel)
- target reward r+ = 1 if p_k/d_k > 0.9, else exp(||p_i - p_k|| + 0.01 p_k/d_k)  (as extracted; the sign convention looks odd, check the PDF)
- non-target penalty r- = p_k/(0.9 d_k) if key touched with p_k/d_k > 0.1 else 0
- r = prod_k r+  - 0.15 sum r-  + 0.5 r_correct - 0.05 r_energy ; energy = exp(-0.75 sum_h(||v_wrist|| + 0.1 sum_i ||v_i||)^2)
- goal weight 0.9, 0.05 per hand imitation.
Takeaway: they require ~90% depth, not 50%, and ignore light touches below 10%. Our 50% criterion is looser than theirs, so targets that satisfy 90% will satisfy our metric with margin.

Accuracy reported [V]: per-frame precision/recall/F1 vs MIDI. Dataset reconstruction after IK repair: P 88.55 / R 92.53 / F1 86.49. Policy F1 above 0.8 on all 14 test pieces; ablation F1 on four pieces: Fur Elise 98.15 (full) vs 73.20 (RL alone); Rondo Alla Turca 94.65 vs 34.36; Clementi Op.36 96.21 vs 75.28; Sleep Away 83.75 vs 49.97. Takeaway: without a good reference trajectory even RL fails badly (34% on a leap-heavy piece), consistent with our finding that targets, not constraints, drive error. They say the policy struggles with finger crossover and does not model key velocity.

Dataset repair procedure worth copying [V]: where the Disklavier says a key was pressed but the capture does not show it, they run IK that optimizes only local joint rotations and wrist orientation, with the fingertip allowed to move at most 1 cm, loss = masked tip-position term + temporal smoothness (lambda = 0.0005), L-BFGS 100 epochs. That is a soft-bounded "snap to the key" pass, which is the kind of post-pass we lack.

Note from Tipiano: FürElise official code is not released, and their re-implementation of the diffusion part suffered mean collapse (F1 < 0.01). Do not plan on reproducing FürElise's policy. Their data is what we already use. [V, from Tipiano]

Hand size: no explicit normalisation in the policy; MANO shape beta from calibration video is fixed per fit. [V]

Actionable: (a) 90%-depth target semantics; (b) the post-pass snap-to-key IK with a 1 cm leash; (c) use their F1-type per-frame check as a sanity metric alongside ours. Effort low to medium.

---------------------------------------------------------------------
## 2. RoboPianist (Zakka et al., CoRL 2023) and follow-ups

Sources: https://proceedings.mlr.press/v229/zakka23a.html , PDF https://proceedings.mlr.press/v229/zakka23a/zakka23a.pdf , https://kzakka.com/robopianist/ , arXiv 2304.04150.

Reward (Appendix B) [V]:
- r_key = 0.5 * mean_i g(||ks_i - 1||_2) + 0.5 * (1 - 1[false positive]); ks = normalised key joint position in [0,1]; g = dm_control tolerance fn with bounds = 0.05, margin = 0.5. So full credit when key is within 0.05 of fully down, decaying over a margin of 0.5. A key is "active"/sounding when its joint is within 0.5 deg of max range.
- r_finger = mean_i g(||p_f - p_k||_2) with bounds 0.01, margin 0.1: a dense term pulling the assigned fingertip to a point on the key surface center.
- r_energy = |tau|^T |v|, weight -0.005. Total = r_key + r_finger - 0.005 r_energy.
- Constant (not count-proportional) penalty for false positives; otherwise agent hovers and never presses. Fingers may rest on inactive keys silently.
Design findings [V]: without fingering labels F1 stays at zero (sparse reward unsolvable); fingering helps present and later presses; larger lookahead horizon L helps reach notes in time; adding a control cost decreased performance (their note).
Fingering source: optimal-transport-based fingering from the environment (listed as such by a student report [S]); Tipiano supplied human annotations to RoboPianist as its baseline input [V].
Metrics [V/S]: precision, recall, F1 per timestep on key activations; a student extension scores onset F1 with a 50 ms window and adds a Gaussian onset-alignment bonus [S, https://cs224r.stanford.edu/ ... project PDFs, unreviewed student reports; treat as anecdotal]. Their MPC baseline and specialist RL both fall short on multi-song generalisation (Fur Elise F1 ~0.7 for specialist, ~0 for 16 songs) [V].

Follow-ups seen in search [S, not read]: RP1M (arXiv 2408.11048), PianoMime (2407.18178), PANDORA (2503.14545), Dexterous Robotic Piano Playing at Scale (2511.02504), HandelBot (2603.12243; fixes the last joint of each finger so only the tip presses; computes wrist position from the sheet music as the wrist pose that places the specified finger on each target key; relaxes false-press penalty because wrong presses are nearly unavoidable on that hardware). The HandelBot wrist rule is the same philosophy as Tipiano: wrist is a function of where the assigned fingertips must be.

Actionable: (a) our target should include a "depth to full travel" overshoot like ks -> 1; (b) a false-positive allowance: accept adjacent-key brushing below threshold so lateral stiffness is not over-penalised; (c) lookahead: the target for note n should already be in effect before onset (see section 5). Effort low.

---------------------------------------------------------------------
## 3. Tipiano (2026), PianoMotion10M (2024), other piano motion work

### 3.1 Tipiano (Bae et al., KAIST/SNU/Yamaha) https://arxiv.org/abs/2604.09692 [V, read full text]
Four cascaded stages, all conditioned on MIDI + fingering annotations (new expert fingering for the FürElise data, 340K finger-key events):
1. Fingertip prior: for each (hand, finger, key) with n >= 10 observations (735 of 880 combos, 84%; rest interpolated along the keyboard), store mean, std, quartiles in piano-aligned coordinates. Reported sigma_x = 16.3 mm (front-back), sigma_y = 11.4 mm (along keyboard), sigma_z = 7.7 mm (height). Baseline: pressing fingers sit at the median p50; non-pressing fingers interpolate between pressing anchors with a 14 mm hover height. "This baseline guarantees correct key contact."
2. Residual refinement (transformer on fingering, then FiLM on Aria MIDI embeddings): p2 = p1 + R(p1, F, h_midi); residuals clamped to +/-80 mm; during presses Y and Z residuals are set to exactly zero (hard mask), only X is free. Ablation: removing the mask drops recall 0.988 -> 0.909 (-8.0%). Smoothing (SmoothNet-style) is also masked to preserve Y and Z during presses.
3. Wrist: eight keyboard-region offsets delta_k = E[p_wrist - mean_fingertips | active keys in region k]; p_wrist,base = mean(refined fingertips) + delta_k; residual +/-50 mm via FiLM CNN; wrist error 36.8 mm.
4. STGCN fills intermediate joints with wrist and tips as fixed anchors.
Metrics [V]: key-only F1 (finger identity ignored), fingertip over key in XY and Z crossing calibrated thresholds (-1.19 mm white, +10.38 mm black, vs Disklavier's 8 mm trigger depth); MPJPE 32.8 mm; acceleration ratio 1.72 (their output is jittery vs GT); F1 0.910 +/- 0.028 over 27 test pieces, 0.88-0.95 across difficulty levels 3-9; FürElise data itself only 0.739 by this metric. Automatic fingering (ArGNN fine-tuned, general match 64.3 -> 68.3%) lowers F1 but the paper says smoothness gets more GT-like.
Expert critique [V]: wrist too rigid, lacks anticipation (especially thumb-under) and vertical wrist motion. So pure fingertip-first output risks looking mechanical; keep a learned wrist residual for naturalness, but never at the expense of the press.
Actionable (ranked high): fingertip-first target stack; hard mask Y/Z at press; per-(finger,key) tip-pad target from a prior; wrist = centroid + region offset (could replace or seed our learned wrist); hover height 14 mm for idle fingers. Caveat: their prior is in MANO fingertip coordinates from captured humans, not our pad geometry, so recompute stats on our own rig.

### 3.2 PianoMotion10M (Gan, Wang, Wu, Zhu; ICLR 2025) https://arxiv.org/abs/2406.09326 , https://github.com/agnJason/PianoMotion10M [S]
116 h of top-view video, 10 M annotated hand poses, 1,966 video/audio/MIDI pairs. Baseline: position predictor for hand location then position-guided gesture generator. Metrics: Frechet / Wasserstein gesture distance, smoothness, position distance. No key-press F1 as a primary metric as far as the summaries say. Relevance: the "predict hand position first, then pose conditioned on it" split is the same wrist-then-pose split we use; not a source of contact guarantees. Low priority.

### 3.3 Separate to Collaborate (Liu et al. 2025), https://arxiv.org/abs/2504.09885 [S]
Dual-stream diffusion, one stream per hand with coordination attention, built on PianoMotion10M. Tipiano reports it reaching only F1 0.121 on FürElise data. Evidence that generative-only motion does not hit keys. Skip.

### 3.4 Others seen [S, unread]: SKY-Piano dataset (arXiv 2607.27296). Not examined.

---------------------------------------------------------------------
## 4. Automatic piano fingering

Key data [V] (Nakamura, Saito, Yoshii, "Statistical Learning and Estimation of Piano Fingering", arXiv 1904.10237; PIG dataset: 150 pieces, 48,726 notes, 100,044 annotations by multiple annotators):
| Model | general match | high-conf | soft | recovery |
| 1st-order HMM | 61.7 | 68.3 | 82.8 | 74.0 |
| 2nd-order HMM | 64.3 | 70.8 | 85.3 | 77.6 |
| 3rd-order HMM | 64.5 | 71.0 | 85.5 | 77.8 |
| Chord HMM | 61.2 | 67.7 | 81.7 | 73.8 |
| DNN (LSTM) | 61.3 | 66.1 | 82.8 | 69.5 |
| Human inter-annotator | 71.4 | 79.1 | 90.8 | 84.3 |
Other facts [V]: chord constraint adds about 1.5-2 pt; going 1st->2nd order adds 2.6, 2nd->3rd only 0.2; extrapolated data scaling adds ~2 pt more, still under the human 71%; the paper only comments qualitatively that the lower note of RH octaves is almost always the thumb. Constraint-based VNS (Balliauw et al.) scored 56.7 on the shared subset. Pitch-difference matching model (arXiv 2108.09058) claims +3% general match over the 3rd-order HMM and a new metric, incapable-performing fingering rate (IFR), which is the idea that matters to us: measure physical playability, not agreement [S]. Ramoneda et al. ArGNN is the open SOTA neural model (68.3% after fine-tune per Tipiano) [V via Tipiano]. Relevant code: the 2026 paper arXiv 2609.28787 recreated the Nakamura HMM because code is not public [S].

Classic cost-based methods [M, verify]: Parncutt, Sloboda, Clarke, Raekallio, Desain (1997) with twelve rules (stretch, small-span, large-span, position-change count and size, weak finger, 3-4-5, 3&4, 4-on-black, thumb-on-black, 5-on-black, thumb passing); Hart, Bosch, Tsai (2000); Al Kasimi, Nichols, Raphael (2007) for polyphonic fingering with Viterbi/HMM-like search; Jacobs (2001) for hand-size dependence of spans; Balliauw et al. (2017) VNS for polyphonic music. Parncutt's span tables are indexed by finger pair with MinPrac / MinComf / MaxComf / MaxPrac in semitones, and the paper has a hand-size adjustment [M].

Which costs matter for our failure types:
- Octaves / wide chords: span costs dominate. The table's practical-max span for finger pair (1,5) is about an octave to a ninth for an average-to-large hand [M]. Our 17.7 cm hand is near that limit on the 1-5 octave (an octave on a modern keyboard is ~165 mm centre to centre [M, standard keyboard dimension]). Replace table limits by measured IK-reachable span on the rig, per hand size.
- Leaps: Parncutt's position-change size term; but pure fingering cost cannot fix it because a leap is a wrist-timing problem (section 5).
- Thumb crossings: thumb-passing cost plus the 3-4-5 / 4-on-black rules; thumb-under needs wrist rotation and forearm travel that a wrist-position-only model won't capture, which Tipiano's experts also flagged.
- Black keys: finger on black key means pad is higher (z) and further back (x); thumb-on-black and 5-on-black penalties exist for this reason. Our 'shallow' on black keys may be a depth reference mismatch (check).
Achievable: ~65-68% agreement with a given human; do not chase it.

Actionable: (a) feasibility-aware Viterbi: add an infinite or steep cost for any (finger-set, key-set, wrist pose) the rig's IK cannot reach, computed offline per hand size; (b) joint wrist+fingering optimisation (wrist is the state in the Viterbi lattice); (c) IFR-style playability metric on our output. Effort medium.

---------------------------------------------------------------------
## 5. Contact-guaranteed IK and timing

Sources are mostly from memory and not re-fetched; only Handrix was seen in search (https://www.dgp.toronto.edu/~gelkoura/handrix/paper.html [S]).

Principles that apply [M unless noted]:
1. Hard vs soft: soft penalty terms trade contact against posture; a hard (equality or very high weight) constraint at press frames with posture as a secondary objective is the standard fix. Practical variants: prioritised/null-space IK (Baerlocher and Boulic 2004), SQP with equality constraints, or a weight continuation (w_contact 1e2 -> 1e4) in LM. Our LM can emulate priority by solving with w_contact huge, then re-solving posture only in the null space of the Jacobian J of the contact rows: dq = -J+ e + (I - J+ J) dq_post.
2. Contact-invariant optimisation (Mordatch, Todorov, Popovic, SIGGRAPH 2012) optimises contact timing and positions jointly with the trajectory. Relevant idea: contact phases (which finger touches which key during [t_on, t_off]) are fixed by the score, so make contact constraints active exactly there and optimise the rest.
3. Handrix (ElKoura and Singh 2003) [S]: procedural fretting-hand animation with a hand model giving sympathetic non-playing-finger motion; shows plain IK makes idle fingers twitch. For idle fingers use a hover-height prior (Tipiano: 14 mm [V]) plus coupling, not independent IK.
4. Timing: the hand has to be positioned before onset. Common practice in animation and in RoboPianist is a lookahead: target for note n becomes active at t_on - T_prep with T_prep covering a minimum-jerk transfer. For a minimum-jerk move the peak speed is 1.875 D / T, so a 200 mm leap in 250 ms demands ~1.5 m/s, which is above what the wrist smoothing or finite finger speeds permit [M, minimum-jerk formula is standard; the feasibility judgement is mine]. Leap-heavy bars will fail unless the travel starts at the previous release (not previous onset) and the transfer is allowed to pass above the keys.
5. Smoothing can destroy arrival: a Whittaker smoother with lambda large will low-pass the wrist path and shift it so the wrist arrives late at onset. Tipiano masks smoothing at presses [V]. Use a weighted Whittaker: minimise sum w_i (y_i - z_i)^2 + lambda sum (Delta^d z_i)^2 with w_i = 1e4 on samples within [t_on - 0.1, t_on + 0.05] and 1 elsewhere, or use constrained splines through the onset samples.
6. Verify and repair: since the metric is computable in-loop, treat each missed note as an optimisation failure and re-solve it locally (bounded wrist offset search, deeper target, finger swap), then re-smooth with those samples pinned. This is the one method with a convergence argument toward 100%.

---------------------------------------------------------------------
## 6. Mapping onto our two miss classes (hypotheses, not facts)

Shallow (~226): pad on key but key < 50% down.
 - H1: target is at the key surface, soft collision reaches equilibrium with pad touching but not pushing. Fix: target below the fully-pressed surface by a margin (overshoot 2-4 mm over key travel, which is ~8-10 mm real [V: Disklavier trigger depth 8 mm]).
 - H2: pad arrives late (after 0.25 s) because of smoothing or insufficient travel time. Check by logging time of arrival vs onset for shallow notes.
 - H3: finger Z range / DIP-PIP flexion limits at the required wrist height; the wrist is too high so the finger cannot flex enough. Fix: add a wrist-height DOF and weight it.
Sideways (~113): pad off the key across.
 - H4: lateral weight in the LM objective is equal to depth weight, so the solver trades lateral error against posture; or target is key centre but the span is infeasible, so the solver compromises (check residual per note, the feasibility test).
 - H5: black-key narrowness and wrong-key brushing at octave leaps with small hand; fingering needs thumb on black key or a different finger set.
Smaller hand worse on octaves: consistent with H5 / reach limits; the oversized hand has slack, the small one does not, so errors reveal reach infeasibility, not constraint tightness (consistent with ragdoll result).

---------------------------------------------------------------------
## 7. Ranked actions (expected impact / effort)

| Rank | Action | Impact | Effort |
|1| Per-note failure diagnosis dump: for every miss record onset, arrival time, pad-to-key lateral error, IK residual per finger, fingering, wrist pose, hand size. Needed before anything else. | enabling | low |
|2| Overshoot press target (ks->1 semantics) with depth margin | high on shallow | low |
|3| Anisotropic IK weights, hard-mask lateral and depth at press frames (Tipiano) | high on sideways | low-med |
|4| Pin the smoother at onsets (weighted Whittaker) and add lookahead arrival | med-high | low-med |
|5| Fingertip-first wrist: wrist = centroid of assigned tips + region offset, learned wrist as residual | high | med |
|6| Verify-and-repair loop on missed notes | high, near-guarantee | med |
|7| Rig-measured reachability in the Viterbi cost (hand-size-aware) | high for Rachmaninoff / small hand | med |
|8| Joint wrist + fingering search per chord slice | med-high | high |
|9| Hover-height prior for idle fingers; reduce idle twitching and stray presses | low-med | low |

---------------------------------------------------------------------
## 8. Experiment list (hypothesis and measurement)

Use the same-seed A/B protocol; always report key sync overall, per piece, per hand size, and per miss class (shallow / sideways / late), not just the 82% headline.

E1. Failure autopsy. Hypothesis: shallow and sideways misses separate cleanly by (arrival time, lateral error, residual). Measure: table of 339+ misses with those columns; counts of late vs IK-infeasible vs target-defined. Decides everything below.

E2. Press overshoot. H1. Set pad target z = surface - (full_travel + m), m in {0, 2, 4, 6} mm; measure shallow count, key sync, and false-press count (adjacent keys > 10% travel). Expect shallow falls most; watch for finger hyperextension and key bottoming artifact.

E3. Lateral stiffness. H4. Rescale LM weights: lateral w_y in {1, 3, 10, 30} x depth weight; measure sideways count and resulting posture cost. Expect sideways falls until reach-infeasible notes remain.

E4. Hard mask at press (Tipiano style). During [t_on, t_on+0.1] fix pad Y, Z targets via null-space projection or w=1e4; let X (along key) float. Measure sync and joint-limit hits; compare with E2+E3 to see if projection adds anything over weights.

E5. Weighted Whittaker pin. H2. Weights 1e4 near onsets, lambda unchanged; measure arrival-time distribution (fraction arriving > 0.25 s late), sync, and jerk (so naturalness does not regress; Tipiano's acceleration-ratio is a usable naturalness proxy, target ~1.0).

E6. Preparation lead. Start each key target at t_prev_release or t_on - T_prep, with T_prep in {0.10, 0.15, 0.25 s}; minimum-jerk transfer. Measure leap-heavy bars separately (Rachmaninoff).

E7. Fingertip-first wrist. Compute wrist = mean(assigned tip targets) + delta_region (8 regions per Tipiano, fit on our learned-wrist outputs or the FürElise data), then add the existing model as a residual clamped to +/-50 mm. Measure sync and the wrist-path deviation from the current model (to confirm it still looks natural; Tipiano's experts found this style stiff).

E8. Verify-and-repair. After full solve, for each missed note try in order: wrist offset search (3-DOF grid +/-20 mm, 5 mm steps), deeper target, alternative finger from the Viterbi top-k; accept the first that passes without breaking neighbours; re-smooth with pinned samples. Measure final sync, number of repairs, per-repair cost.

E9. Reachability-aware fingering. Precompute, per hand size, feasible (finger_i, finger_j) key spans by IK on the rig; replace Parncutt span limits with those. Measure Rachmaninoff sync for 17.7 vs 24 cm hands; expect the gap to shrink. Also check IFR-style infeasible-fingering rate.

E10. Fingering ablation. Same targets, fingering from (a) current Viterbi, (b) reachability-aware, (c) human-annotated fingering for a short passage (PIG or Tipiano annotations if released) to estimate how much error is fingering vs solve. Hypothesis: if (c) is not clearly better than (a), fingering is not the bottleneck.

E11. Idle-finger hover prior. Non-pressing fingers target 14 mm above key surface (Tipiano) with a weak weight; measure stray presses and neighbouring-key false positives, and sideways misses from collision pushing.

E12. Hand-size scaling. Run 17.7, 20, 24 cm with all fixes; plot sync vs hand length. Expectation: after E7-E9 the curves converge; if they don't, the residual is a true reach limit and needs wrist radial/ulnar deviation or different fingering.

---------------------------------------------------------------------
## 9. Unverified / caveats
- Tipiano numbers come from the full text but are from one preprint (not yet reviewed beyond arXiv); its fingertip statistics are for MANO-captured pianists.
- FürElise reward formula as extracted by the fetch tool may have a sign error (exp(||.||) would grow with distance); check the PDF.
- RoboPianist follow-ups (RP1M, PianoMime, PANDORA, HandelBot) were seen as search snippets only.
- Parncutt, Hart, Al Kasimi, Jacobs, Balliauw details, Mordatch 2012, Baerlocher and Boulic 2004, minimum-jerk numbers: from memory, not re-fetched this session.
- 'Octave = 165 mm' and 'real key travel ~8-10 mm' are standard keyboard dimensions from memory; Disklavier trigger depth 8 mm is from Tipiano.
- Student CS224R reports on onset-alignment bonuses are anecdotal.
- No paper found that reports a 50%-travel-within-0.25 s metric; ours is stricter on timing, looser on depth than FürElise.

## Sources
- https://arxiv.org/abs/2410.05791 (FürElise); https://arxiv.org/html/2410.05791v1
- https://proceedings.mlr.press/v229/zakka23a.html ; https://proceedings.mlr.press/v229/zakka23a/zakka23a.pdf ; https://kzakka.com/robopianist/
- https://arxiv.org/abs/2604.09692 (Tipiano)
- https://arxiv.org/abs/2406.09326 ; https://github.com/agnJason/PianoMotion10M (PianoMotion10M)
- https://arxiv.org/abs/2504.09885 (Separate to Collaborate)
- https://arxiv.org/abs/1904.10237 (Nakamura et al., PIG); https://arxiv.org/abs/2108.09058 (pitch-difference model)
- https://arxiv.org/abs/2603.12243 (HandelBot); https://arxiv.org/abs/2408.11048 (RP1M); https://arxiv.org/abs/2407.18178 (PianoMime); https://arxiv.org/abs/2503.14545 (PANDORA)
- https://www.dgp.toronto.edu/~gelkoura/handrix/paper.html (Handrix)

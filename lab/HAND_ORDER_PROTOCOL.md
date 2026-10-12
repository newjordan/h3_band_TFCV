# No-crossing performance protocol, 2026-10-11

User requirement: the hands must not cross in this performance. The complete
old minute has five wrist inversions at playback seconds 47.917–52.917, with
the right root 251.803 mm left of the left root at the worst saved sample.
The original v19 part 17 contains a lone right-hand bass note at 293.193 s;
the learned model attracts its otherwise idle wrist to that first note for
the entire earlier rest. Staff splitting after window extraction is also
unstable if one staff is silent.

Hypothesis: persistent score ownership, a bounded first-note anticipation
window, and hard bimanual wrist bounds remove this failure without changing
the score, timing, anatomy, skin or solver budgets. Allocate from the full
performed ledger. Original releases lock ownership; new notes use a single
ordered pitch partition. Preserve every event. Report redistribution and
all capacity/span conflicts; an impossible score is not silently repaired.
The existing fingering/event shortening remains visible against original
releases and is not considered correct musical support.

Whole-hand solves use a common partition plane and hard left/right root-x
bounds. Supported wrist references receive weight 8 versus idle weight 1.
After smoothing, solve the ordered root positions inside both physical arm
workspaces before solving the fingers. Do not modify serialized wrist/skin
poses after solving. Minimum root separation is 60 mm, an engineering
planning bound, not a proof of skin separation. Actual interhand geometry,
body/costume, speed, key support and continuous collision remain independent
acceptance gates. Ordered references can still miss notes and are rejected
as complete realistic motion if any gate fails.

Independent actual-skin checks use production Blender level-2 LBS surfaces,
unchanged topology/weights, CPU parity, both quad diagonals, finite instrument
geometry and assigned distal support. Compare matched source frames 36 and
72 (288.5 and 290 s) in both stress trials. An additional adaptive diagnostic
selects the new minute's worst wrist-gap frame after controller measurement,
1079 (284.858 s), before inspecting its skin audit. This selection is not a
preregistered whole-motion geometry test. All failures remain recorded.

Matched full-budget stress trial: performed score 287–295 s, 24 Hz, size 1,
seed 0, E36 contact weights, 2 wrist passes, 2 ease passes, 20 finger
iterations, finger refinement and 2 contact repair passes. Compare legacy
policy against ordered policy; other composure switches stay off to isolate
this correction. Fingerprints are frozen and identical for both trials.

Fresh minute: eight independent 7.5-second solved chunks at 239.9–299.9 s,
identical full ledger ownership, full budgets, size 1 and 24 Hz. Stitch only
unaltered solved poses; audit every original sample and each chunk boundary
for order, joint/root speeds, exact arm lengths and original note support.
No blending or fake note depression. Chunk-boundary discontinuities are
explicit failures, not disguised by a smooth render. Rendering selects
every other solved frame at 12 fps, 768×432 CPU grayscale with original
timed audio. A wrist-order pass does not accept the complete rig.

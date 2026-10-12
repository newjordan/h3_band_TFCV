# Actual-skin initialization experiment, 2026-10-12

Preregistered before numerical trials. Full user goal remains natural piano,
drummer and whole-body playing with no geometry clipping, controlled resting
digits, whole-wrist reach, musical/key/foot/force relationships and continuous
motion. This experiment repairs a prerequisite, not the goal's success scope.

## Frozen input and obligations

Use the first actual warmup pose (316.9 s, internal frame 0) of
`artifacts/realism/physical_key_actual_legacy/internal.anim.json`, size 1,
24 Hz, notched keys. Keep original key commands, assignments, pads/objective,
activation, shoulders, arm lengths, scale, skin weights and subdivision 2.
Preserve original performed-source gates as diagnostic compatibility duties;
do not interpret transcription offsets as verified physical key-off truth.
Hold-required pitches have onset at least 80 ms earlier and offset later than
this instant. Request actual finite underside anchors for every assigned
required pitch even when the legacy activation has prematurely faded. Record
unassigned required pitches explicitly; neither reassign nor drop them.
The fixed source commands are only partial/zero key travel for some duties.
Skin touching such a key is not proof of a physically correct press.

All cells use calibrated soft-union finite-key placement, actual-anchor
objective, pooled soft human-form weight 4, and the same required-anchor
vertices/targets. Thumb form and forces remain unvalidated. Next physical
frame 1 is supplied with full root/elbow evidence. No previous frame exists.
Hard limits: anatomy; 12 rad/s finger joints; 0.8 m/s wrist translation;
4 rad/s wrist rotation and elbow plane; 35 mm wrist workspace; fixed arm
lengths and shoulders; existing finger-tube spacing. Skin/instrument gap
10 micrometres; anchor top gap 10–100 micrometres. Keep the 0.3-rad component
root workspace, which is not itself a total rotation or clinical limit.

## Contrasts

Four cells, both hands, eight independent solves:

| Witnesses | Initialization |
|---|---|
| legacy | bounded source |
| legacy | quiet-finger rest |
| both interior | bounded source |
| both interior | quiet-finger rest |

Quiet-finger rest changes only initial joint angles of digits with activation
below 0.05 and no required held pitch. Preparing/held fingers retain their
source angles. Wrist is unchanged. Seeds are clipped to identical anatomy and
next-frame joint-speed bounds. No comfort statistics replace hard bounds or
acceptance. Objective and human-form weight do not change with the seed.

Legacy optimizes full nonincident pairs from diagonal 0. Both-interior uses
both independent quad triangulations, qualified by diagonal identity, and
includes unexpected connected-face folds. Alternatives are never intersected
with each other. Nonincident pairs keep full triangles and the 10-µm gap.
Incident pairs use uniform barycentric inset 0.01 (each interior corner has
weights 0.98/0.01/0.01); margin is min(10 µm, 0.25 × inset × source minimum
triangle altitude), frozen when discovered. This is a conditioning/witness
choice, not a clearance tolerance. Near-feature folds can escape it. The
witness contrast bundles second-diagonal coverage and incident constraints;
do not attribute results to only one of them.

Each solve: 100 SLSQP iterations/pass, 80 bounded feasibility evaluations/pass,
4 passes maximum, gap and existing finite tolerances unchanged. Report actual
iterations, passes, evaluations, pair counts and runtime; matching budgets do
not mean matching work. Three CPU workers, BLAS/OMP/MKL one thread each; no GPU.

## Acceptance and independent evidence

Never export a rejected numerical pose as an accepted candidate. Retain all
solver coordinates as labeled failure diagnostics. Reconstruct every cell's
two numerical hands for independent production-Blender snapshots, including
failures, and audit both full/untrimmed diagonals, instrument triangles,
distal finite-key support and interhand crossings. Check CPU/Blender parity,
source/code hashes, all requested anchors, next-frame bounds and exact arms.
Also audit the untouched initial source. No post-outcome pose/frame selection.
Rendered grids must visibly label unvalidated numerical diagnostics.

Optimizer success, reduced collision counts, interior slack, tests, graph
approval, zero sampled triangle intersections, or one accepted hand pose do
not establish full musical/geometry/naturalness performance. Missing required
pitches, wrong key travel and an infeasible next pose remain explicit even if
a local hand passes. No controller-default adoption in this experiment.
Positive mesh distance, containment, skin/body/costume/feet, forces, nonlinear
continuous-time paths and drummer acceptance remain required future work.

## Primary evidence and engineering inference

Li et al., IPC (2020), section 2, PDF page 3, defines an admissible trajectory
from an intersection-free start with positive primitive distances; its step
uses linear nodal interpolation. Page 12 explicitly requires nonzero initial
separation. Section 3.1 notes scene-dependent limitations of local signed
proxies. Re-read primary PDF on 2026-10-12:
https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf .

Engineering inference: a feasible start is needed before barrier-style path
planning. These local articulated SAT/interior constraints are a restoration
hypothesis, not IPC or its guarantees. Complete untrimmed static audit is a
necessary veto; positive distance and nonlinear trajectory safety remain open.

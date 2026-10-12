# Experimental physical orientation API

Set piano overrides before compiling/preparing a score:

```python
from blockout import piano
piano.WRIST_SO3_HISTORY = True
piano.WRIST_SIGNED_ANGLES = False
score = piano.compile_score(original_performed_notes)
plan = piano.prepare_plan(compiled_score=score, start=282.9, dur=4., fps=24,
                          physical_warmup=True)
animation = piano.solve_plan(plan)
```

Both flags remain false by default. The flags-off matched run reproduces the
previous warm trajectory exactly. `WRIST_SO3_HISTORY` bundles absolute physical
history transport and offline manifold filtering; primitive attribution awaits
separate controls. Every chronological pass records a predecessor once per
finished physical frame. Same-frame reconciliation seeds already use the current
reference and are not transported as previous-time states.

`blockout.rig.orientation` provides absolute/relative encoding, explicit polar
separation for exported reference precision, spatial-increment smoothing and
forward/backward-aware forearm direction coordinates. Fixed wrist-reference
stretch is retained in physical matrices. The fit uses rotation-vector norms,
principal logs, fixed sampling and free endpoint anchoring; future frames and
the complete prelude participate. It is noncausal and local, with no global
optimum, velocity/acceleration, anatomical or geometry guarantee. Output receipt
counts are optimizer-reported nfev, not all finite-Jacobian residual probes.

Known unresolved chart domain: exactly pure lateral forearm direction has
undefined flexion `atan2(0,0)`. The current function returns the zero convention
with 90-degree deviation; explicit rejection/domain handling is pending. No
exported pose in this four-treatment study enters that domain (minimum chart
denominator 0.926 across all histories). Direction coordinates also omit axial
twist and are not a clinical wrist model. There are no backward-facing samples
in this passage, so synthetic reversal tests do not attribute its observed
clipping to the old asin ambiguity.

The recorded four treatments all remain unaccepted. SO(3) reduces angular
motion but introduces initial right-hand penetration and loses an active
finite-key support. Combined treatment reduces matched left-hand clipping and
self folds, but worsens initialization and original-gate coverage. The initial
physical pose is already infeasible in every treatment. Actual warmup, full
body/costume, forces, pedal/key actions and nonlinear continuous-time clearance
remain required. See `orientation_20261011.json` and `SCORE_SEMANTICS_FINDING.md`.

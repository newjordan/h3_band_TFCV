# Piano composure trial, 2026-10-11

Hypothesis: gently curled unused digits carried by the wrist, and wrist
translation for an isolated reach, improve composed playing without dropping
notes or weakening anatomy, motion or surface checks.

Four runs: baseline, free control, wrist carry, both. Each uses the complete
current piano controller on the same performed Knights Suite passage,
247–253 seconds, 24 Hz, measured hand scale 0.7284, seed 0, E36 contact
weights/repair budget and identical notched geometry. Source/code hashes must
remain identical. This does not establish transfer to the size-1 v19 preview.
The existing baseline already fails realism checks; relative improvement is
diagnostic evidence and never acceptance.

Free control removes the old idle lift/extension/fanning and uses the existing
second-order target filter at 7 Hz, damping 1, response 0, four integration
substeps per 1/24-second frame. State is target position/velocity in metres
and metres per second, initialized from the first input with zero velocity.
Actual inverse kinematics uses wrist-frame relaxed FK pad targets. The
preparation displacement is at most 4 mm before the key-floor constraint;
the floor can raise it further, which must be reported rather than concealed.
Free long-finger comfort gain is 16, thumb gain 1, fading continuously to
pressed gain 1 by the existing smooth activation. These are engineering
objective gains, not physical forces. Pressed targets and gains stay intact.

Wrist carry translates the reference laterally toward the sole assigned
finger's relaxed pad, faded by activation, limited to 120 mm per reference
correction. Another original-score held pitch vetoes isolated translation.
It also tracks all held slices and removes the reach-based early-release
heuristic. Existing unavoidable fingering/event conflicts remain recorded
against the original score; removing that heuristic is part of this declared
wrist-planning factor. Whole-hand IK, temporal smoothing, original joint
bounds, 12 rad/s joint speed cap and physical arm workspace still apply.
Wrist velocity is measured; no new hard wrist-speed or force guarantee exists.

Acceptance requires every original note's key at least 50% down within
250 ms of onset, and support at every 24-Hz sample from onset+40 ms to
release−20 ms. These are existing onset and declared sampled hold diagnostics;
actual rendered distal support is an additional independent check. Report
all early releases and missing assignments. No onset/hold failures, anatomy
or speed violations, instrument or same/interhand surface crossings are
acceptable. Check both quad diagonals on actual evaluated Blender level-2
surfaces at the identical frames 48 and 108 (249 and 251.5 score seconds).
Sampled passing cannot establish positive clearance, containment, full-body
or costume safety, force/action behavior or nonlinear articulated CCD.

Measure unused-middle PIP and wrist-local excursion, including common unused
frames across all four runs, active MCP abduction, achieved wrist movement,
key-clipped relaxed reference versus achieved pad, and fresh matched video.
Keep every failure. Both switches remain off by default. A passing excerpt
would still not accept the complete music/body/costume/drummer performance.

Native method review: episode
39148ff9245c48bc6f6163abb5db398e44caa098622ecb837c933f05ff1f4db8,
five completed roles; verdict PASS for diagnostic methodology only.

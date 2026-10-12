# Coherent score plan and physical warmup, 2026-10-11

The full objective remains natural playing rigs, full music/body/costume and
drummer, with no geometry clipping, controlled fingers, physical key action
and continuous-time validation. This experiment addresses a prerequisite;
no passing excerpt will accept the complete performance.

Observed: independent 7.5-second solves produce seven joint-speed failures
at seven joins, maximum 40.366 rad/s versus the unchanged 12 rad/s bound.
One wrist jumps 111.643 mm. Although score ownership is globally ordered,
fingering, body centre, spring history, role initialization, refined paths
and physical previous-frame state are window-local. The two-second prelude
only filters targets: actual hand IK begins at the exported window.
Actual closest-hands skin still clips the instrument by 6.533 mm.

Hypothesis: compile complete original-score fingering and fixed body placement
once for a performance, solve its physical prelude, and export render windows
from the same trajectory. This eliminates simulation resets and may improve
contact/posture. It does not directly impose actual-surface feasibility.

Four treatments on performed Knights Suite 282.9–286.9 s, 24 Hz, original
size 1, seed 0, existing E36 solver budgets, unchanged geometry/anatomy:
windowed (two independent two-second legacy solves), coherent (one four-second
complete-score plan), warm (same with two seconds of physical prelude),
composed (warm with existing free-control and wrist-carry factors enabled).
The composed treatment is an additional controller factor, not an isolated
warmup result. Warmup adds 48 internal frames; disclose extra work rather than
claim equal total computation. Per-frame budgets are unchanged. Every
original note/time/velocity is immutable; existing scheduling truncations
remain measured against original events.

Prepare/solve APIs must keep the full plan's fingering, centre, role/spring
arrays and one physical previous frame throughout each solver pass. Render
partitioning changes only time origins and selects already solved frames;
rejoining must restore identical poses, key depths, body data and timestamps.
No blending, shifted notes, fake key depression or post-solve pose patches.
Keep legacy compilation/warmup available as experimental controls; do not
silently alter previous experiment semantics.

Inspect all 96 visible samples for original onsets/holds, anatomy, 12 rad/s
joint cap, wrist linear/angular motion, exact arm lengths, wrist and MCP
order, inactive-middle curl, active abduction and control-point spacing.
Evaluate actual production Blender level-2 LBS surfaces independently with
CPU/topology parity, both quad diagonals, instrument/self/interhand surfaces
and finite assigned distal support at identical visible frames 47 and 48
(284.858 and 284.900 s). Selection precedes outcomes. Sampled geometry is not
positive-distance, containment or nonlinear articulated CCD proof. Initial
surface infeasibility, missing musical support, body/costume, force/action
and remaining drummer failures still prevent complete acceptance.

Use the native five-role research graph to review this method and criticize
warmup work/context imbalance, nonlinear path representation and posture
claims. Its verdict concerns methodology, never empirical rig approval.

Native review completed five of five nodes, episode
`e085d72d3a37db6ce0f96f39477b8a15ea4495c39b2ac99789d6579273fe42b7`.
Its methodological PASS explicitly treats A→B as a combined change of
preparation, fingering, body placement and history. Recommended additional
same-plan midpoint history-reset and controller-without-warmup controls are
pending; the four preregistered treatments cannot isolate those interactions.
All physical warmup poses are retained and their temporal/anatomical bounds
reported. Actual warmup surface feasibility is unverified, not assumed.
The complete original ledger also contains pitch 20 at 37.196–39.941 s,
outside this 88-key instrument's 21–108 range; compilation preserves and
reports this event. No whole-score playability claim is possible yet.

# Prepared score, continuous solve and render windows

The isolated controller offers explicit opt-in preparation and physical
warmup. Existing `animate` calls retain legacy window compilation and skip
physical warmup, keeping earlier experiments available as controls.
No-crossing ownership/root constraints remain enabled by default.

```python
from blockout import piano
from blockout.performance_windows import export_window, rejoin_windows

# Set controller overrides and hand dimensions before compilation.
score = piano.compile_score(original_performed_notes)
plan = piano.prepare_plan(compiled_score=score, start=282.9, dur=4., fps=24,
                          physical_warmup=True)
animation = piano.solve_plan(plan)
parts = [export_window(animation, 0, 48), export_window(animation, 48, 96)]
assert rejoin_windows(parts)['frames'] == animation['frames']
```

Compilation copies input notes, preserving touch/lift and original releases.
It computes full-score hand ownership/fingering and a fixed body centre.
Existing fingering/event scheduling may shorten motion holds; those events
must still be assessed against the immutable original ledger. Notes outside
the physical keyboard are retained and explicitly reported.

Preparation copies the compiled hands and computes one requested trajectory's
roles, spring histories, free/press activation, body/head/torso arrays and
two-second prelude. These filters and path smoothing use lookahead. Preparing
separate time windows does not make their trajectories interchangeable;
solve the entire intended performance once before exporting render windows.

Solving consumes the prepared plan once. Controller changes between the
three stages are rejected. Physical warmup executes the prelude through all
hand passes and the arm-track solve before trimming the visible output.
Every pass carries its own chronological physical predecessor; contact
reconciliation warm starts remain frame-local numerical states. Warmup adds
work at unchanged maximum per-frame budgets. Its poses are retained in
`warmup.frames`; aggregate collision/reach diagnostics include internal
frames, while visible pose diagnostics are labeled separately.

Export selects unchanged solved poses and changes only their time origin.
Each partition records source timestamps and checksums; rejoining rejects
mixed sources, gaps, incomplete coverage, altered poses/keys or changed
timestamps. It restores the exact recorded timestamps rather than relying
on floating-point addition/subtraction. Window QA must be recomputed;
the original solve owns its aggregate diagnostics.

This API is a continuity prerequisite, not a surface-feasible controller.
SO(3) transport/filtering, explicit wrist velocity limits, actual skin and
warmup contact, stable contact/forces, pedal semantics, nonlinear CCD,
full body/costume, complete music and drummer remain unresolved. Soft curl
and wrist-carry settings remain experimental and off by default.

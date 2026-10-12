# Experimental source voices and physical key actions

`compile_score(notes, key_action_plan=plan)` is opt-in. Default allocation,
release heuristics and motion remain legacy. The complete original notes stay
in `PianoScore.notes`; animation exports retain the source-action ledger and
a separate `physical_key_plan`. Preparation copies these structures.

`propose_key_plan(notes, timing_receipt, source_channels, baseline_ledger,
pedal, conditional_releases=False, offset_tolerance=.030)` requires a complete
timing receipt and explicitly verified raw-input channels. General `Note`
objects do not carry channel evidence. This suite's parser sees 2,046 note-ons,
all channel 0, across two tracks; the lab verifies this before supplying the
mapping. Both original MIDI bytes and the performed source remain hashed.

Only identical canonical onset/pitch/channel across distinct tracks may share
an experimental strike. Merely close repeated notes, different channels and
ambiguous same-track duplicates remain distinct. A group uses its earliest
performed onset, maximum velocity, maximum original end, and attributes from
the strongest voice, with deterministic tie-breaking. Every source voice's
timing and velocity differences are recorded. One shared strike does not
reproduce the existing soundtrack's separate jittered impulses; original
audio is not replaced and waveform equivalence is false.

Conditional release uses a previous scheduled release only if each member is
inside the recovered pedal domain, has uninterrupted ideal binary sustain
through its original offset, and has an estimated pedal-up within the explicit
30 ms comparison window. The window is not estimated model accuracy. Releases
remain hypotheses, not unique physical key-offs. Unknown or failed cases keep
conservative source gates. The validator recomputes coverage against the
bounded recorded hypothesis rather than accepting an `accepted` flag alone.

The plan is a complete disjoint source partition. Compilation rejects a
wrong-source hash, missing/duplicate/unknown memberships, changed action
policy and unsupported release evidence. Motion copies preserve source IDs,
touch and lift. Source-voice coverage and physical-action count are reported
separately: many voices under a shared action are not a scheduling bijection.

Strict plan scheduling keeps every requested hold through fingering repair,
finger reuse and reach scheduling. Nearby repeated pitches survive slicing.
Holds stay candidates after eight seconds. Impossible overlapping assignments
remain in `requested_interval_conflicts`; they are not solved or hidden by an
early release. The ordinary controller can still choose only one target per
finger at each instant, so support failures must remain in the report.

The full-score 318.9–322.9 s experiment measures current scheduling,
conservative holds and conditional releases. Original source onsets/gates and
proposed physical holds are scored independently; solved key depth is a
diagnostic, not finite skin contact. Actual production meshes include first
warmup pose and two fixed instants. No default adoption, realistic force/key
or foot action, continuous skin/body/costume clearance, or drummer acceptance
is implied by the API or passing tests.

# Source/action and estimated pedal evidence

`midi.read` keeps raw note-offs unchanged. `midi.control_changes(path, 64)`
separately returns tempo-mapped CC64 events with value, channel, track, tick
and original event index. Streams are not silently merged into one pedal.

`feeling.perform(notes, chart, seed, timing_receipt={})` optionally fills the
given empty dict with one input/output identity, shared mapped times, actual
note timing variation and clamp flags for every generated note. It preserves
the original RNG sequence and notes. The shared time map applies to controllers;
the independent chord/note jitter does not. Reusing a nonempty receipt fails.

`piano.compile_score` now assigns a distinct internal ID to every performed
source attack. `score.action_ledger`, `plan.action_ledger` and the exported
animation's `key_action_ledger` retain original timing, ownership and each
slicing/repair/reuse/reach-release stage. Missing IDs stay explicit; the ledger
does not reschedule them or claim that a proposed hold makes contact. The legacy
seven-value events remain unchanged. Roll copies preserve IDs and touch/lift.

`key_actions.BinaryPedalTimeline` queries an explicit bounded binary hypothesis,
retaining original events around a crop so its starting state is inherited.
It rejects invalid, unordered and contradictory coincident transitions.
Release coverage requires uninterrupted down state over `[release, offset)`.
Queries outside the recovered domain fail rather than return a guessed state.
`release_hypotheses` returns nonunique intervals near pedal-up estimates;
comparison tolerance is not transcription accuracy, and it selects no key-off.

`legacy_sampler_damping` reproduces the old sampler's programmed reset rule,
including its final three-second tail. Equal damping times do not certify
real piano acoustics, repeated-key interaction or a complete performance.

The original Note interface still lacks channel identity. The recovered CC
reader preserves it; general multi-channel note/action lineage remains open.
The current study does not change default holds, key/pedal motions, original
scoring or the soundtrack. Physical key/foot actions, all geometry and forces,
natural finger/wrist form, full body/costume and drummer still need acceptance.

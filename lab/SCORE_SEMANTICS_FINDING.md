# Original music and physical key actions, 2026-10-11

**Later source recovery:** the cached upstream transcription was found at
`/tmp/claude-1000/suite/work/raw.mid`, with 282 estimated CC64 events. The
note-only export discarded them. Both delivered Chopin and suite MIDI files
now reproduce byte for byte from the archived fitting data and unchanged
writer. Recovery keeps all 282 events: 280 mapped into the Chopin domain,
one earlier event supplies its initial down state, and one terminal event
stays outside the crop. No terminal up transition is fabricated. See
`key_actions_20261011.json` and `KEY_ACTION_PROTOCOL.md`.

The preview's preserved audio is synthesized Salamander audio with the
sampler's automatic bass-change resets, not the source Frank Levy recording.
The archived rendering command supplied no downbeats. In the full-score
controller, 1,478 action proposals shorten source holds; 1,003 of those cuts
would change that sampler's damping time. Estimated Chopin pedal covers 391
cuts uninterrupted, fails 394, and leaves 693 outside the recovered domain.
These counts are conditional diagnostics, not verified releases or sound
equivalence. The source ledger also reveals two attacks lost at slicing.
All original notes, legacy controller targets and soundtrack remain intact.

Primary target-processing code supports pedal-extended note offsets by
default. The exact training invocation for this cached checkpoint remains
unverified; neither all-key-off nor all-pedal-off interpretation is justified.
The earlier evidence below refers to the delivered MIDI, which still has no
CC events. Absence there no longer means the upstream pedal estimates are
unavailable.

This is evidence, not a changed arrangement or permission to omit notes.
The immutable performed suite contains 2,046 events. Raw MIDI and performed
MIDI both reach 22 simultaneous distinct pitches; raw gates exceed ten
distinct pitches for 41.135 seconds and performed gates for 42.398 seconds.
The maximum is a descending Chopin passage: earlier pitches remain gated
until nearly the same release, after the hand has traversed several octaves.
No controller-change events, including sustain CC64, exist in the suite MIDI.
The reader uses raw note-offs and does not extend them with pedal messages.

Repository authority: `examples/piano/README.md` identifies the Chopin as our
audio transcription of Frank Levy's recording, fitted bar lines and no pedal
data. Thus its inferred note offsets are not verified physical finger-release
times. They must not be used alone to claim that a human hand must retain all
22 keys. Conversely, absent pedal data does not prove which gate is acoustic
sustain or authorize arbitrary note shortening. The full original onset,
pitch, velocity, release ledger and audio remain unchanged in every experiment.
Existing original-gate coverage remains a disclosed compatibility measurement.

The primary transcription implementation supports a combined note/pedal model
and returns separate estimated note and pedal events:
https://raw.githubusercontent.com/qiuqiangkong/piano_transcription_inference/master/piano_transcription_inference/inference.py
Its parent research repository documents note, pedal and combined training:
https://github.com/bytedance/piano_transcription
This is a concrete recovery avenue, not proof that a new estimate will be
correct. Do not invent pedal ground truth from filenames or long offsets.

Next required state representation: preserve source transcription/acoustic
events, author or recover validated physical key-action intervals separately,
and retain explicit damper-pedal motion/contact. Every source attack must stay
accounted for; damper sustain cannot hide a missing press, and authored key
release cannot silently replace the original source ledger. Check score,
recording and timing together and report uncertainty and edit provenance.
Original theme pitch20 remains outside the existing 21–108 instrument; the
score cannot become wholly playable by silently deleting/transposing it.

Evidence artifact: `artifacts/realism/orientation_score_semantics.json` retains
source hash, raw/performed peak identities and every overfull transition and
interval. Ten pitches is a diagnostic threshold, not full geometric
playability proof or a categorical assertion that one finger can never touch
two adjacent keys. Natural whole-score physical interpretation remains open.

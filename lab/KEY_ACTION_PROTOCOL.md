# Source attacks, estimated pedals and candidate key actions

This investigation continues the whole piano/drummer/body/costume, natural
form, original music, force/contact and continuous geometry objective.
The previous turn completed publication and committed `e260fc4`. The current
turn recovered new upstream evidence; it is concrete progress, not a wait.

The raw Chopin transcription retained 282 CC64 events. `mk.py` passed only
notes to the suite MIDI writer, losing them. Archive the source MIDI, fitted
bar/note JSON and original scripts before depending on temporary paths.
Recovery must reproduce both Chopin and full-suite MIDI bytes exactly before
mapping controllers. Keep raw channel/track/order, inferred state preceding
the 1.02-second crop, all out-of-domain events and endpoint state. Do not
invent a pedal release at the crop end or pedal states for earlier pieces.

Map through the same two bar/tick exports and the shared performance time
map. Individual note onset/offset jitter does not apply to the recovered
pedal stream; retain that uncertainty explicitly. All original 2,046 source
events, velocities, releases and soundtrack stay immutable. All source
hashes must still match after the investigation.

Assign each source attack a distinct internal ID. Trace ownership, slicing,
fingering repair, finger reuse and reach release. Record missing/duplicate/
shifted attacks as failures; a ledger accounting for them is not a playable
score. The legacy scheduled events and body centre must reproduce the frozen
baseline exactly. Check prepared numerical targets independently against the
prior committed controller. No default hold, key, pedal or scoring changes
are accepted in this investigation. Roll copies must retain IDs/touch/lift.

For each shortened action inside the Chopin domain, test uninterrupted binary
pedal coverage over its proposed release-to-source-offset interval. Report
all failures, including unknown earlier-piece coverage, instead of selecting
only covered actions. Independently compare every proposed release with the
existing soundtrack sampler's exact reset/damping rule. The cached rendering
command used no downbeats; do not substitute a different reset schedule.
Neither comparison establishes acoustic equivalence or physical truth.

Offset/pedal alignment windows are fixed diagnostic windows: 0, 10, 30 and
50 milliseconds in performed time. Keep counts for each, not a selected best
window. A possible release interval is nonunique and is not a chosen key-off.
Half pedalling, undamped treble, string/reverb dynamics, source transcription
confidence, rearticulation and actual key/foot motion require further work.

Required tests cover CC tempo/running status/channel/order, threshold63/64,
inherited crop state, uninterrupted sustain and intervening releases,
ambiguous or invalid input, nonunique offset inversion, final sampler tail,
lost repeated attacks, stage provenance, immutable source/copy identity and
ledger preservation through preparation/export. Run the entire rig suite.
Publish source recovery, hypotheses and unresolved music/contact limitations.
Native graph completion reviews the method only; preserve failed runs and
do not infer an exposed reviewer verdict from completion metadata.

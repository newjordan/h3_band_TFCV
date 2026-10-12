# Primary-source notes for source/action separation

Freshly inspected 2026-10-11. These establish model/data conventions, not the
physical correctness of our recovered performance or checkpoint settings.

The [MIDI Association CC table](https://midi.org/midi-1-0-control-change-messages)
defines CC64's binary boundary: values through 63 mean off, from 64 mean on.
Keep track/channel identity and values; do not convert them to measured feet.

ByteDance's [target-processing implementation](https://raw.githubusercontent.com/bytedance/piano_transcription/master/utils/utilities.py)
defaults to extending note offsets through a pedal interval. This supports
audible-sustain target semantics; we have not verified this checkpoint's
training invocation. Its inverse does not uniquely recover physical key-off.

Kong and colleagues' [primary paper, section III-E](https://arxiv.org/html/2010.01815v3)
describes separate note/pedal models and binary pedal estimation, excluding
half-pedal technique from that model. A successful model estimate is not a
human measurement. The [author's inference code](https://raw.githubusercontent.com/qiuqiangkong/piano_transcription_inference/master/piano_transcription_inference/inference.py)
returns separate estimated notes and pedal events and can write both to MIDI.

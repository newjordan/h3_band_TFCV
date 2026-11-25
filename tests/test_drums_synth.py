"""Drum motion from a GM file, and the additive preview synth."""
import json
from pathlib import Path

import numpy as np

from blockout import drums, drumsynth, midi, synth
from blockout.midi import Note

ROOT = Path(__file__).resolve().parents[1]


def test_hits_from_midi_use_the_gm_map():
    notes = [
        Note(0.0, 0.1, 36, 110, 0),
        Note(0.5, 0.6, 38, 100, 0),
        Note(1.0, 1.1, 46, 90, 0),
        Note(1.5, 1.6, 60, 80, 0),  # not a drum
    ]
    hits = drums.hits_from_midi(notes)
    assert [h["piece"] for h in hits] == ["kick", "snare", "hihat"]
    assert hits[2]["open"] is True
    assert hits[0]["vel"] == 110


def test_short_groove_puts_the_stick_on_the_drum(tmp_path):
    out = tmp_path / "anim.json"
    hits = drums.hits_from_midi(midi.read(ROOT / "examples" / "drums" / "rock_groove.mid"))
    window = [h for h in hits if h["t"] < 1.0]
    anim = drums.animate(window, fps=24, start=0.0, dur=1.0, schedule=[(0.0, "groove")], seed=0)
    summary = drums.check(anim)
    json.dump({"frames": len(anim["frames"])}, out.open("w"))
    assert len(anim["frames"]) == 24
    assert summary["hand_hits"] > 0
    assert summary["contact_err_mm_max"] is not None
    assert summary["contact_err_mm_max"] < 1.0
    assert summary["tip_penetrating_frames"] == 0
    assert summary["max_reach_m"] <= summary["arm_length_m"] + 1e-6


def test_synth_and_drumsynth_make_sound():
    notes = [Note(0.0, 0.4, 60, 100, 0), Note(0.5, 0.9, 64, 80, 0)]
    audio = synth.render(notes, dur=1.5)
    assert audio.shape[1] == 2
    assert audio.shape[0] == int(1.5 * synth.SR) + 1
    assert float(np.max(np.abs(audio))) > 0.1
    faster = synth.render(notes, speed=2.0, dur=1.5)
    assert float(np.max(np.abs(faster[: int(0.2 * synth.SR)]))) > 0.05

    hits = [
        {"t": 0.0, "piece": "kick", "vel": 110},
        {"t": 0.25, "piece": "snare", "vel": 100},
        {"t": 0.5, "piece": "hihat", "vel": 80, "open": False},
    ]
    drums_audio = drumsynth.render(hits, dur=1.0, seed=0)
    assert drums_audio.shape == (int(1.0 * synth.SR) + 1, 2)
    assert float(np.max(np.abs(drums_audio))) > 0.1

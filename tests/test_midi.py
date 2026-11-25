"""Standard MIDI reading: tempo, running status, and the beat grid."""
from pathlib import Path

import pytest

from blockout import drum_groove, midi
from tests.support import write_midi

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_running_status_survives_a_tempo_event(tmp_path):
    path = tmp_path / "run.mid"
    tempo = (500_000).to_bytes(3, "big")
    write_midi(path, [
        (0, bytes([0x90, 60, 100])),
        (0, bytes([0xFF, 0x51, 0x03]) + tempo),
        (240, bytes([60, 0])),          # note-off, status byte omitted
        (240, bytes([64, 80])),         # next note-on, still running
        (480, bytes([0x80, 64, 0])),
    ])
    notes = midi.read(path)
    assert [(n.pitch, n.velocity) for n in notes] == [(60, 100), (64, 80)]
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(0.25)
    assert notes[1].start == pytest.approx(0.25)
    assert notes[1].end == pytest.approx(0.5)


def test_tempo_change_is_applied(tmp_path):
    path = tmp_path / "tempo.mid"
    write_midi(path, [
        (0, bytes([0xFF, 0x51, 0x03]) + (500_000).to_bytes(3, "big")),
        (0, bytes([0x90, 60, 90])),
        (480, bytes([0x80, 60, 0])),
        (480, bytes([0xFF, 0x51, 0x03]) + (250_000).to_bytes(3, "big")),
        (480, bytes([0x90, 62, 90])),
        (960, bytes([0x80, 62, 0])),
    ])
    notes = midi.read(path)
    assert notes[0].end - notes[0].start == pytest.approx(0.5)
    assert notes[1].start == pytest.approx(0.5)
    assert notes[1].end - notes[1].start == pytest.approx(0.25)


def test_compound_meter_beats_in_dotted_quarters(tmp_path):
    path = tmp_path / "sixeight.mid"
    # 6/8, one bar, 120 bpm. Eighth = 240 ticks, bar = 1440, dotted-quarter beat = 720 ticks = 0.75 s.
    write_midi(path, [
        (0, bytes([0xFF, 0x51, 0x03]) + (500_000).to_bytes(3, "big")),
        (0, bytes([0xFF, 0x58, 0x04, 6, 3, 24, 8])),
        (0, bytes([0x90, 60, 80])),
        (1440, bytes([0x80, 60, 0])),
    ])
    beats, downbeats = midi.beats(path)
    assert downbeats == pytest.approx([0.0])
    assert beats == pytest.approx([0.0, 0.75])


def test_four_four_beat_grid(tmp_path):
    path = tmp_path / "four.mid"
    write_midi(path, [
        (0, bytes([0xFF, 0x58, 0x04, 4, 2, 24, 8])),
        (0, bytes([0x90, 60, 80])),
        (1920, bytes([0x80, 60, 0])),
    ])
    beats, downbeats = midi.beats(path)
    assert downbeats == pytest.approx([0.0])
    assert beats == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_example_piano_files_parse():
    expected = {
        "clair_de_lune.mid": 1468,
        "KV331_3_RondoAllaTurca.mid": 1614,
        "brahms_intermezzo_op118_no2.mid": 1677,
        "rach_prelude_op3_no2.mid": 1725,
        "chopin_nocturne_cs_minor_posth.mid": 940,
        "knights_suite.mid": 2046,
    }
    for name, count in expected.items():
        notes = midi.read(EXAMPLES / "piano" / name)
        assert len(notes) == count
        assert notes[0].start >= 0
        assert all(n.end >= n.start for n in notes)


def test_drum_groove_roundtrip(tmp_path):
    path = tmp_path / "groove.mid"
    written = drum_groove.notes(112, 0)
    drum_groove.write(path, written, 112)
    read = midi.read(path)
    assert len(read) == len(written) == 224
    tol = 60 / 112 / drum_groove.DIV  # one tick at this tempo; the writer rounds seconds onto the grid
    used = [False] * len(read)
    for start, pitch, vel in written:
        match = next(i for i, note in enumerate(read)
                     if not used[i] and note.pitch == pitch and note.velocity == vel
                     and abs(note.start - start) <= tol + 1e-9)
        used[match] = True
    assert all(used)


def test_shipped_drum_groove_matches_the_writer():
    notes = midi.read(EXAMPLES / "drums" / "rock_groove.mid")
    assert len(notes) == len(drum_groove.notes(112, 0)) == 224

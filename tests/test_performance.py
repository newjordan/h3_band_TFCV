"""Feeling and playback speed share one timeline, and springs move toward a target."""
import json
from pathlib import Path

import numpy as np
import pytest

from blockout import dynamics
from blockout.midi import Note
from blockout.piano import carried_note, load_performance
from tests.support import write_midi


def _piece(path, feeling):
    # Two bars of 4/4 at the default 120 bpm, plus a rest so the beat grid covers both bars.
    write_midi(path, [
        (0, bytes([0xFF, 0x58, 0x04, 4, 2, 24, 8])),
        (0, bytes([0x90, 60, 100])),
        (480, bytes([0x80, 60, 0])),
        (1920, bytes([0x90, 67, 100])),
        (2400, bytes([0x80, 67, 0])),
        (7680, bytes([0x90, 64, 40])),
        (8160, bytes([0x80, 64, 0])),
    ])
    Path(str(path) + ".feeling.json").write_text(json.dumps({
        "rows": [
            {"bar": 1, "strength": 0.2, "touch": 0.15, "freedom": 0.0, "tempo": 1.0, "head": "focused"},
            {"bar": 3, "strength": 0.8, "touch": 0.9, "freedom": 0.0, "tempo": 1.6, "head": "wild"},
        ]
    }))
    return Path(str(path) + ".feeling.json")


def test_speed_scales_a_feeling_performance_once(tmp_path):
    mid = tmp_path / "bars.mid"
    chart = _piece(mid, None)
    slow, b_slow, _, sched_slow, efn = load_performance(mid, speed=1.0, feeling=chart, seed=1)
    fast, b_fast, _, sched_fast, _ = load_performance(mid, speed=2.0, feeling=chart, seed=1)
    assert len(slow) == len(fast) == 3
    for a, b in zip(slow, fast):
        assert b.start == pytest.approx(a.start / 2)
        assert b.end == pytest.approx(a.end / 2)
        assert b.touch == pytest.approx(a.touch)
        assert b.lift == pytest.approx(a.lift)
    assert b_fast == pytest.approx([t / 2 for t in b_slow])
    assert sched_fast[0][0] == pytest.approx(sched_slow[0][0] / 2)
    assert sched_fast[-1][1] == "wild"
    # The second note sits in the faster bar, so rubato moves it. It must not also be halved twice.
    assert fast[1].start == pytest.approx(slow[1].start / 2)
    assert fast[1].start != pytest.approx(slow[1].start / 4)
    assert np.isfinite(efn(fast[1].start))


def test_speed_without_feeling_matches_the_score(tmp_path):
    mid = tmp_path / "plain.mid"
    write_midi(mid, [
        (0, bytes([0x90, 60, 90])),
        (960, bytes([0x80, 60, 0])),
    ])
    notes, beats, _, sched, efn = load_performance(mid, speed=2.0, schedule="0:focused,1:groove")
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(0.5)
    assert beats[0] == pytest.approx(0.0)
    assert beats[1] == pytest.approx(0.25)
    assert sched == [(0.0, "focused"), (1.0, "groove")]
    assert efn is None


def test_carried_note_keeps_touch():
    src = Note(1.0, 2.0, 60, 80, 0)
    src.touch, src.lift = 0.7, 1.4
    out = carried_note(src, 0.5, 1.0)
    assert (out.start, out.end, out.pitch) == (0.5, 1.0, 60)
    assert out.touch == pytest.approx(0.7)
    assert out.lift == pytest.approx(1.4)


def test_spring_moves_toward_the_target_and_stays_finite():
    so = dynamics.SecondOrder(f=2.0, z=0.8, r=0.0, x0=0.0)
    y = 0.0
    for _ in range(200):
        y = so.step(0.01, 1.0)
    assert float(np.asarray(y).reshape(-1)[0]) == pytest.approx(1.0, abs=0.05)

    track = np.zeros((30, 2))
    track[15:] = 1.0
    out = dynamics.filter_track(track, fps=24, f=3.0, z=0.7, r=0.0)
    assert out.shape == track.shape
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] > 0.5
    assert np.isfinite(out).all()


def test_drift_is_seeded():
    t = np.linspace(0, 4, 50)
    a = dynamics.drift(t, 0.02, seed=3)
    b = dynamics.drift(t, 0.02, seed=3)
    c = dynamics.drift(t, 0.02, seed=4)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.max(np.abs(a)) <= 0.02 + 1e-9

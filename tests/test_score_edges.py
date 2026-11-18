"""Short clips and a missing beats file used to drop the score the tools exist to report."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h3band"))

from h3band import stem_quality  # noqa: E402
import strum_score2  # noqa: E402


def _tone(path, seconds, sr=48000):
    n = sr * seconds
    t = np.arange(n) / sr
    amp = np.where((t < 5) | (t >= 10), 0.4, 0.01) if seconds > 5 else np.full(n, 0.2)
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([x, x], 1), sr)


class StemWindows(unittest.TestCase):
    def test_short_clip_uses_the_whole_envelope(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "short.wav")
            _tone(wav, 1)
            st, env = stem_quality.stats(wav)
        self.assertLess(len(env), 5 * stem_quality.FPS)
        hi, lo = np.percentile(env, 95), np.percentile(env, 5)
        self.assertEqual(st["win5_dyn_range_db"], round(float(hi - lo), 1))
        self.assertEqual(st["win5_silent_frac"], round(float((env < hi - 25).mean()), 3))
        self.assertTrue(np.isfinite(st["win5_dyn_range_db"]))

    def test_long_clip_keeps_five_second_windows(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "long.wav")
            _tone(wav, 12)
            st, env = stem_quality.stats(wav)
        w = 5 * stem_quality.FPS
        wins = [env[i:i + w] for i in range(0, len(env) - w + 1, w)]
        self.assertGreaterEqual(len(wins), 2)
        dyn = round(float(np.median([np.percentile(v, 95) - np.percentile(v, 5) for v in wins])), 1)
        sil = round(float(np.median([(v < np.percentile(v, 95) - 25).mean() for v in wins])), 3)
        self.assertEqual(st["win5_dyn_range_db"], dyn)
        self.assertEqual(st["win5_silent_frac"], sil)


def _motion(path, n=48):
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = np.zeros((48, 64, 3), np.uint8)
        y = 8 + (i % 12)
        frame[y:y + 6, 20:28] = (0, 220, 40)
        frame += rng.integers(0, 8, frame.shape, dtype=np.uint8)
        vw.write(frame)
    vw.release()


class StrumBeats(unittest.TestCase):
    def test_missing_default_beats_still_scores_margin_a(self):
        os.environ.pop("H3B_BEATS", None)
        with tempfile.TemporaryDirectory() as d:
            vid = os.path.join(d, "motion.mp4")
            events = os.path.join(d, "events.json")
            _motion(vid)
            Path(events).write_text('{"events": [{"t": 0.25}, {"t": 0.75}, {"t": 1.25}]}')
            r = strum_score2.analyze(vid, events, 0.0)
        self.assertIsNone(r["sync_margin_b"])
        self.assertIsNone(r["b_lock"])
        self.assertEqual(r["sync_margin"], r["sync_margin_a"])
        self.assertIsInstance(r["sync_margin"], float)
        self.assertTrue(np.isfinite(r["a_xcorr"]))

    def test_explicit_missing_beats_file_still_errors(self):
        with tempfile.TemporaryDirectory() as d:
            vid = os.path.join(d, "motion.mp4")
            events = os.path.join(d, "events.json")
            _motion(vid)
            Path(events).write_text('{"events": [{"t": 0.2}]}')
            with self.assertRaises(FileNotFoundError):
                strum_score2.analyze(vid, events, 0.0, beats_path=os.path.join(d, "nope.json"))


if __name__ == "__main__":
    unittest.main()

"""Decoded media regressions: repeats at seams must not move the performance clock."""

import json
import pathlib
import subprocess
import tempfile
import unittest
import wave
from unittest import mock

import cv2
import numpy as np

from h3_tools import skeleton_piano as piano
from h3_tools import piano_clock_audit as clock


class TimingTest(unittest.TestCase):
    def test_join_removes_overlap_in_video_and_reference(self):
        # Each frame carries its global dense index as a grey value. Three clips
        # overlap by 9, so the last contributes a different decimation phase.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = "clock_test"
            d = root / run
            d.mkdir()
            comfy = root / "input"
            (comfy / "band/skeleton").mkdir(parents=True)
            paths = []
            for k, start in enumerate((0, 15, 30)):
                frames = np.stack(
                    [
                        np.full((32, 48, 3), 30 + 3 * i, np.uint8)
                        for i in range(start, start + 24)
                    ]
                )
                path = d / f"c{k}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        "48x32",
                        "-r",
                        "24",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "0",
                        "-pix_fmt",
                        "yuv420p",
                        str(path),
                    ],
                    input=frames.tobytes(),
                    check=True,
                )
                (comfy / piano.cin(run, k, "plate")).write_bytes(path.read_bytes())
                paths.append(path)
            sound = root / "sound.wav"
            with wave.open(str(sound), "wb") as f:
                f.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                f.writeframes(np.zeros((4 * 48000, 2), "<i2").tobytes())
            config = {"clips": [24] * 3, "overlap": 9, "f0": 0, "sound": sound}
            with (
                mock.patch.dict(piano.RUNS, {run: config}),
                mock.patch.object(piano, "ROOT", root),
                mock.patch.object(piano, "COMFY_IN", comfy),
                mock.patch.object(piano, "out_mp4", side_effect=lambda r, k: paths[k]),
            ):
                piano.join(run)
            cap = cv2.VideoCapture(str(d / f"{run}.mp4"))
            values = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                values.append(float(frame.mean()))
            cap.release()
            self.assertEqual(
                len(values), 54, "join repeated overlap frames or truncated video"
            )
            np.testing.assert_allclose(values, 30 + 3 * np.arange(54), atol=3)
            meta = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-show_streams",
                        "-of",
                        "json",
                        str(d / f"{run}_sbs.mp4"),
                    ]
                )
            )
            self.assertEqual(
                int(meta["streams"][0]["nb_read_frames"]),
                54,
                "reference side-by-side must use identical trimming",
            )

    def test_final_mux_uses_exact_source_sample_not_rounded_seek(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            d = root / "final_probe"
            d.mkdir()
            video, sound = root / "video.mp4", root / "sound.wav"
            frames = np.full((32, 32, 48, 3), 80, np.uint8)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "48x32",
                    "-r",
                    "24",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                input=frames.tobytes(),
                check=True,
            )
            rng = np.random.default_rng(3)
            samples = (rng.normal(0, 2000, (48000, 2))).astype("<i2")
            with wave.open(str(sound), "wb") as f:
                f.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                f.writeframes(samples.tobytes())
            config = {
                "clips": [32],
                "f0": 0,
                "densify": 4,
                "keep": 8,
                "take_f0": 7,
                "tol": 1,
            }
            real_run = subprocess.run

            def run_ffmpeg_only(*args, **kwargs):
                # Exercise the actual final mux; edge scoring and its display
                # composites need Blender passes and are outside this test.
                command = args[0]
                if (
                    command[0] == "ffmpeg"
                    and str(d / "final_probe_final.mp4") == command[-1]
                ):
                    return real_run(*args, **kwargs)
                return subprocess.CompletedProcess(command, 0, stdout="fixture")

            with (
                mock.patch.dict(piano.RUNS, {"final_probe": config}),
                mock.patch.object(piano, "ROOT", root),
                mock.patch.object(piano, "TAKE_SOUND", sound),
                mock.patch.object(piano, "out_mp4", return_value=video),
                mock.patch.object(piano.subprocess, "run", side_effect=run_ffmpeg_only),
            ):
                piano.final_shot("final_probe")
            metadata = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "a:0",
                        "-show_entries",
                        "stream=duration_ts,time_base",
                        "-of",
                        "json",
                        str(d / "final_probe_final.mp4"),
                    ]
                )
            )["streams"][0]
            from fractions import Fraction

            playback_samples = (
                Fraction(metadata["time_base"]) * metadata["duration_ts"] * 48000
            )
            self.assertEqual(
                playback_samples,
                8 * 2000,
                "MP4 edit-list timescale rounded off source audio samples",
            )
            original, actual = clock.pcm(sound), clock.pcm(d / "final_probe_final.mp4")
            self.assertGreaterEqual(
                len(actual), 8 * 2000, "mux cut off final audio samples"
            )
            self.assertLess(
                len(actual),
                8 * 2000 + 1024,
                "unexpected surplus beyond AAC packet padding",
            )
            from scipy.signal import correlate

            offset, width, radius = 512, 10000, 64
            source = original[
                7 * 2000 + offset - radius : 7 * 2000 + offset + width + radius
            ]
            lag = (
                int(
                    np.argmax(
                        correlate(source, actual[offset : offset + width], mode="valid")
                    )
                )
                - radius
            )
            self.assertEqual(
                lag, 0, "decimal -ss introduced an avoidable source-sample offset"
            )

    def test_sample_clock_audio_keeps_exact_impulse_locations(self):
        # Seek to frame 7; exact 4x time dilation must place every impulse at
        # dense sample 4*(source_sample - frame7_sample), not a WSOLA estimate.
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = pathlib.Path(tmp) / "src.wav", pathlib.Path(tmp) / "dense.wav"
            samples = np.zeros((48000, 2), "<i2")
            start = 7 * 2000
            positions = start + np.array([123, 3881, 10210, 17821])
            samples[positions] = 24000
            with wave.open(str(src), "wb") as f:
                f.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                f.writeframes(samples.tobytes())
            self.assertTrue(
                callable(getattr(piano, "sample_clock_audio", None)),
                "missing exact sample-clock conditioning path",
            )
            piano.sample_clock_audio(src, dst, start_frame=7, frames=10, density=4)
            with wave.open(str(dst), "rb") as f:
                self.assertEqual(f.getframerate(), 48000)
                data = np.frombuffer(f.readframes(f.getnframes()), "<i2").reshape(-1, 2)
            self.assertEqual(len(data), 10 * 2000 * 4)
            for position in (positions - start) * 4:
                measured = (
                    position
                    - 12
                    + int(np.argmax(np.abs(data[position - 12 : position + 13, 0])))
                )
                self.assertEqual(measured, position)


class ClockAuditTest(unittest.TestCase):
    def run_audit(
        self,
        pts_shift=0.0,
        audio_shift=0,
        absent=False,
        audio_samples=None,
        playback_samples=18000,
        wrong_frames=False,
    ):
        # The audit must reject broken timing even when all source filenames
        # exist. Exercise the retained-frame phase after a 9-frame overlap.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = "audit_probe"
            (root / run).mkdir()
            final = root / run / f"{run}_final.mp4"
            if not absent:
                final.write_bytes(b"fixture: decoding mocked below")
            source = root / "source.mp4"
            source.touch()
            config = {
                "clips": [24, 24],
                "overlap": 9,
                "f0": 0,
                "take_f0": 0,
                "densify": 4,
                "keep": 9,
            }
            rng = np.random.default_rng(31)
            reference_audio = rng.normal(0, 0.1, 96000)
            output_audio = np.roll(reference_audio[: 9 * 2000], audio_shift)
            if audio_samples is not None:
                output_audio = output_audio[:audio_samples]
            meta = {
                "frames": [
                    {"best_effort_timestamp_time": str(i / 24 + pts_shift)}
                    for i in range(9)
                ]
            }
            expected_calls = []
            for i in (0, 5, 6, 7, 8):
                dense = 4 * i
                clip = 0 if dense < 24 else 1
                local = dense if clip == 0 else dense - 24 + 9
                expected_calls.extend([(final, i), (source, local)])
            with (
                mock.patch.dict(piano.RUNS, {run: config}),
                mock.patch.object(piano, "ROOT", root),
                mock.patch.object(piano, "out_mp4", return_value=source),
                mock.patch.object(clock, "probe", return_value=meta),
                mock.patch.object(
                    clock.subprocess,
                    "check_output",
                    return_value=json.dumps(
                        {
                            "streams": [
                                {
                                    "time_base": "1/48000",
                                    "duration_ts": playback_samples,
                                }
                            ]
                        }
                    ).encode(),
                ),
                mock.patch.object(
                    clock,
                    "decoded_frame",
                    side_effect=lambda path, index: np.full(
                        (2, 2, 3), 100 if wrong_frames and path == final else 0
                    ),
                ) as decoder,
                mock.patch.object(
                    clock, "pcm", side_effect=[output_audio, reference_audio]
                ),
            ):
                report = clock.audit(run)
            return (
                report,
                [(c.args[0], c.args[1]) for c in decoder.call_args_list],
                expected_calls,
            )

    def test_overlap_phase_is_global_not_restarted_per_clip(self):
        report, actual, expected = self.run_audit()
        self.assertEqual(actual, expected)
        self.assertEqual(report["clock_verdict"], "PASS")
        row = next(r for r in report["splice_checks"] if r["final_frame"] == 6)
        self.assertEqual(row["local_frame"], 9)
        self.assertTrue(report["execution_verdict"].startswith("UNKNOWN"))

    def test_shifted_pts_is_rejected(self):
        report, _, _ = self.run_audit(pts_shift=1 / 24)
        self.assertEqual(report["clock_verdict"], "FAIL")
        self.assertTrue(any("timestamps" in f for f in report["failures"]))

    def test_shifted_audio_is_rejected(self):
        report, _, _ = self.run_audit(audio_shift=64)
        self.assertEqual(report["clock_verdict"], "FAIL")
        self.assertEqual(
            report["audio_alignment_windows"][0]["source_minus_output_samples"], -64
        )

    def test_truncated_audio_cannot_pass_on_an_early_window(self):
        report, _, _ = self.run_audit(audio_samples=16000)
        self.assertEqual(report["clock_verdict"], "FAIL")
        self.assertTrue(any("audio duration" in f for f in report["failures"]))

    def test_short_container_duration_cannot_hide_in_decoded_padding(self):
        report, _, _ = self.run_audit(playback_samples=17984)
        self.assertEqual(report["clock_verdict"], "FAIL")
        self.assertTrue(
            any("Container audio duration" in f for f in report["failures"])
        )

    def test_grossly_wrong_splice_content_is_rejected(self):
        report, _, _ = self.run_audit(wrong_frames=True)
        self.assertEqual(report["clock_verdict"], "FAIL")
        self.assertTrue(
            any("Unexpected frame content" in f for f in report["failures"])
        )

    def test_missing_final_cannot_pass(self):
        report, actual, _ = self.run_audit(absent=True)
        self.assertEqual(report["clock_verdict"], "UNAVAILABLE")
        self.assertEqual(actual, [])


if __name__ == "__main__":
    unittest.main()

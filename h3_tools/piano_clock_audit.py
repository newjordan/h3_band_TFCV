"""Audit decoded delivery clocks, not pianist execution.

Run from the repository: python -m h3_tools.piano_clock_audit RUN --output report.json
No render is launched and existing media are never modified. A clean clock is
necessary but cannot certify anatomy, key identity, contact or graceful motion.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import pathlib
import subprocess

import cv2
import numpy as np
from scipy.signal import correlate

from h3_tools import skeleton_piano as piano


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_frames",
                "-show_entries",
                "stream=nb_frames,avg_frame_rate,time_base:frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def decoded_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode {path} frame {index}")
        return frame.astype(np.float32)
    finally:
        cap.release()


def pcm(path):
    return np.frombuffer(
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "f32le",
                "-",
            ]
        ),
        "<f4",
    )


def audit(run, final=None):
    config = piano.RUNS[run]
    if not {"keep", "take_f0", "densify"}.issubset(config):
        raise ValueError(
            "Clock audit currently requires a shot run with keep/take_f0/densify"
        )
    final = (
        pathlib.Path(final)
        if final is not None
        else piano.ROOT / run / f"{run}_final.mp4"
    )
    k, n = config["densify"], config["keep"]
    clips = piano.clip_frames(run)
    overlap = config.get("overlap", 0)
    # Joined-frame offsets (the later clips lose their leading overlap).
    segments, offset = [], 0
    for c, (_, length) in enumerate(clips):
        trim = overlap if c else 0
        segments.append((offset, offset + length - trim, c, trim))
        offset += length - trim
    report = {
        "run": run,
        "final": str(final),
        "video_fps": 24,
        "take_frame_start": config["take_f0"],
        "expected_frames": n,
        "density": k,
        "overlap": overlap,
        "joined_dense_frames": offset,
        "scope": "Digital delivery clock only; not musical/physical execution.",
        "execution_verdict": "UNKNOWN: no calibrated output contact/note tracker in this audit",
    }
    failures = []
    if not final.exists():
        report.update(
            clock_verdict="UNAVAILABLE", failures=["Missing delivered final video"]
        )
        return report
    report["final_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
    meta = probe(final)
    times = np.array([float(f["best_effort_timestamp_time"]) for f in meta["frames"]])
    report["decoded_frames"] = len(times)
    if len(times) != n:
        failures.append(f"Frame count {len(times)} != expected {n}")
    if len(times):
        error = float(np.max(np.abs(times - np.arange(len(times)) / 24)))
        report["max_pts_error_seconds"] = error
        if error > 0.000002:
            failures.append("Non-uniform or offset presentation timestamps")
    # Check both sides of every splice plus start/end. No per-clip decimation:
    # selected joined dense index is k*i, independent of splice phase.
    candidates = {0, n - 1}
    for start, _, _, _ in segments[1:]:
        first_after = math.ceil(start / k)
        candidates.update((first_after - 1, first_after, first_after + 1))
    rows = []
    for i in sorted(x for x in candidates if 0 <= x < min(n, len(times))):
        dense = k * i
        segment = next((s for s in segments if s[0] <= dense < s[1]), None)
        if segment is None:
            failures.append(f"No clip covers final frame {i}")
            continue
        start, _, c, trim = segment
        path = piano.out_mp4(run, c)
        local = dense - start + trim
        if not path.exists():
            failures.append(f"Missing source clip {c}")
            continue
        actual = decoded_frame(final, i)
        expected = decoded_frame(path, local)
        mae = float(np.mean(np.abs(actual - expected)))
        rows.append(
            {
                "final_frame": i,
                "take_frame": config["take_f0"] + i,
                "joined_dense_frame": dense,
                "clip": c,
                "local_frame": local,
                "decoded_rgb_mae": mae,
            }
        )
        # Lossy re-encoding changes pixels. This is a gross splice-identity
        # diagnostic, not a contact/hand accuracy threshold.
        if mae > 8:
            failures.append(
                f"Unexpected frame content at final frame {i}: MAE {mae:.3f}"
            )
    report["splice_checks"] = rows
    report["splice_mae_limit"] = 8
    audio_stream = json.loads(
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
                str(final),
            ]
        )
    )["streams"][0]
    playback_samples = (
        Fraction(audio_stream["time_base"]) * audio_stream["duration_ts"] * 48000
    )
    report["audio_playback_samples"] = float(playback_samples)
    if playback_samples != n * 2000:
        failures.append(
            f"Container audio duration {playback_samples} samples != expected {n * 2000}"
        )
    output_audio, source_audio = pcm(final), pcm(piano.TAKE_SOUND)
    source_start = config["take_f0"] * 2000
    expected_samples = n * 2000
    report["expected_audio_samples"] = expected_samples
    report["decoded_audio_samples"] = len(output_audio)
    # AAC can decode trailing packet padding. Never allow missing samples;
    # permit less than one AAC packet of surplus, excluded from landmarks.
    if not expected_samples <= len(output_audio) < expected_samples + 1024:
        failures.append(
            f"Final audio duration: {len(output_audio)} decoded samples, expected {expected_samples} plus <1024 padding"
        )
    checks = []
    width, radius = min(24000, expected_samples - 1024), 256
    starts = {512, expected_samples - width - 512}
    starts.update(second * 48000 + 512 for second in range(int(n / 24)))
    for begin in sorted(starts):
        if (
            width < 1024
            or begin < radius
            or begin + width > min(len(output_audio), expected_samples)
        ):
            continue
        if (
            source_start + begin - radius < 0
            or source_start + begin + width + radius > len(source_audio)
        ):
            failures.append("Source audio unavailable for alignment window")
            continue
        target = output_audio[begin : begin + width].astype(float)
        region = source_audio[
            source_start + begin - radius : source_start + begin + width + radius
        ].astype(float)
        cc = correlate(region, target, mode="valid", method="fft")
        lag = int(np.argmax(cc)) - radius
        aligned = region[lag + radius : lag + radius + width]
        norm = float(np.linalg.norm(aligned) * np.linalg.norm(target))
        score = float(np.dot(aligned, target) / norm) if norm else 0
        checks.append(
            {
                "output_start_sample": begin,
                "source_minus_output_samples": lag,
                "normalized_correlation": score,
            }
        )
    report["audio_alignment_windows"] = checks
    if not checks or any(
        abs(c["source_minus_output_samples"]) > 0 or c["normalized_correlation"] < 0.98
        for c in checks
    ):
        failures.append(
            "Final audio alignment not at zero sample lag at all checked windows"
        )
    report["audio_alignment_tolerance_samples"] = 0
    report["audio_tolerance_note"] = (
        "Zero decoded lag at checked windows, NOT proof of musical execution"
    )
    report["failures"] = failures
    report["clock_verdict"] = "PASS" if not failures else "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=sorted(piano.RUNS))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--video",
        type=pathlib.Path,
        help="Candidate final to audit; original remains untouched",
    )
    args = parser.parse_args()
    report = audit(args.run, args.video)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["clock_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

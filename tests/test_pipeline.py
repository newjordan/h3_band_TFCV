"""Scorers keep a result on short inputs, and the chain asks the v2 strum scorer."""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from h3band import stem_quality
import drum_sync_score
import h3_run
import run_chain
import shot_preflight
import strum_score2
import sync_score


def _tone(path, seconds, sr=48000):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([x, x], 1), sr)


def _video(path, n=8):
    import cv2
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    rng = np.random.default_rng(1)
    for i in range(n):
        frame = np.zeros((48, 64, 3), np.uint8)
        y = 8 + (i % 6)
        frame[y:y + 6, 20:28] = (0, 180, 40)
        frame += rng.integers(0, 6, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_stem_shorter_than_a_frame_is_finite(tmp_path):
    wav = tmp_path / "tiny.wav"
    _tone(wav, 100 / 48000)
    stats, env = stem_quality.stats(wav)
    assert len(env) >= 1
    assert np.isfinite(stats["dyn_range_db"])
    assert np.isfinite(stats["win5_dyn_range_db"])
    assert np.isfinite(stats["win5_silent_frac"])


def test_stem_cli_writes_a_table(tmp_path, capsys):
    wav = tmp_path / "a.wav"
    _tone(wav, 0.5)
    out = tmp_path / "out.json"
    import sys
    old = sys.argv
    try:
        sys.argv = ["stem_quality.py", str(out), f"dry={wav}"]
        stem_quality.main()
    finally:
        sys.argv = old
    data = json.loads(out.read_text())
    assert "dry" in data
    assert "dyn_range_db" in data["dry"]
    assert "stem" in capsys.readouterr().out


def test_strum_short_clip_still_scores(tmp_path):
    vid = tmp_path / "short.mp4"
    events = tmp_path / "events.json"
    _video(vid, 8)
    events.write_text(json.dumps({"events": [{"t": 0.1}, {"t": 0.2}]}))
    os.environ.pop("H3B_BEATS", None)
    result = strum_score2.analyze(str(vid), str(events), 0.0)
    assert result["frames"] == 8
    assert result["p_a"] is None
    assert result["sync_margin_b"] is None
    assert isinstance(result["sync_margin"], float)
    assert np.isfinite(result["sync_margin"])


def test_drum_sync_short_clip(tmp_path):
    vid = tmp_path / "short.mp4"
    hits = tmp_path / "hits.json"
    _video(vid, 8)
    hits.write_text(json.dumps({"hits": [
        {"t": 0.1, "piece": "snare", "vel": 100},
        {"t": 0.3, "piece": "kick", "vel": 110},
    ]}))
    result = drum_sync_score.analyze(str(vid), str(hits), 0.0)
    assert result["frames"] == 8
    assert result["hits"] == 1  # kick is not a stick
    assert result["p"] is None
    assert np.isfinite(result["sync_margin"])


def test_chain_scores_guitar_with_strum_v2(tmp_path):
    pre = {"song_window": [1.5, 6.0], "stems": {"vocals": {"out": "v.wav"}}}
    guitar = run_chain.score_command({"name": "g", "stem": "guitar", "box_start": [1, 2, 3, 4]}, "take.mp4", pre)
    vocal = run_chain.score_command({"name": "v", "stem": "vocals", "box_start": [0, 0, 10, 10]}, "take.mp4", pre)
    assert guitar[-1] == "--json"
    assert "strum_score2.py" in guitar[1]
    assert "sync_score.py" in vocal[1]
    assert run_chain.score_command({"name": "x", "stem": "theremin", "box_start": [0, 0, 1, 1]}, "t.mp4", pre) is None


def test_h3_job_dump_builds_the_lock_graph(capsys):
    job = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "band_aid_skeleton_job.json"
    assert h3_run.main([str(job), "--dump"]) == 0
    captured = capsys.readouterr()
    graph = json.loads(captured.out)
    kinds = {node["class_type"] for node in graph.values()}
    assert "H3BandZoneLatent" in kinds
    assert "MiniMaxH3FunControlNetApply" in kinds
    assert "SaveVideo" in kinds
    zone = next(node for node in graph.values() if node["class_type"] == "H3BandZoneLatent")
    assert zone["inputs"]["audio_denoise"] == 0.0
    assert zone["inputs"]["video_denoise"] == h3_run.video_mask({"video_strength": 0.88})
    image = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ImageToVideo")
    assert "first_frame" not in image["inputs"]
    assert "nodes" in captured.err
    assert "video mask" in captured.err


def test_video_mask_ceils_onto_the_token_grid():
    strength = 0.88
    raw = h3_run.H3_VIDEO_SHIFT * strength / (1.0 + (h3_run.H3_VIDEO_SHIFT - 1.0) * strength)
    masked = h3_run.video_mask({"video_strength": strength})
    assert 0.0 < masked <= 1.0
    assert masked + 1e-12 >= raw
    assert abs(masked * 256 - round(masked * 256)) < 1e-6
    assert h3_run.video_mask({"video_denoise": 0.5}) == 0.5
    assert h3_run.video_mask({}) == 1.0


def test_frame_grid_snaps_to_the_h3_length():
    assert shot_preflight.align_frame_count(124) == 124
    assert shot_preflight.align_frame_count(10) == 22
    assert shot_preflight.align_frame_count(1) == 5
    assert shot_preflight.latent_index(0) == 0
    assert shot_preflight.latent_index(1) == 1
    assert shot_preflight.latent_index(4) == 1
    assert shot_preflight.latent_index(5) == 2
    assert shot_preflight.latent_index(17) == 5


def test_zone_box_marks_tokens_it_touches():
    zone = {"box_start": "0,0,64,32", "box_end": "0,0,64,32"}
    tokens, _boxes, (b0, b1) = shot_preflight.zone_tokens(zone, 5, 2, 128, 64)
    assert tokens.shape[0] == 2
    assert b0 == [0.0, 0.0, 64.0, 32.0]
    assert tokens[0].sum() > 0
    assert not tokens[0, -1, -1]


def test_vocal_envelope_survives_a_tiny_wav(tmp_path):
    wav = tmp_path / "tiny.wav"
    _tone(wav, 0.001)
    env = sync_score.vocal_envelope(wav, 24, 4)
    assert env.shape == (4,)
    assert np.isfinite(env).all()

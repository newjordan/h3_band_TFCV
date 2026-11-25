"""Lock check: matching edges score, a blank frame does not crash, every frame is returned."""
import numpy as np

from comfy_nodes.lock_core import edges_from_rgb, lock_frames


def _plate(shift=0, n=3, h=48, w=64):
    frames = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        x = 16 + shift
        frames[i, 10:30, x:x + 8] = 220
    mask = np.zeros((n, h, w), np.float32)
    mask[:, 4:40, 8:48] = 1.0
    return frames, mask


def test_identical_geometry_locks():
    plate, mask = _plate()
    recall, precision, report, overlay = lock_frames(plate, plate, mask, tolerance_px=3)
    assert overlay.shape == plate.shape
    assert recall > 0.9
    assert precision > 0.9
    assert "lock check" in report
    red = overlay[0, :, :, 0] == 255
    assert red.any()


def test_shifted_geometry_scores_below_the_match():
    plate, mask = _plate()
    shifted, _ = _plate(shift=12)
    same, _, _, _ = lock_frames(plate, plate, mask)
    moved, _, _, overlay = lock_frames(shifted, plate, mask)
    assert moved < same
    assert overlay.shape[0] == plate.shape[0]


def test_blank_frames_return_an_overlay_and_zero():
    blank = np.zeros((2, 32, 32, 3), np.uint8)
    mask = np.ones((2, 32, 32), np.float32)
    recall, precision, report, overlay = lock_frames(blank, blank, mask)
    assert (recall, precision) == (0.0, 0.0)
    assert overlay.shape == blank.shape
    assert "no reference edges" in report


def test_no_frames():
    empty = np.zeros((0, 8, 8, 3), np.uint8)
    mask = np.zeros((0, 8, 8), np.float32)
    recall, _, report, overlay = lock_frames(empty, empty, mask)
    assert recall == 0.0
    assert report == "no frames"
    assert overlay.shape[0] == 0


def test_edges_follow_the_rectangle():
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[10:20, 10:20] = 255
    edges = edges_from_rgb(frame, 20, 60, 1)
    assert edges[10:20, 10:20].any()
    assert not edges[0, 0]

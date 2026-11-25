"""Numpy/OpenCV half of the Band Aid lock nodes.

Blockout Reference Edges and Blockout Lock Check both measure the same Canny edges. Keeping that math
here means it can run without ComfyUI or torch, and a frame with no edges still returns an overlay
instead of raising on an empty stack.
"""
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def need_cv2():
    if cv2 is None:
        raise RuntimeError("Band Aid lock nodes need opencv-python: pip install opencv-python-headless")


def edges_from_rgb(rgb_u8, low, high, blur):
    """RGB uint8 frame -> boolean edge mask. `blur` is the Gaussian kernel size (forced odd)."""
    need_cv2()
    grey = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    if blur > 1:
        k = int(blur) | 1
        grey = cv2.GaussianBlur(grey, (k, k), 0)
    return cv2.Canny(grey, low, high) > 0


def _fit_mask(mask, width, height):
    m = np.asarray(mask, dtype=np.float32)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m


def lock_frames(render_u8, normal_u8, mask, tolerance_px=3.0, mask_dilate_px=7, min_edges=20):
    """How well `render_u8` sits on `normal_u8`, inside `mask`.

    render_u8, normal_u8: uint8 RGB (N, H, W, 3). mask: float (N, H, W), on where > 0.5.
    Returns (recall, precision, report, overlay_u8). recall and precision are the median over
    frames that have at least `min_edges` reference edges inside the dilated mask. The overlay
    has one frame per input frame (reference edges in red over the render), including frames
    that are too sparse to score.
    """
    need_cv2()
    render_u8 = np.asarray(render_u8)
    normal_u8 = np.asarray(normal_u8)
    mask = np.asarray(mask)
    n = min(len(render_u8), len(normal_u8), len(mask))
    if n == 0:
        empty = np.zeros((0, 1, 1, 3), np.uint8)
        return 0.0, 0.0, "no frames", empty
    height, width = render_u8.shape[1], render_u8.shape[2]
    kernel = np.ones((2 * int(mask_dilate_px) + 1,) * 2, np.uint8)
    recall, precision, worst = [], [], (2.0, -1)
    overlay = np.empty((n, height, width, 3), np.uint8)
    for i in range(n):
        ref = normal_u8[i]
        if ref.shape[0] != height or ref.shape[1] != width:
            ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_AREA)
        region = _fit_mask(mask[i], width, height)
        region = cv2.dilate((region > 0.5).astype(np.uint8), kernel) > 0
        render_edges = edges_from_rgb(render_u8[i], 20, 60, 3) & region
        ref_edges = edges_from_rgb(ref, 20, 60, 3) & region
        frame = render_u8[i].copy()
        frame[ref_edges] = (255, 0, 0)
        if int(ref_edges.sum()) >= min_edges:
            dist_render = cv2.distanceTransform((~render_edges).astype(np.uint8), cv2.DIST_L2, 3)
            dist_ref = cv2.distanceTransform((~ref_edges).astype(np.uint8), cv2.DIST_L2, 3)
            rec = float(np.mean(dist_render[ref_edges] <= tolerance_px))
            prec = float(np.mean(dist_ref[render_edges] <= tolerance_px)) if render_edges.any() else 0.0
            recall.append(rec)
            precision.append(prec)
            worst = min(worst, (rec, i))
            cv2.putText(frame, f"f{i} lock {100 * rec:.0f}%", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        overlay[i] = frame
    if not recall:
        return 0.0, 0.0, "no reference edges inside the mask", overlay
    rec_a, pre_a = np.array(recall), np.array(precision)
    report = (f"Band Aid lock check, {len(rec_a)} frames, tolerance {tolerance_px:g} px: reference edges matched "
              f"p10/p50/p90 {np.percentile(rec_a, 10):.2f} / {np.median(rec_a):.2f} / {np.percentile(rec_a, 90):.2f}; "
              f"render edges on the reference p50 {np.median(pre_a):.2f}; worst frame {worst[1]} ({worst[0]:.2f}). "
              f"Ceiling (a Blender plate of the same geometry) ~0.97.")
    return float(np.median(rec_a)), float(np.median(pre_a)), report, overlay

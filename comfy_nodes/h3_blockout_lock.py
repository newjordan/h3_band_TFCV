"""Band Aid: lock MiniMax H3 to a blockout (a reference performance rendered in Blender).

The reference is the motion: a grey plate of the animated performer, its normal pass and its mask. H3 paints the look
over it (fl2va mode B, the plate at video_strength ~0.88, H3 Fun ControlNet Union 2.0 at strength 1.0 on the edges of
the normal pass). These two nodes make the control and measure the lock:

  Blockout Reference Edges  IMAGE (the blockout's normal pass) -> IMAGE (Canny edges: the Fun ControlNet's control)
  Blockout Lock Check       IMAGE (the H3 render) + IMAGE (the normal pass) + MASK (the performer's mask)
                            -> lock score, report, and an overlay (the reference edges in red over the render)

Lock score: the share of the reference's edge pixels (inside the dilated mask) that have an edge of the render within
`tolerance_px` (median over frames), and the reverse (does the render draw where the reference has nothing). A
different rendering of the same geometry (the Blender grey plate) scores ~0.97; that is the ceiling.
Needs opencv-python (cv2). Images are ComfyUI IMAGE tensors [frames, H, W, 3] in 0..1; MASK is [frames, H, W].
"""
import numpy as np
import torch

from .lock_core import edges_from_rgb, lock_frames, need_cv2


def _u8_rgb(img):
    """One IMAGE frame (H, W, 3) float 0..1 -> uint8 RGB."""
    return (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


class BlockoutReferenceEdges:
    """The Fun ControlNet's control from a blockout: Canny of its normal pass. The normal pass has an edge at every
    finger and joint (a grey beauty pass has almost none on the fingers), so this is what pins the hands."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"normal_pass": ("IMAGE",),
                             "low": ("INT", {"default": 20, "min": 0, "max": 255}),
                             "high": ("INT", {"default": 60, "min": 0, "max": 255}),
                             "blur": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2})},
                "optional": {"mask": ("MASK",),
                             "mask_dilate_px": ("INT", {"default": 0, "min": 0, "max": 128})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control",)
    FUNCTION = "run"
    CATEGORY = "Band Aid"

    def run(self, normal_pass, low, high, blur, mask=None, mask_dilate_px=0):
        need_cv2()
        import cv2
        out = []
        for i in range(normal_pass.shape[0]):
            e = edges_from_rgb(_u8_rgb(normal_pass[i]), low, high, blur)
            if mask is not None and mask_dilate_px > 0:
                m = (mask[min(i, mask.shape[0] - 1)].cpu().numpy() > 0.5).astype(np.uint8)
                k = np.ones((2 * mask_dilate_px + 1,) * 2, np.uint8)
                e &= cv2.dilate(m, k) > 0
            out.append(torch.from_numpy(np.repeat(e[..., None], 3, axis=2).astype(np.float32)))
        return (torch.stack(out),)


class BlockoutLockCheck:
    """How well a render sits on the blockout: per frame, inside the performer's (dilated) mask, the distance from
    each reference edge to the nearest render edge (recall) and back (precision)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"render": ("IMAGE",), "normal_pass": ("IMAGE",), "mask": ("MASK",),
                             "tolerance_px": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 32.0, "step": 0.5}),
                             "mask_dilate_px": ("INT", {"default": 7, "min": 0, "max": 64})}}

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("lock_recall", "lock_precision", "report", "overlay")
    FUNCTION = "run"
    CATEGORY = "Band Aid"

    def run(self, render, normal_pass, mask, tolerance_px, mask_dilate_px):
        need_cv2()
        n = min(render.shape[0], normal_pass.shape[0], mask.shape[0])
        render_u8 = np.stack([_u8_rgb(render[i]) for i in range(n)]) if n else np.zeros((0, 1, 1, 3), np.uint8)
        normal_u8 = np.stack([_u8_rgb(normal_pass[i]) for i in range(n)]) if n else np.zeros((0, 1, 1, 3), np.uint8)
        mask_np = mask[:n].detach().cpu().numpy() if n else np.zeros((0, 1, 1), np.float32)
        recall, precision, report, overlay = lock_frames(
            render_u8, normal_u8, mask_np, tolerance_px, mask_dilate_px)
        return (float(recall), float(precision), report, torch.from_numpy(overlay.astype(np.float32) / 255.0))


NODE_CLASS_MAPPINGS = {"BandAidBlockoutReferenceEdges": BlockoutReferenceEdges,
                       "BandAidBlockoutLockCheck": BlockoutLockCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"BandAidBlockoutReferenceEdges": "Blockout Reference Edges (Band Aid)",
                              "BandAidBlockoutLockCheck": "Blockout Lock Check (Band Aid)"}

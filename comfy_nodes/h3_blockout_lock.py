"""Band Aid: lock MiniMax H3 to a blockout (a reference performance rendered in Blender).

The reference is the motion: a grey plate of the animated performer, its normal pass and its mask. H3 paints the look
over it (fl2va mode B, the plate at video_strength ~0.88, H3 Fun ControlNet Union 2.0 at strength 1.0 on the edges of
the normal pass). These two nodes make the control and measure the lock:

  Blockout Pose Control     the rig's joints (blender_piano.py --pose2d JSON) + IMAGE (the normal pass) + MASK (the hands)
                            -> IMAGE: the overlay control. An OpenPose hand skeleton of the stock hand's joints over the
                            keys' edges, hands blanked: where every joint and key is, nothing about the hand's look
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

try:
    import cv2
except ImportError:                                   # reported when a node runs, so the package still loads
    cv2 = None


def _need_cv2():
    if cv2 is None:
        raise RuntimeError("Band Aid lock nodes need opencv-python: pip install opencv-python-headless")


def _u8(img):
    """One IMAGE frame (H, W, 3) float 0..1 -> uint8 grey."""
    a = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def _edges(grey, low, high, blur):
    if blur > 1:
        grey = cv2.GaussianBlur(grey, (blur | 1, blur | 1), 0)
    return cv2.Canny(grey, low, high) > 0


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
        _need_cv2()
        out = []
        for i in range(normal_pass.shape[0]):
            e = _edges(_u8(normal_pass[i]), low, high, blur)
            if mask is not None and mask_dilate_px > 0:
                m = (mask[min(i, mask.shape[0] - 1)].cpu().numpy() > 0.5).astype(np.uint8)
                k = np.ones((2 * mask_dilate_px + 1,) * 2, np.uint8)
                e &= cv2.dilate(m, k) > 0
            out.append(torch.from_numpy(np.repeat(e[..., None], 3, axis=2).astype(np.float32)))
        return (torch.stack(out),)


class BlockoutPoseControl:
    """The standard-hand overlay control: the stock MPFB2 hand's 21 joints per hand and the elbow (projected through the
    shot camera by blockout/blender_piano.py --pose2d) drawn as an OpenPose hand skeleton, over the Canny edges of the
    normal pass with the hand region blanked. With a plate of the piano alone (--hide hands) and the look first in the
    prompt, H3 paints any look (skeleton, monster, robot, paw, real) on the rig's joints, and the keys go down."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_json": ("STRING", {"default": "band/pose2d.json",
                                                      "tooltip": "blender_piano.py --pose2d output, absolute or under ComfyUI's input folder"}),
                             "normal_pass": ("IMAGE",), "mask": ("MASK",),
                             "first_frame": ("INT", {"default": 0, "min": 0, "max": 1 << 20,
                                                     "tooltip": "The JSON frame that goes with the first image"}),
                             "hand_blank_px": ("INT", {"default": 12, "min": 0, "max": 64})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control",)
    FUNCTION = "run"
    CATEGORY = "Band Aid"

    def run(self, pose_json, normal_pass, mask, first_frame, hand_blank_px):
        _need_cv2()
        import json, os, sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(here, "h3_tools"))
        import pose_control
        path = pose_json
        if not os.path.isabs(path):
            import folder_paths
            path = os.path.join(folder_paths.get_input_directory(), path)
        P = json.load(open(path))
        frames = P["frames"]
        H, W = normal_pass.shape[1], normal_pass.shape[2]
        sx, sy = W / P["size"][0], H / P["size"][1]
        k = np.ones((2 * hand_blank_px + 1,) * 2, np.uint8)
        out = []
        for i in range(normal_pass.shape[0]):
            e = _edges(_u8(normal_pass[i]), 20, 60, 3)
            m = mask[min(i, mask.shape[0] - 1)].cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            e &= ~(cv2.dilate((m > 0.5).astype(np.uint8), k) > 0)
            img = np.zeros((H, W, 3), np.uint8)
            img[e] = 255
            hands = {s: v for s, v in frames.get(str(first_frame + i), {}).items() if s in ("L", "R")}   # not "lens"
            for side in sorted(hands, key=lambda s: -np.mean([q[2] for q in hands[s]["hand"]])):
                hand = [[x * sx, y * sy, z] for x, y, z in hands[side]["hand"]]
                el = hands[side]["elbow"]
                pose_control.draw(img, hand, [el[0] * sx, el[1] * sy, el[2]], side, W / 1280)
            out.append(torch.from_numpy(img[..., ::-1].astype(np.float32) / 255.0))   # drawn in BGR -> IMAGE RGB
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
        _need_cv2()
        n = min(render.shape[0], normal_pass.shape[0], mask.shape[0])
        H, W = render.shape[1], render.shape[2]
        k = np.ones((2 * mask_dilate_px + 1,) * 2, np.uint8)
        rec, pre, worst, over = [], [], (2.0, -1), []
        for i in range(n):
            ref = normal_pass[i]
            if ref.shape[0] != H or ref.shape[1] != W:
                ref = torch.nn.functional.interpolate(ref.permute(2, 0, 1)[None], size=(H, W), mode="area")[0].permute(1, 2, 0)
            m = mask[i].cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = cv2.dilate((m > 0.5).astype(np.uint8), k) > 0
            er, en = _edges(_u8(render[i]), 20, 60, 3) & m, _edges(_u8(ref), 20, 60, 3) & m
            ov = (render[i].cpu().numpy() * 255).astype(np.uint8).copy()
            ov[en] = (255, 0, 0)
            if en.sum() >= 20:
                d_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
                d_n = cv2.distanceTransform((~en).astype(np.uint8), cv2.DIST_L2, 3)
                r = float(np.mean(d_r[en] <= tolerance_px))
                p = float(np.mean(d_n[er] <= tolerance_px)) if er.any() else 0.0
                rec.append(r); pre.append(p)
                worst = min(worst, (r, i))
                cv2.putText(ov, f"f{i} lock {100 * r:.0f}%", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            over.append(torch.from_numpy(ov.astype(np.float32) / 255.0))
        if not rec:
            return (0.0, 0.0, "no reference edges inside the mask", torch.stack(over))
        R, P = np.array(rec), np.array(pre)
        report = (f"Band Aid lock check, {len(R)} frames, tolerance {tolerance_px:g} px: reference edges matched "
                  f"p10/p50/p90 {np.percentile(R, 10):.2f} / {np.median(R):.2f} / {np.percentile(R, 90):.2f}; "
                  f"render edges on the reference p50 {np.median(P):.2f}; worst frame {worst[1]} ({worst[0]:.2f}). "
                  f"Ceiling (a Blender plate of the same geometry) ~0.97.")
        return (float(np.median(R)), float(np.median(P)), report, torch.stack(over))


NODE_CLASS_MAPPINGS = {"BandAidBlockoutPoseControl": BlockoutPoseControl,
                       "BandAidBlockoutReferenceEdges": BlockoutReferenceEdges,
                       "BandAidBlockoutLockCheck": BlockoutLockCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"BandAidBlockoutPoseControl": "Blockout Pose Control (Band Aid)",
                              "BandAidBlockoutReferenceEdges": "Blockout Reference Edges (Band Aid)",
                              "BandAidBlockoutLockCheck": "Blockout Lock Check (Band Aid)"}

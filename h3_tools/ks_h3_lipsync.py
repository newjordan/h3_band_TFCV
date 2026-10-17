"""Knight Shift: lock a vocal into MiniMax-H3's own audio stream (and optionally re-noise an existing take).

KSH3LipSyncLatent: takes an H3 AV latent (from MiniMaxH3ImageToVideo / EmptyMiniMaxH3LatentAV), encodes AUDIO with
the H3 audio VAE into the target audio stream, optionally swaps in a VAE-encoded existing take as the video stream,
and sets per-stream noise masks (upstream per-token mask support, PR #15375): audio_denoise 0 = frozen vocal clock,
video_denoise 1 = generate, 0.35 = re-sync an existing take while keeping its motion.
"""
import logging
import os
import torch
import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio

# KS_H3_STOCK_MATMUL=1: run every quantized layer (NVFP4 DiT, NVFP4-AWQ text encoder) as dequantized weights x bf16
# activations, i.e. no NVFP4/FP8/INT8 activation quantization anywhere (the closest to stock bf16 these weights allow).
if os.environ.get("KS_H3_STOCK_MATMUL") == "1":
    import comfy.ops
    from comfy.quant_ops import QUANT_ALGOS
    _orig_disabled = comfy.ops.get_disabled_quant_formats

    def _all_formats_emulated(device=None):
        return set(_orig_disabled(device)) | set(QUANT_ALGOS.keys())

    comfy.ops.get_disabled_quant_formats = _all_formats_emulated
    logging.info("[KSH3LipSync] KS_H3_STOCK_MATMUL=1: quantized matmul disabled, weights dequantized to bf16 compute")


class KSH3LipSyncLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "audio_vae": ("VAE",), "audio": ("AUDIO",),
                             "video_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "audio_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})},
                "optional": {"video_latent": ("LATENT",),
                             "face_box_start": ("STRING", {"default": ""}),
                             "face_box_end": ("STRING", {"default": ""}),
                             "face_denoise": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                             # lockstep: one video mask value per latent frame ("0.62,0.92,0.92,0.92,0.62,..."); overrides
                             # video_denoise frame by frame (low = held hard to the plate, 0 = pinned clean)
                             "frame_masks": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "ks/minimax"

    def run(self, latent, audio_vae, audio, video_denoise, audio_denoise, video_latent=None,
            face_box_start="", face_box_end="", face_denoise=0.92, frame_masks=""):
        s = latent["samples"]
        if not getattr(s, "is_nested", False) or len(s.tensors) != 2:
            raise ValueError("KSH3LipSyncLatent expects an H3 AV latent")
        video, aud = s.tensors[0], s.tensors[1]
        if video_latent is not None:
            v = video_latent["samples"]
            if getattr(v, "is_nested", False):
                v = v.tensors[0]
            if tuple(v.shape) != tuple(video.shape):
                raise ValueError("video_latent shape {} != target {}".format(tuple(v.shape), tuple(video.shape)))
            video = v.to(device=video.device, dtype=video.dtype)
        z, rt = _encode_ref_audio(audio_vae, audio)
        z = z.to(device=aud.device, dtype=aud.dtype)
        T = aud.shape[-1]
        amask = torch.full_like(aud, float(audio_denoise))
        if rt >= T:
            z = z[..., :T]
        else:
            z = torch.cat([z, torch.zeros(z.shape[:-1] + (T - rt,), dtype=z.dtype, device=z.device)], dim=-1)
            amask[..., rt:] = 1.0
        if z.shape[0] != aud.shape[0]:
            z = z.expand(aud.shape[0], *z.shape[1:]).clone()
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video.clone(), z.contiguous()))
        vmask = torch.full_like(video, float(video_denoise))
        if frame_masks.strip():
            fm = [float(v) for v in frame_masks.split(",")]
            for k in range(min(len(fm), video.shape[2])):
                vmask[:, :, k] = fm[k]
            logging.info("[KSH3LipSync] lockstep frame masks %s", ",".join(f"{v:.3f}" for v in fm[:video.shape[2]]))
        if face_box_start.strip():
            # box (pixel x0,y0,x1,y1) moving linearly from the first to the last frame; latent k covers
            # FRAME_PER_TOKEN frames (1,4,4,4,4 pattern), 16 px per latent cell
            b0 = [float(v) for v in face_box_start.split(",")]
            b1 = [float(v) for v in (face_box_end or face_box_start).split(",")]
            fpt = (1, 4, 4, 4, 4)
            lt, lh, lw = video.shape[2], video.shape[3], video.shape[4]
            starts, f = [], 0
            for k in range(lt):
                starts.append(f); f += fpt[k % 5]
            nframes = f
            for k in range(lt):
                u = (starts[k] + (fpt[k % 5] - 1) / 2.0) / max(1, nframes - 1)
                x0, y0, x1, y1 = [a + (b - a) * u for a, b in zip(b0, b1)]
                cx0, cy0 = max(0, int(x0 // 16)), max(0, int(y0 // 16))
                cx1, cy1 = min(lw, int(-(-x1 // 16))), min(lh, int(-(-y1 // 16)))
                vmask[:, :, k, cy0:cy1, cx0:cx1] = float(face_denoise)
            logging.info("[KSH3LipSync] face box %s -> %s at denoise %.2f (rest %.2f)", b0, b1, face_denoise, video_denoise)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))
        logging.info("[KSH3LipSync] video %s denoise %.2f | audio latent %d of %d frames, denoise %.2f, swapped_video=%s",
                     tuple(video.shape), video_denoise, min(rt, T), T, audio_denoise, video_latent is not None)
        return (out,)


NODE_CLASS_MAPPINGS = {"KSH3LipSyncLatent": KSH3LipSyncLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"KSH3LipSyncLatent": "KS H3 Lip-Sync Latent (lock vocal stream)"}

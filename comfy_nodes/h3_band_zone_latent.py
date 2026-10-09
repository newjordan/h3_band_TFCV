"""H3 Band Zone Latent: lock a track into MiniMax H3's own audio stream, with per-zone video denoise.

Takes an H3 AV latent (from MiniMaxH3ImageToVideo / EmptyMiniMaxH3LatentAV), encodes AUDIO with the H3
audio VAE into the target audio stream, optionally swaps in a VAE-encoded existing take as the video
stream, and sets per-stream noise masks:
  audio_denoise 0   -> the track is frozen: clean and pinned in the audio stream from the first step
  video_denoise 1   -> generate; 0 -> lock the take; in between -> run those tokens at that fraction of sigma
  zone box          -> pixel x0,y0,x1,y1, moving linearly from the first to the last frame; the box runs at
                       zone_denoise while the rest of the frame runs at video_denoise. 16 px per latent cell.

Requires a ComfyUI with MiniMax H3 support and per-token noise masks on nested AV latents
(upstream ComfyUI PR #15375). Install: copy this file into ComfyUI/custom_nodes/.

H3BAND_STOCK_MATMUL=1 runs every quantized layer as dequantized weights x bf16 activations
(no activation quantization), the closest to stock bf16 that quantized weights allow.
"""
import logging
import os

import comfy.nested_tensor
import torch
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio

if os.environ.get("H3BAND_STOCK_MATMUL") == "1":
    import comfy.ops
    from comfy.quant_ops import QUANT_ALGOS
    _orig_disabled = comfy.ops.get_disabled_quant_formats

    def _all_formats_emulated(device=None):
        return set(_orig_disabled(device)) | set(QUANT_ALGOS.keys())

    comfy.ops.get_disabled_quant_formats = _all_formats_emulated
    logging.info("[H3BandZoneLatent] H3BAND_STOCK_MATMUL=1: quantized matmul disabled, weights dequantized to bf16 compute")


class H3BandZoneLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "audio_vae": ("VAE",), "audio": ("AUDIO",),
                             "video_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "audio_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})},
                "optional": {"video_latent": ("LATENT",),
                             "zone_box_start": ("STRING", {"default": ""}),
                             "zone_box_end": ("STRING", {"default": ""}),
                             "zone_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "h3band"

    def run(self, latent, audio_vae, audio, video_denoise, audio_denoise, video_latent=None,
            zone_box_start="", zone_box_end="", zone_denoise=1.0):
        s = latent["samples"]
        if not getattr(s, "is_nested", False) or len(s.tensors) != 2:
            raise ValueError("H3BandZoneLatent expects an H3 AV latent")
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
        if zone_box_start.strip():
            # latent k covers FRAME_PER_TOKEN frames (1,4,4,4,4 pattern), 16 px per latent cell
            b0 = [float(v) for v in zone_box_start.split(",")]
            b1 = [float(v) for v in (zone_box_end or zone_box_start).split(",")]
            fpt = (1, 4, 4, 4, 4)
            lt, lh, lw = video.shape[2], video.shape[3], video.shape[4]
            starts, f = [], 0
            for k in range(lt):
                starts.append(f)
                f += fpt[k % 5]
            nframes = f
            for k in range(lt):
                u = (starts[k] + (fpt[k % 5] - 1) / 2.0) / max(1, nframes - 1)
                x0, y0, x1, y1 = [a + (b - a) * u for a, b in zip(b0, b1)]
                cx0, cy0 = max(0, int(x0 // 16)), max(0, int(y0 // 16))
                cx1, cy1 = min(lw, int(-(-x1 // 16))), min(lh, int(-(-y1 // 16)))
                vmask[:, :, k, cy0:cy1, cx0:cx1] = float(zone_denoise)
            logging.info("[H3BandZoneLatent] zone %s -> %s at denoise %.2f (rest %.2f)", b0, b1, zone_denoise, video_denoise)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))
        logging.info("[H3BandZoneLatent] video %s denoise %.2f | audio latent %d of %d frames, denoise %.2f, swapped_video=%s",
                     tuple(video.shape), video_denoise, min(rt, T), T, audio_denoise, video_latent is not None)
        return (out,)


NODE_CLASS_MAPPINGS = {"H3BandZoneLatent": H3BandZoneLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"H3BandZoneLatent": "H3 Band Zone Latent (lock audio stream, per-zone denoise)"}

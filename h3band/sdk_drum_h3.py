#!/usr/bin/env python3
"""MiniMax H3 drummer takes over the drum blockout, run on ComfyUI through the Comfy SDK.

  sdk_drum_h3.py OUT.mp4 --midi GROOVE.mid [--audio DRUMS.wav] [--blockout | --no-blockout] [--denoise D]
                 [--keyframe FRAME.png] [--start S] [--frames 124] [--seed 42] [--steps 8] [--prompt TEXT]
                 [--motion smooth|snap|loose] [--teach] [--emotion 0:calm,3:intense]
                 [--force F] [--range R] [--body B] [--flair X]

One graph: the drum stem is trimmed to [start, start + frames/24] and frozen into H3's audio stream by
H3 Band Zone Latent (audio_denoise 0). The MIDI's hits drive H3 Band Drum Blockout over the same window.
Without --audio, H3 Band Drum Synth renders the MIDI as the drum stem. --teach runs H3 Band Drum Teacher on
the hits first, so the blockout plays chokes, rimshots, cross-sticks, bells and flams and the synth sounds them.
--emotion and the sliders set the blockout's stroke heights, arm lift and body (blockout/drum_style.py).
  --blockout     the grey blockout, VAE-encoded, is the starting video latent, sampled over the last --steps
                 of round(steps / denoise) steps: H3 re-skins the blockout and keeps its stick timing
                 (0.6-0.8 on the example groove; at 0.89 the timing is gone)
  --no-blockout  the video starts from noise (denoise 1): what H3 does from the audio alone
--keyframe pins frame 0 (fl2va first frame); use the same one for both to compare like with like.
Stack: w4a8 fl2va DiT, Turbo v4 LoRA + Turbo Sampler, realism LoRA, sigma shift 12/3 (h3_workflow_build.py).

COMFY_BASE_URL defaults to http://127.0.0.1:8189 (comfy-api-proxy in front of the ComfyUI).
"""
import argparse, os

from comfy_sdk import Comfy, Progress, StatusChange

PROMPT = ("r34l1sm Live-action concert footage, cinematic. A drummer in a black t-shirt plays a black rock drum "
          "kit on a dark stage under a warm spotlight, seen from the front. Two wooden drumsticks strike the snare, "
          "hi-hat, toms and cymbals exactly on each drum hit, the right foot works the kick pedal, the cymbals "
          "shake when struck. Realistic motion, sharp detail. The only sound is the drum kit.")
W, H, FPS = 1280, 736, 24
SLIDERS = ("force", "range", "body", "flair")     # blockout/drum_style.py


def total_steps(a):
    return max(a.steps, round(a.steps / a.denoise)) if a.blockout else a.steps


def build(a):
    g = {}

    def n(nid, cls, **inputs):
        g[str(nid)] = {"class_type": cls, "inputs": inputs}
        return [str(nid), 0]

    model = n(1, "UNETLoader", unet_name="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", weight_dtype="default")
    clip = n(2, "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    vae = n(3, "VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    avae = n(4, "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    model = n(5, "MiniMaxH3TurboLoRA", model=model, lora_name="minimax_h3_turbo_v4_step600_ema.safetensors",
              strength=1.0, low_vram=False)
    model = n(6, "LoraLoaderModelOnly", model=model, lora_name="h3-realism-people-t2v-i2v-r2v.safetensors",
              strength_model=1.0)
    model = n(7, "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
    hits = None
    if a.blockout or not a.audio:
        hits = n(12, "H3BandDrumHitsMIDI", midi="groove.mid")
        if a.teach:
            hits = n(30, "H3BandDrumTeacher", hits=hits, choke_gap=1.0)
    audio = n(10, "LoadAudio", audio="drums.wav") if a.audio else n(31, "H3BandDrumSynth", hits=hits, seed=0)
    audio = n(11, "TrimAudioDuration", audio=audio, start_index=a.start, duration=a.frames / FPS)
    cond = {"clip": clip, "vae": vae, "prompt": a.prompt, "width": W, "height": H, "length": a.frames}
    if a.keyframe:
        cond["first_frame"] = n(14, "LoadImage", image="keyframe.png")
    g["15"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond}
    zone = {"latent": ["15", 1], "audio_vae": avae, "audio": audio, "video_denoise": 1.0, "audio_denoise": 0.0}
    if a.blockout:
        style = {"emotion": a.emotion, **{k: getattr(a, k) for k in SLIDERS}}
        frames = n(13, "H3BandDrumBlockout", hits=hits, start=a.start, frames=a.frames, fps=FPS, width=W, height=H,
                   view="front", head_schedule="0:groove", seed=0, camera="", engine="WORKBENCH", motion=a.motion,
                   **style)
        zone["video_latent"] = n(16, "VAEEncode", pixels=frames, vae=vae)
    lat = n(17, "H3BandZoneLatent", **zone)
    # BasicScheduler's own denoise floors steps/denoise (8 steps at 0.9 -> the full schedule); round it here
    total = total_steps(a)
    sig = n(22, "BasicScheduler", model=model, scheduler="simple", steps=total, denoise=1.0)
    if total > a.steps:
        g["29"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": sig, "step": total - a.steps}}
        sig = ["29", 1]
    g["24"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": n(20, "RandomNoise", noise_seed=a.seed), "guider": n(21, "BasicGuider", model=model, conditioning=["15", 0]),
        "sampler": n(23, "MiniMaxH3TurboSampler"), "sigmas": sig, "latent_image": lat}}
    video = n(25, "VAEDecode", samples=["24", 0], vae=vae)
    sound = n(26, "VAEDecodeAudio", samples=["24", 0], vae=avae)
    v = n(27, "CreateVideo", images=video, audio=sound, fps=float(FPS))
    n(28, "SaveVideo", video=v, filename_prefix="h3band/drum_h3", format="mp4", codec="auto")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--audio", help="drum audio under the take (default: H3 Band Drum Synth renders the MIDI)")
    ap.add_argument("--blockout", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--denoise", type=float, default=0.75)
    ap.add_argument("--keyframe")
    ap.add_argument("--start", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=124, help="17k+5 frames (124 = 5.2 s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--motion", choices=("smooth", "snap", "loose"), default="smooth",
                    help="blockout motion (blockout/drums.py MOTIONS): how exact the blockout is")
    ap.add_argument("--teach", action="store_true",
                    help="H3 Band Drum Teacher picks techniques (chokes, rimshots, ...) for the blockout and synth")
    ap.add_argument("--emotion", default="", help="blockout/drum_style.py timeline, e.g. 0:calm,3:intense")
    for k in SLIDERS:
        ap.add_argument(f"--{k}", type=float, default=-1.0, help=f"{k} slider 0..1 over every emotion preset")
    a = ap.parse_args()

    os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
    client = Comfy(api_key=os.environ.get("COMFY_API_KEY"))
    graph = build(a)
    wf = client.workflows.from_json(graph)
    if a.audio:
        wf.set_input("10", "audio", client.assets.from_file(a.audio))
    if "12" in graph:
        wf.set_input("12", "midi", client.assets.from_file(a.midi))
    if a.keyframe:
        wf.set_input("14", "image", client.assets.from_file(a.keyframe))
    job = client.submit(wf)
    print(f"[sdk] job {job.id}: blockout={a.blockout} denoise={a.steps / total_steps(a):.3f} "
          f"(last {a.steps} of {total_steps(a)} steps) keyframe={a.keyframe}", flush=True)
    shown = -1
    for ev in job.events():
        if isinstance(ev, Progress) and int(ev.value * 10) > shown:
            shown = int(ev.value * 10)
            print(f"[sdk] {ev.value:.0%} {ev.current_node or ''}", flush=True)
        elif isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled"):
            break
    job = job.result()
    out = job.get_outputs("28")[0]
    out.to_file(a.out)
    print(f"[sdk] {out.name} -> {a.out}")


if __name__ == "__main__":
    main()

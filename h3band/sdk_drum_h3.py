#!/usr/bin/env python3
"""MiniMax H3 drummer takes over the drum blockout, run on ComfyUI through the Comfy SDK.

  sdk_drum_h3.py OUT.mp4 --midi GROOVE.mid [--audio DRUMS.wav] [--blockout | --no-blockout] [--denoise D | --start-noise S]
                 [--zone-video ZONE.mp4 --zone-noise S]
                 [--keyframe FRAME.png] [--start S] [--frames 124] [--seed 42] [--steps 8] [--prompt TEXT]
                 [--motion smooth|snap|loose] [--teach] [--emotion 0:calm,3:intense]
                 [--force F] [--range R] [--body B] [--flair X]
                 [--drummer chad|carey|baker] [--mocap chad,carey] [--blend SPEC] [--classifier auto|jev|rules]
                 [--kit PRESET] [--camera EX,EY,EZ,LX,LY,LZ] [--blockout-video GREY.mp4]
                 [--control-video DEPTH.mp4] [--control-strength 1.0] [--control-patch NAME] [--control-fill zeros|grey]
                 [--no-turbo]

One graph: the drum stem is trimmed to [start, start + frames/24] and frozen into H3's audio stream by
H3 Band Zone Latent (audio_denoise 0). The MIDI's hits drive H3 Band Drum Blockout over the same window.
Without --audio, H3 Band Drum Synth renders the MIDI as the drum stem. --teach runs H3 Band Drum Teacher on
the hits first, so the blockout plays chokes, rimshots, cross-sticks, bells and flams and the synth sounds them.
--emotion and the sliders set the blockout's stroke heights, arm lift and body (blockout/drum_style.py).
--drummer plays the blockout as one mocap drummer on their kit; --mocap and --blend blend drummers
(blockout/drum_blend.py), with the notes labelled by --classifier (Jev needs TYPESAFE_API_KEY in the
server's environment). --kit (or the drummer's kit) goes to H3 Band Drum Kit, which the blockout and synth share.
--camera puts the blockout's camera at eye EX,EY,EZ looking at LX,LY,LZ (metres; the default is the front view).
--blockout-video starts from a grey video rendered earlier (1280x736, --frames long) in place of the node's render.
--control-video steers every step with H3's Fun ControlNet Union 2.0 (ComfyUI >= 0.38, the patch in models/model_patches):
  a depth, pose or edge video of the blockout (blender_drums.py --depth), 1280x736, --frames long, at --control-strength.
  The patch reads 49 channels: the control latent, a visibility mask and the latent of a masked source video. Given no
  mask, ComfyUI pads the last 24 with zeros (--control-fill zeros); --control-fill grey passes a mask of all ones, so
  they hold the latent of an empty (mean-grey) frame, the no-inpaint input ComfyUI's Z-Image Fun 2.0 node feeds.
--no-turbo drops the Turbo LoRA and samples --steps of plain Euler (the ControlNet's model card: 40 steps, guidance 1).
  --blockout     the grey blockout, VAE-encoded, is the starting video latent, sampled over the last --steps
                 of round(steps / denoise) steps. Sigma shift 12 keeps the schedule's top end near pure noise:
                 denoise 0.6 / 0.75 / 0.89 start at noise 0.95 / 0.97 / 0.99, which leaves next to nothing of the
                 blockout. --start-noise 0.6 (denoise 0.11, the last 8 of 72 steps) starts where it still shows.
  --no-blockout  the video starts from noise (denoise 1): what H3 does from the audio alone
--zone-video starts the bright parts of a drum_zone_mask.py video (sticks, hands) at --zone-noise and the rest at
  --start-noise, through H3's per-token noise labels: the sticks keep the blockout's timing, the rest gets the noise it
  needs to look real.
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
DRUMMER_KITS = {"chad": "standard", "carey": "big", "baker": "double_pedal"}     # blockout/drum_blend.py DRUMMERS


def total_steps(a):
    return max(a.steps, round(a.steps / a.denoise)) if a.blockout else a.steps


def start_sigma(a, shift=12.0):
    """the noise level the take starts at: the simple schedule is even in t, and shift maps t to sigma"""
    t = a.steps / total_steps(a)
    return shift * t / (1 + (shift - 1) * t)


def build(a):
    g = {}

    def n(nid, cls, **inputs):
        g[str(nid)] = {"class_type": cls, "inputs": inputs}
        return [str(nid), 0]

    model = n(1, "UNETLoader", unet_name="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", weight_dtype="default")
    clip = n(2, "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    vae = n(3, "VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    avae = n(4, "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    if a.turbo:
        model = n(5, "MiniMaxH3TurboLoRA", model=model, lora_name="minimax_h3_turbo_v4_step600_ema.safetensors",
                  strength=1.0, low_vram=False)
    model = n(6, "LoraLoaderModelOnly", model=model, lora_name="h3-realism-people-t2v-i2v-r2v.safetensors",
              strength_model=1.0)
    model = n(7, "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
    if a.control_video:
        ctl = n(43, "GetVideoComponents", video=n(42, "LoadVideo", file="control.mp4"))
        fill = {"mask": n(48, "SolidMask", value=1.0, width=W, height=H)} if a.control_fill == "grey" else {}
        model = n(45, "MiniMaxH3FunControlNetApply", model=model, model_patch=n(44, "ModelPatchLoader", name=a.control_patch),
                  vae=vae, strength=a.control_strength, start_percent=0.0, end_percent=1.0, control_video=ctl, **fill)
    hits = None
    if a.blockout or not a.audio:
        hits = n(12, "H3BandDrumHitsMIDI", midi="groove.mid")
        if a.teach:
            hits = n(30, "H3BandDrumTeacher", hits=hits, choke_gap=1.0)
    preset = a.kit or DRUMMER_KITS.get(a.drummer)
    kit = {"kit": n(32, "H3BandDrumKit", preset=preset)} if preset else {}
    audio = n(10, "LoadAudio", audio="drums.wav") if a.audio else n(31, "H3BandDrumSynth", hits=hits, seed=0, **kit)
    audio = n(11, "TrimAudioDuration", audio=audio, start_index=a.start, duration=a.frames / FPS)
    cond = {"clip": clip, "vae": vae, "prompt": a.prompt, "width": W, "height": H, "length": a.frames}
    if a.keyframe:
        cond["first_frame"] = n(14, "LoadImage", image="keyframe.png")
    g["15"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond}
    zone = {"latent": ["15", 1], "audio_vae": avae, "audio": audio, "video_denoise": 1.0, "audio_denoise": 0.0}
    if a.blockout and a.blockout_video:
        frames = n(41, "GetVideoComponents", video=n(40, "LoadVideo", file="blockout.mp4"))
        zone["video_latent"] = n(16, "VAEEncode", pixels=frames, vae=vae)
    elif a.blockout:
        style = {"emotion": a.emotion, **{k: getattr(a, k) for k in SLIDERS}}
        frames = n(13, "H3BandDrumBlockout", hits=hits, start=a.start, frames=a.frames, fps=FPS, width=W, height=H,
                   view="front", head_schedule="0:groove", seed=0, camera=a.camera, engine="WORKBENCH", motion=a.motion,
                   drummer=a.drummer or "none", mocap=a.mocap, blend=a.blend, **style, **kit,
                   **({"classifier": a.classifier} if a.classifier != "auto" else {}))
        zone["video_latent"] = n(16, "VAEEncode", pixels=frames, vae=vae)
    if a.blockout and a.zone_video:
        # the zone (sticks, hands) starts at --zone-noise while the rest of the frame starts at the schedule's top
        zone["zone_video"] = n(47, "GetVideoComponents", video=n(46, "LoadVideo", file="zone.mp4"))
        zone["zone_denoise"] = round(min(1.0, a.zone_noise / start_sigma(a)), 4)
    lat = n(17, "H3BandZoneLatent", **zone)
    # BasicScheduler's own denoise floors steps/denoise (8 steps at 0.9 -> the full schedule); round it here
    total = total_steps(a)
    sig = n(22, "BasicScheduler", model=model, scheduler="simple", steps=total, denoise=1.0)
    if total > a.steps:
        g["29"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": sig, "step": total - a.steps}}
        sig = ["29", 1]
    g["24"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": n(20, "RandomNoise", noise_seed=a.seed), "guider": n(21, "BasicGuider", model=model, conditioning=["15", 0]),
        "sampler": n(23, "MiniMaxH3TurboSampler") if a.turbo else n(23, "KSamplerSelect", sampler_name="euler"),
        "sigmas": sig, "latent_image": lat}}
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
    ap.add_argument("--denoise", type=float, default=0.75,
                    help="the share of the schedule run; with shift 12, 0.75 still starts at 97%% noise (--start-noise)")
    ap.add_argument("--start-noise", type=float,
                    help="start the blockout take at this noise level (0..1), e.g. 0.6; sets --denoise to match")
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
    ap.add_argument("--drummer", choices=tuple(DRUMMER_KITS), help="play as one mocap drummer, on their kit")
    ap.add_argument("--mocap", default="", help="mocap drummers to blend, e.g. chad,carey")
    ap.add_argument("--blend", default="", help="blend matrix or timeline, e.g. groove=chad,fill=carey")
    ap.add_argument("--classifier", choices=("auto", "jev", "rules"), default="auto",
                    help="what labels the notes for --blend (blockout/drum_blend.py CLASSIFIERS)")
    ap.add_argument("--kit", help="H3 Band Drum Kit preset (standard, double_pedal, big)")
    ap.add_argument("--blockout-video", help="grey video to start from in place of H3 Band Drum Blockout")
    ap.add_argument("--zone-video", help="h3band/drum_zone_mask.py video: its bright parts start at --zone-noise")
    ap.add_argument("--zone-noise", type=float, default=0.6)
    ap.add_argument("--camera", default="", help="blockout camera eye and look-at, e.g. -1.4,1.55,1.6,0,0.05,0.85")
    ap.add_argument("--control-video", help="depth/pose/edge video for H3's Fun ControlNet, applied at every step")
    ap.add_argument("--control-strength", type=float, default=1.0)
    ap.add_argument("--control-patch", default="minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors")
    ap.add_argument("--control-fill", choices=("zeros", "grey"), default="zeros",
                    help="the patch's 24 masked-video channels: ComfyUI's zero padding, or an empty frame's latent")
    ap.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    if a.start_noise:
        a.denoise = a.start_noise / (12.0 - 11.0 * a.start_noise)      # invert start_sigma

    os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
    client = Comfy(api_key=os.environ.get("COMFY_API_KEY"))
    graph = build(a)
    wf = client.workflows.from_json(graph)
    if a.audio:
        wf.set_input("10", "audio", client.assets.from_file(a.audio))
    if "12" in graph:
        wf.set_input("12", "midi", client.assets.from_file(a.midi))
    if a.blockout and a.blockout_video:
        wf.set_input("40", "file", client.assets.from_file(a.blockout_video))
    if a.control_video:
        wf.set_input("42", "file", client.assets.from_file(a.control_video))
    if "46" in graph:
        wf.set_input("46", "file", client.assets.from_file(a.zone_video))
    if a.keyframe:
        wf.set_input("14", "image", client.assets.from_file(a.keyframe))
    job = client.submit(wf)
    print(f"[sdk] job {job.id}: blockout={a.blockout} denoise={a.steps / total_steps(a):.3f} "
          f"(last {a.steps} of {total_steps(a)} steps, from noise {start_sigma(a) if a.blockout else 1:.3f}) "
          f"keyframe={a.keyframe} control={a.control_video}"
          + (f" zone={a.zone_video} from noise {a.zone_noise:.3f}" if "46" in graph else ""), flush=True)
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

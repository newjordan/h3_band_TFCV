#!/usr/bin/env python3
"""Run one MiniMax H3 audio-locked render on a ComfyUI with H3 support (H3B_COMFY_URL).

  h3_run.py JOB.json [--dump]

JOB = {tag, mode: A|A_ref|A_guide|B, image, audio, [video], prompt, seed, steps, width, height, length,
       [video_denoise | video_strength], [zone_box_start, zone_box_end, zone_denoise], [scheduler], [sampler], [model],
       [control: {video, strength, start, end, patch}], [no_first]}
Paths in image/audio/video are relative to the ComfyUI input dir.

A       = first frame (fl2va keyframe) + the track frozen INTO the target audio stream (mask 0), video generated.
A_ref   = ref2va (set "model" to a ref2va checkpoint): still as <Picture 1>, track as <Audio 1> reference AND frozen
          in the target audio stream.
A_guide = first frame + track pinned as a MiniMaxH3AddGuide audio guide at frame 0 (target audio generated).
B       = existing take VAE-encoded as the video stream at video_denoise, track frozen in the audio stream,
          first frame of the take as fl2va keyframe. With a zone box, only the box runs at zone_denoise:
          video_denoise 0 + zone_denoise 1.0 = an isolated run (everything outside the box locked to the take).
--dump prints the API graph instead of queueing it.
Writes RESULTS_DIR/<tag>.json; the take lands in COMFY_OUTPUT/<OUTPUT_PREFIX>/<tag>_*.mp4.
"""
import datetime, json, sys, time, urllib.request

import config


def api(path, data=None):
    req = urllib.request.Request(config.COMFY_URL + path,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


H3_VIDEO_SHIFT = 12.0       # MiniMax H3's sampling shift


def video_mask(j):
    """The video stream's mask value. "video_denoise" is the mask itself, a start sigma in H3's SHIFTED sigma space
    (shift 12), which is compressed near 1: 0.92 re-runs only the late half of the flow. "video_strength" s is the
    fraction of (unshifted) flow time to regenerate, converted with sigma = S*s / (1 + (S-1)*s) and rounded up to the
    1/256 token grid. Band Aid's lock: video_strength 0.88 (the blockout plate) + the Fun ControlNet at 1.0."""
    if "video_strength" in j:
        s = float(j["video_strength"])
        sig = H3_VIDEO_SHIFT * s / (1.0 + (H3_VIDEO_SHIFT - 1.0) * s)
        return min(1.0, -(-round(sig * 256.0, 6) // 1) / 256.0)
    return float(j.get("video_denoise", 1.0))


def build(j):
    g = {}

    def n(nid, cls, **inputs):
        g[str(nid)] = {"class_type": cls, "inputs": inputs}
        return [str(nid), 0]

    model = n(1, "UNETLoader", unet_name=j.get("model", config.H3_UNET), weight_dtype="default")
    clip = n(2, "CLIPLoader", clip_name=j.get("clip", config.H3_CLIP), type="minimax", device="default")
    vae = n(3, "VAELoader", vae_name=j.get("vae", config.H3_VAE))
    if j.get("control"):
        # Band Aid lock: H3 Fun ControlNet Union 2.0 on a control video (the Canny of the blockout's normal pass,
        # Blockout Reference Edges), injected into the DiT blocks at every step in [start, end]
        c = j["control"]
        patch = n(40, "ModelPatchLoader", name=c.get("patch", config.H3_FUN_CONTROLNET))
        cv = n(41, "LoadVideo", file=c["video"])
        g["42"] = {"class_type": "GetVideoComponents", "inputs": {"video": cv}}
        model = n(43, "MiniMaxH3FunControlNetApply", model=model, model_patch=patch, vae=vae,
                  strength=float(c.get("strength", 1.0)), start_percent=float(c.get("start", 0.0)),
                  end_percent=float(c.get("end", 1.0)), control_video=["42", 0])
    avae = n(4, "VAELoader", vae_name=j.get("audio_vae", config.H3_AUDIO_VAE))
    aud = n(6, "LoadAudio", audio=j["audio"])
    mode = j["mode"]
    if mode == "B":
        vid = n(20, "LoadVideo", file=j["video"])
        g["21"] = {"class_type": "GetVideoComponents", "inputs": {"video": vid}}
        frames = ["21", 0]
        first = n(22, "ImageFromBatch", image=frames, batch_index=0, length=1)
        if j.get("image"):                         # a styled first frame (e.g. the previous clip's last frame)
            first = n(5, "LoadImage", image=j["image"])
        venc = n(23, "VAEEncode", pixels=frames, vae=vae)
    else:
        first = n(5, "LoadImage", image=j["image"])
    if mode == "A_ref":
        g["7"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": clip, "vae": vae, "audio_vae": avae, "prompt": j["prompt"], "width": j["width"],
            "height": j["height"], "length": j["length"], "ref_image_size": j.get("ref_image_size", "match"),
            "ref_images.ref_image_0": first, "ref_audios.ref_audio_0": aud}}
    else:
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": clip, "vae": vae, "prompt": j["prompt"], "width": j["width"], "height": j["height"],
            "length": j["length"]}}
        if not j.get("no_first"):                  # no_first: the plate + control only, no grey frame 0 pinned
            g["7"]["inputs"]["first_frame"] = first
    pos, lat = ["7", 0], ["7", 1]
    if mode == "A_guide":
        pos = n(8, "MiniMaxH3AddGuide", positive=pos, audio_vae=avae, latent=lat, audio=aud, frame_idx=0)
    else:
        extra = {"video_latent": venc} if mode == "B" else {}
        for k in ("zone_box_start", "zone_box_end", "zone_denoise"):
            if k in j:
                extra[k] = j[k]
        lat = n(8, "H3BandZoneLatent", latent=lat, audio_vae=avae, audio=aud,
                video_denoise=video_mask(j), audio_denoise=0.0, **extra)
    smp = n(9, "KSamplerSelect", sampler_name=j.get("sampler", "res_multistep"))
    sig = n(10, "BasicScheduler", model=model, scheduler=j.get("scheduler", "simple"), steps=j["steps"], denoise=1.0)
    noise = n(11, "RandomNoise", noise_seed=j["seed"])
    guider = n(12, "BasicGuider", model=model, conditioning=pos)
    g["13"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": noise, "guider": guider, "sampler": smp, "sigmas": sig, "latent_image": lat}}
    out = ["13", 0]
    img = n(14, "VAEDecode", samples=out, vae=vae)
    au = n(15, "VAEDecodeAudio", samples=out, vae=avae)
    v = n(16, "CreateVideo", images=img, audio=au, fps=24.0)
    n(17, "SaveVideo", video=v, filename_prefix=f"{config.OUTPUT_PREFIX}/{j['tag']}", format="mp4", codec="auto")
    return g


def main():
    j = json.load(open(sys.argv[1]))
    g = build(j)
    if len(sys.argv) > 2 and sys.argv[2] == "--dump":
        print(json.dumps(g, indent=1))
        return
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = api("/prompt", {"prompt": g, "client_id": j["tag"]})
    if r.get("node_errors"):
        print("node errors:", json.dumps(r["node_errors"])[:3000])
        sys.exit(2)
    pid = r["prompt_id"]
    print(f"[h3_run] {j['tag']} queued {pid}", flush=True)
    status, fails = None, 0
    while True:
        time.sleep(10)
        try:
            h = api(f"/history/{pid}")
            fails = 0
        except Exception as e:
            fails += 1
            print("history poll failed:", e, flush=True)
            if fails >= 30:
                status = "server-unreachable"
                break
            continue
        if pid in h:
            st = h[pid].get("status", {})
            status = "success" if st.get("completed") and st.get("status_str") == "success" else "error"
            msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
            if msgs:
                print(json.dumps(msgs)[:3000], flush=True)
            break
    wall = time.time() - t0
    rec = {"t": datetime.datetime.now().isoformat(timespec="seconds"), "job": j, "prompt_id": pid,
           "status": status, "wall_s": round(wall, 1), "s_per_video_s": round(wall / (j["length"] / 24.0), 1)}
    json.dump(rec, open(config.RESULTS_DIR / f"{j['tag']}.json", "w"), indent=1)
    print(f"[h3_run] {j['tag']} {status} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()

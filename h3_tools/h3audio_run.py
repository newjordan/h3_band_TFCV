#!/usr/bin/env python3
"""Run one MiniMax-H3 audio-locked render against h3audio-comfy (127.0.0.1:18191).

  h3audio_run.py JOB.json      JOB = {tag, mode: A|A_guide|B, image, audio, [video], prompt, seed, steps,
                                      width, height, length, [video_denoise], [scheduler], [model]}
A       = first frame (fl2va keyframe) + vocal frozen INTO the target audio stream (mask 0), video generated.
A_ref   = ref2va (set "model" to the ref2va checkpoint): still as <Picture 1>, vocal as <Audio 1> reference AND frozen
          in the target audio stream.
C       = any mode + "control": {video, strength, start, end}: Fun ControlNet Union structure lock;
          "guides": [{image, frame_idx}]: extra keyframe anchors (MiniMaxH3AddGuide).
A_guide = first frame + vocal pinned as a MiniMaxH3AddGuide audio guide at frame 0 (target audio generated).
B       = existing take VAE-encoded as the video stream at video_denoise (shifted-sigma mask) or video_strength
          (fraction of unshifted flow time to regenerate; see video_mask), vocal frozen in the audio stream,
          first frame of the take as fl2va keyframe (or "image", if given: e.g. a styled frame over a blockout plate).
Writes ~/h3audio/results/<tag>.json; frames in output/ks_lipsync/<tag>/, mp4 in output/ks_lipsync/<tag>_*.mp4.
"""
import json, sys, time, urllib.request, subprocess, pathlib, datetime

API = "http://127.0.0.1:18191"
RES = pathlib.Path.home() / "h3audio" / "results"


def api(path, data=None):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


H3_VIDEO_SHIFT = 12.0  # supported_models.MiniMaxH3 sampling_settings["shift"]


def video_mask(j):
    """Mask value for the plate's video stream.

    The H3 node's mask m holds every video row at sigma = m * sigma_step, i.e. m IS the SDEdit start sigma in the
    model's SHIFTED sigma space (shift 12). That space is extremely compressed near 1: sigma 0.92 is only t = 0.49 of
    the unshifted flow time, so "video_denoise 0.92" re-runs just the late half and the plate's colour/material win.
    "video_strength" s is the fraction of flow time to regenerate (s=1: from pure noise); converted here with
    sigma = S*s / (1 + (S-1)*s) and rounded UP to the node's 1/256 token grid (the model labels rows on that grid).
    """
    if "video_strength" in j:
        s = float(j["video_strength"])
        sig = H3_VIDEO_SHIFT * s / (1.0 + (H3_VIDEO_SHIFT - 1.0) * s)
        return min(1.0, -(-round(sig * 256.0, 6) // 1) / 256.0)
    return float(j.get("video_denoise", 1.0))


def strength_to_mask(s):
    sig = H3_VIDEO_SHIFT * s / (1.0 + (H3_VIDEO_SHIFT - 1.0) * s)
    return min(1.0, -(-round(sig * 256.0, 6) // 1) / 256.0)


def lock_masks(j):
    """Lockstep profile, one mask per latent frame. j["lock"] = {"every": k, "anchor": s_a, "free": s_f, "width": w}:
    every k-th latent frame (and w-1 after it) is an anchor held to the plate at strength s_a; the rest at s_f.
    Latent frames: 5 per 17 pixel frames (73 frames = 22 latent frames)."""
    L = j["lock"]
    n, f = 0, 0                                     # latent frames: the VAE packs pixel frames 1,4,4,4,4 per 17
    while f < int(j["length"]):
        f += (1, 4, 4, 4, 4)[n % 5]; n += 1
    out = []
    for k in range(n):
        anchor = k % int(L["every"]) < int(L.get("width", 1))
        out.append(strength_to_mask(L["anchor"] if anchor else L["free"]))
    return out


def build(j):
    g = {}
    def n(nid, cls, **inputs):
        g[str(nid)] = {"class_type": cls, "inputs": inputs}
        return [str(nid), 0]
    model = n(1, "UNETLoader", unet_name=j.get("model", "minimax_h3_fl2va_pruned_nvfp4.safetensors"), weight_dtype="default")
    clip = n(2, "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    vae = n(3, "VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
    if j.get("control"):
        # structure lock: H3 Fun ControlNet Union 2.0 on a control video (canny / depth / gray ...) of the plate,
        # injected into the DiT blocks at every step in [start, end] (it does not fade like a plate blend)
        c = j["control"]
        patch = n(40, "ModelPatchLoader", name=c.get("patch", "minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors"))
        cv = n(41, "LoadVideo", file=c["video"])
        g["42"] = {"class_type": "GetVideoComponents", "inputs": {"video": cv}}
        model = n(43, "MiniMaxH3FunControlNetApply", model=model, model_patch=patch, vae=vae,
                  strength=float(c.get("strength", 1.0)), start_percent=float(c.get("start", 0.0)),
                  end_percent=float(c.get("end", 1.0)), control_video=["42", 0])
    if j.get("pull"):                                # KV pull toward the reference look (custom_nodes/ks_h3_attn.py)
        pl = j["pull"]
        model = n(46, "KSH3RefPull", model=model, blocks=str(pl.get("blocks", "all")), heads=str(pl.get("heads", "all")),
                  mult=int(pl.get("mult", 3)), segments=pl.get("segments", "ref_img"), steps=str(pl.get("steps", "all")))
    if j.get("probe"):                               # attention mass per head per segment, written after sampling
        pr = j["probe"]
        model = n(47, "KSH3AttnProbe", model=model, blocks=pr.get("blocks", "0,6,12,18,24,30,36,42,49"),
                  steps=pr.get("steps", "0,5,10,15"), out_path=pr["out"])
    if j.get("control2"):                            # a second Fun ControlNet patch chained on the first
        c2 = j["control2"]
        patch2 = n(50, "ModelPatchLoader", name=c2.get("patch", "minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors"))
        cv2_ = n(51, "LoadVideo", file=c2["video"])
        g["52"] = {"class_type": "GetVideoComponents", "inputs": {"video": cv2_}}
        model = n(53, "MiniMaxH3FunControlNetApply", model=model, model_patch=patch2, vae=vae,
                  strength=float(c2.get("strength", 1.0)), start_percent=float(c2.get("start", 0.0)),
                  end_percent=float(c2.get("end", 1.0)), control_video=["52", 0])
    avae = n(4, "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    aud = n(6, "LoadAudio", audio=j["audio"])
    mode = j["mode"]
    if mode == "B":
        vid = n(20, "LoadVideo", file=j["video"])
        g["21"] = {"class_type": "GetVideoComponents", "inputs": {"video": vid}}
        frames = ["21", 0]
        first = n(22, "ImageFromBatch", image=frames, batch_index=0, length=1)
        if j.get("image"):  # separate keyframe, e.g. a styled first frame over a grey blockout plate
            first = n(5, "LoadImage", image=j["image"])
        venc = n(23, "VAEEncode", pixels=frames, vae=vae)
    else:
        first = n(5, "LoadImage", image=j["image"])
        if j.get("video"):                         # any mode: the plate in the video stream at video_strength
            vid = n(20, "LoadVideo", file=j["video"])
            g["21"] = {"class_type": "GetVideoComponents", "inputs": {"video": vid}}
            venc = n(23, "VAEEncode", pixels=["21", 0], vae=vae)
    if mode == "A_ref":
        # ref2va: still = <Picture 1>, vocal = <Audio 1> reference, and the same vocal frozen in the target audio stream
        g["7"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": clip, "vae": vae, "audio_vae": avae,
                  "prompt": j["prompt"], "width": j["width"], "height": j["height"], "length": j["length"],
                  "ref_image_size": j.get("ref_image_size", "match"), "ref_images.ref_image_0": first,
                  "ref_audios.ref_audio_0": aud}}
        for i, im in enumerate(j.get("refs", []), start=1):  # more character references: <Picture 2>, ...
            g["7"]["inputs"][f"ref_images.ref_image_{i}"] = n(70 + i, "LoadImage", image=im)
    else:
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": clip, "vae": vae, "prompt": j["prompt"],
                  "width": j["width"], "height": j["height"], "length": j["length"]}}
        if not j.get("no_first"):                    # no_first: t2va + plate/control only (no grey frame 0 pinned)
            g["7"]["inputs"]["first_frame"] = first
    pos, lat = ["7", 0], ["7", 1]
    for i, gd in enumerate(j.get("guides", [])):     # lockstep anchors: clean keyframe cond rows at frame_idx
        im = n(60 + 2 * i, "LoadImage", image=gd["image"])
        pos = n(61 + 2 * i, "MiniMaxH3AddGuide", positive=pos, vae=vae, latent=lat, image=im, frame_idx=int(gd["frame_idx"]))
    if mode == "A_guide":
        pos = n(8, "MiniMaxH3AddGuide", positive=pos, audio_vae=avae, latent=lat, audio=aud, frame_idx=0)
    else:
        extra = {"video_latent": venc} if (mode == "B" or j.get("video")) else {}
        if "lock" in j:                                  # lockstep: per-latent-frame strengths (see frame_masks)
            extra["frame_masks"] = ",".join(f"{v:.6f}" for v in lock_masks(j))
        for k in ("face_box_start", "face_box_end", "face_denoise"):
            if k in j:
                extra[k] = j[k]
        lat = n(8, "KSH3LipSyncLatent", latent=lat, audio_vae=avae, audio=aud,
                video_denoise=video_mask(j), audio_denoise=0.0, **extra)
    smp = n(9, "KSamplerSelect", sampler_name=j.get("sampler", "res_multistep"))
    sig = n(10, "BasicScheduler", model=model, scheduler=j.get("scheduler", "simple"), steps=j["steps"], denoise=1.0)
    noise = n(11, "RandomNoise", noise_seed=j["seed"])
    guider = n(12, "BasicGuider", model=model, conditioning=pos)
    g["13"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": noise, "guider": guider, "sampler": smp,
               "sigmas": sig, "latent_image": lat}}
    out = ["13", 0]
    if j.get("probe"):
        out = n(48, "KSH3AttnProbeDump", latent=out)
    img = n(14, "VAEDecode", samples=out, vae=vae)
    au = n(15, "VAEDecodeAudio", samples=out, vae=avae)
    v = n(16, "CreateVideo", images=img, audio=au, fps=24.0)
    n(17, "SaveVideo", video=v, filename_prefix=f"ks_lipsync/{j['tag']}", format="mp4", codec="auto")
    n(18, "SaveImage", images=img, filename_prefix=f"ks_lipsync/{j['tag']}/frame")
    return g


def main():
    j = json.load(open(sys.argv[1]))
    RES.mkdir(parents=True, exist_ok=True)
    g = build(j)
    if len(sys.argv) > 2 and sys.argv[2] == "--dump":
        print(json.dumps(g, indent=1)); return
    t0 = time.time()
    r = api("/prompt", {"prompt": g, "client_id": j["tag"]})
    if r.get("node_errors"):
        print("node errors:", json.dumps(r["node_errors"])[:3000]); sys.exit(2)
    pid = r["prompt_id"]; print(f"[h3audio] {j['tag']} queued {pid}", flush=True)
    status = None
    while True:
        time.sleep(10)
        try:
            h = api(f"/history/{pid}")
        except Exception as e:
            print("history poll failed:", e, flush=True)
            if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "h3audio-comfy"],
                              capture_output=True, text=True).stdout.strip() != "true":
                status = "container-down"; break
            continue
        if pid in h:
            st = h[pid].get("status", {})
            status = "success" if st.get("completed") and st.get("status_str") == "success" else "error"
            msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
            if msgs: print(json.dumps(msgs)[:3000], flush=True)
            break
    wall = time.time() - t0
    rec = {"t": datetime.datetime.now().isoformat(timespec="seconds"), "job": j, "prompt_id": pid,
           "status": status, "wall_s": round(wall, 1),
           "s_per_video_s": round(wall / (j["length"] / 24.0), 1)}
    json.dump(rec, open(RES / f"{j['tag']}.json", "w"), indent=1)
    print(f"[h3audio] {j['tag']} {status} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()

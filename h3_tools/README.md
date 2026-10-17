# H3 tools (MiniMax-H3 in ComfyUI)

- `ks_h3_lipsync.py` -> ComfyUI `custom_nodes/`. `KSH3LipSyncLatent`: locks a stem into H3's audio stream, swaps a plate into
  the video stream at a per-token noise mask, optional per-latent-frame lockstep masks (`frame_masks`).
- `h3audio_run.py` -> one H3 render against a ComfyUI API (edit `API`). JSON job: modes A / A_ref / A_guide / B, plus
  `control` (Fun ControlNet Union 2.0: `{video, strength, start, end}`), `guides` (keyframe anchors), `lock`, `refs`, `no_first`,
  `video_strength` (fraction of the unshifted flow path; mask = 12s/(1+11s)).
- Models: `minimax_h3_fl2va_pruned_nvfp4`, `minimax_h3_ref2va_pruned_nvfp4`, and
  `model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors` (Comfy-Org/MiniMax-H3 on Hugging Face).
- `make_controls.py` (canny/gray control videos from plate frames), `fidelity.py` (plate structure correlation per frame:
  catches cuts/drift), `pick_anchors.py` (lockstep anchors from a pass-1 take).
- `jitter.py`, `zig.py`, `stall.py`, `miss.py`: hand-motion QA on piano anim JSONs (wrist shake, finger reversals, solver stalls,
  missed keys).

Why plates drift in plain mode B: a masked plate token is re-blended each step as m*x + (1-m)*plate; at strength 0.88, m = 0.992,
so the plate steers only the first noisy steps. The ControlNet path (`blockout/h3_lock.py`) holds structure at every step.
Control strength 1.0: 1.2 turns gauntlets into bare hands (it follows the grey hand too closely).

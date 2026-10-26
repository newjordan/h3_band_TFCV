# Band Aid Comfy workflow handoff

| File | Format | Qualification |
|---|---|---|
| `fl2va_control_api.json` | Comfy API prompt dictionary | Reuses the existing plate + Fun ControlNet + `H3BandZoneLatent` graph; new portable prompt/output prefix. Offline structure checked, not rerun. |
| `ref2va_video_reference_api.json` | Comfy API prompt dictionary | Experimental native video-reference route. Actual local source socket: `ref_videos.ref_video_0`. No Fun ControlNet, no fast adapter. No inference qualification. |
| `*_canvas.json` | Comfy canvas v0.4 draft | Editable layout aid only; dynamic sockets/widget ordering may need UI re-save. Not UI-verified. API files are authoritative. |

## Input mapping (under ComfyUI/input)
- `band_aid/skeleton_c0_plate_nohands.mp4`: 124 dense plate frames, 24 fps, 1280×704; FL2VA source.
- `band_aid/skeleton_c0_posekeys.mp4`: matching control frames, same clock; FL2VA structure input.
- `band_aid/skeleton_c0_plate.mp4`: same short shot with **stock/standard hands**, 124 dense frames at 24 fps; Ref2VA motion reference. Do not feed skeletal Blender look geometry.
- `band_aid/skeleton_c0_sound.wav`: conditioning audio aligned to the dense clock. Event alignment still needs validation.

These are filenames to map, not assets shipped by the kit. A 31-frame final delivery clip is NOT a substitute for the 124-frame dense reference.

## Model mapping
| Graph node | Example filename | Location |
|---|---|---|
| UNETLoader, FL2VA | `minimax_h3_fl2va_pruned_nvfp4.safetensors` | `models/diffusion_models/` |
| UNETLoader, Ref2VA | `minimax_h3_ref2va_pruned_nvfp4.safetensors` | `models/diffusion_models/` |
| CLIPLoader | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| VAELoader, video | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| VAELoader, audio | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| ModelPatchLoader, FL2VA only | `minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors` | `models/model_patches/` |

Use compatible files already installed by the operator; quantization and required kernels are runtime-specific. Nothing downloads automatically.

## Review / import without running
Open the canvas draft in ComfyUI if a UI is already available, inspect widget values and named socket links, and re-save in that target version. Do not start a server solely for this task. API graphs are raw prompt dictionaries for Comfy's API/workflow tooling; they are **not** a canvas format. Do not blindly POST them: inspect `/object_info` on an already-approved server first and validate every required node/input/model choice.

The Ref2VA graph generates video from motion-reference conditioning and uses the supplied AUDIO directly for `CreateVideo` output. It does **not** freeze that audio into the model latent and does not audio-condition the reference node. This isolates the video-reference experiment and avoids promising an unqualified audio-lock path. The source WAV is an output soundtrack, not proof of visual/acoustic synchronization.

## Execution gates, only later
1. Explicit operator approval, headroom check, queue/lease preservation and new unique output prefix.
2. Target-server schema validation for all nodes, especially autogrow `ref_videos.ref_video_0`.
3. Decode/probe assets only when permitted; verify frames, dimensions and audio duration/clock.
4. Do not reuse static calibration after changing checkpoint/reference/shape.
5. Run only the short shot first. Review all contacts and skeletal appearance together, then restore original delivery clock/audio for final comparison.

Do not treat the ~6.6-second multi-B300 benchmark in the announcement as a speed estimate for this machine.

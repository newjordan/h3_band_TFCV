# Control of MiniMax-H3 for plate-faithful hands + a consistent look: literature review

Date: 2026-10-10. Method: web search plus page fetches. Anything tagged [UNVERIFIED] is inference or from a single secondary source. Where I only got aggregator pages (ComfyUI Wiki, RunComfy) rather than the model card or code, that is noted. I did not read any paper body in full; paper claims come from search summaries.

## 0. Diagnosis in one paragraph

Whole-frame structure follows the plate and the hands do not. Both control paths you use are coarse in exactly the hand region:
- The Fun control branch injects at only 10 of 50 blocks (see 1.1) and was trained on generic video. Canny of a grey plate gives the hand a few edges that the model can "explain" with a different, more plausible human hand.
- The plate-in-video-stream path (strength 0.88) is a partial-noise init. Fine finger geometry is high-frequency, so it is the first thing the sampler rewrites once the noise level exceeds the finger scale.
- Raising control to 1.2 pushes the model toward its prior for "hands in a piano scene", which is bare skin. That is the model's prior fighting a style it only gets from text and ref tokens.
- More steps or a refiner changing nothing is consistent with the hand layout being fixed in the first high-noise steps and later steps only polishing. [UNVERIFIED inference; test with experiment 1.]

So there are two separate problems: (a) geometry, which needs a conditioning signal that is spatially precise at finger scale and strong early; (b) identity (gauntlet), which needs appearance tokens that win the argument against the human-hand prior inside the hand region. They interact: a stronger geometry signal from a grey plate carries "skin-like" cues unless the plate itself is already gauntlet-shaped.

## 1. MiniMax-H3 and Fun ControlNet Union

### 1.1 Facts (secondary sources unless noted)
Source: ComfyUI Wiki articles on v1 and v2.0, which summarize the Hugging Face model cards. I did not reach the model cards or the VideoX-Fun repo directly.
- v1 (2026-08-24, ~6.8 GB): Canny, Depth, HED, MLSD, Pose, plus inpainting via a 49-channel control input (latent + masked latent + mask). Control blocks at layers [0,10,20,30,40]. https://comfyui-wiki.com/en/news/2026-08-24-minimax-h3-fun-controlnet-union
- v2.0 (2026-09-22, ~13.5 GB): adds Scribble, Layout (colored boxes on white, VACE style) and Gray (luminance video). Control blocks doubled to 10 (layers 0,5,10,...,45), credited with tighter structural adherence. Inpaint masked pixels now use "post_norm" (holes at mid-grey) and need config `minimax_h3_control_inpaint_post_norm.yaml`; the v1 config loads silently half-built. https://comfyui-wiki.com/en/news/2026-09-22-minimax-h3-fun-controlnet-union-2
- Model card examples all use steps 40, guidance_scale 1.0 (guidance-distilled; >1 degrades), `control_context_scale` 1.0. 1.0 is described as strongest; 0 disables. Wiki pages state no recommended intermediate value, no start/end percent, nothing on combining controls.
- ComfyUI: PR #16471 (merged 2026-09-22): ModelPatchLoader + `MiniMaxH3FunControlNetApply`. Kijai converted patches: `Kijai/MiniMax-H3-experimental` (bf16 and int8). Model card: huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0. Code: https://github.com/aigc-apps/VideoX-Fun (examples/minimax_h3_fun/predict_v2v_control.py).
- Not found: any official hand or fine-detail tip; any guidance on stacking two control types in one pass; any start/end-percent semantics for the ComfyUI apply node. [UNVERIFIED whether the node exposes start/end percent; check the node signature locally.]

### 1.2 H3 native ComfyUI features relevant to us
Source: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native.md
- `MiniMaxH3AddGuide`: guides anchored at any frame, VAE-encoded and appended to positive conditioning (not used as start latent). Chainable. `denoise` does not scale guide strength.
- Latent noise masks (PR #15375): per-token mask, 0 preserve, 1 regenerate, snapped to the 2x2 latent patch grid. This is the mechanism your 0.88 plate injection uses; a spatially varying mask is available, so hand region can be treated differently from background.
- Ref2va: separate weights from fl2va; up to 9 reference images, 3 reference videos, 3 audio; references tagged `<Picture N>` in connection order, and the page says explicitly assigning a role (identity, style, motion, camera, voice) in the prompt works much better. `ref_image_size` = `match` (downscaled) vs `max` (up to 2048 short edge, better identity, slower).
- RunComfy notes `<Picture 1>` tagging and a warning that sage_attention can blur output on large machines: https://www.runcomfy.com/zh-CN/comfyui-workflows/minimax-h3-comfyui-4-step-reference-to-video-audio [third-party, UNVERIFIED].
- Comfy docs say nothing on per-token timesteps.

### 1.3 Actionable here (rank by impact in section 6)
- Switch to v2.0 if not already; confirm the correct post_norm config / Kijai v2.0 patch (not v1).
- Try Depth or Gray/Scribble in addition to or instead of Canny (no published guidance; experiment).
- Use control_context_scale between 0.6 and 1.0 with a gauntlet-bearing plate rather than >1.0.
- Vary the noise-mask strength spatially: lower strength in hand mask (more plate fidelity per the init) vs background. Counter-intuitive but testable (see exp 3).
- Role-tagged ref prompt, `ref_image_size=max`, multiple crops of the gauntlet as separate refs.

## 2. Attention-based appearance and identity transfer

### 2.1 Prior art
- MagicAnimate (appearance encoder; its keys/values from the reference are concatenated with the video self-attention keys/values before scores are computed): CVPR 2024; code in https://gitee.com/imcheese/magic-animate and supplement https://openaccess.thecvf.com/content/CVPR2024/supplemental/Xu_MagicAnimate_Temporally_Consistent_CVPR_2024_supplemental.pdf
- Animate Anyone ReferenceNet (spatial-attention appearance, Pose Guider for motion, temporal attention). Reference implementation `mutual_self_attention.py`: https://huggingface.co/spaces/xunsong/Moore-AnimateAnyone/raw/75c09e23439d39f074adefe7e51b20ac5f505238/src/models/mutual_self_attention.py . Design lesson: appearance goes through the same self-attention as the denoised tokens (KV extension), while pose/structure comes through a separate additive path. This is exactly your split: Fun control = structure path; ref tokens/KV pull = appearance path.
- AnyV2V: keeps source video queries/keys (and conv features) injected only in early sampling steps, with separate thresholds for conv and attention. https://arxiv.org/pdf/2403.14468. Lesson: injection windows limited to early steps are standard.
- Stable Flow (FLUX): self-attention injection of reference K/V restricted to "vital layers" found by bypass ablation; vital layers are not simply early or late. https://arxiv.org/html/2411.14430v1 . Lesson: do a per-block bypass sweep for H3 rather than guessing.
- OminiControl: condition tokens concatenated into the packed sequence and processed jointly by the DiT blocks (with LoRA); https://arxiv.org/pdf/2411.15098 . That is structurally what H3 ref2va already does, so your KV-pull is an amplification of an existing mechanism, not a new one.
- UNO, I2VEdit, TokenFlow, FreeControl, VideoBooth, Champ, StableAnimator: not read. [UNVERIFIED, no results retrieved.]

### 2.2 Layers / heads
- "Analysis of Attention in Video Diffusion Transformers" (Wen et al.): attention is strongly local in space/time; individual layers play distinct roles (e.g. camera); sparsifying attention breaks quality because a few layers are intolerant. https://arxiv.org/html/2504.10317v1
- "Controlling Motion Transfer in Diffusion Transformers via Attention Heads" (Jung et al., arXiv 2607.11081, reportedly ECCV 2026): finds motion-specialized temporal heads and structure-specialized spatial heads; uses attention-entropy as a structure cue and selective feature injection. https://arxiv.org/abs/2607.11081 [which backbones were used: UNVERIFIED]
- Sparse-vDiT: diagonal, multi-diagonal, vertical-stripe head patterns recur: https://ojs.aaai.org/index.php/AAAI/article/view/37287
- No source found that separates appearance vs structure layers in Wan/HunyuanVideo/H3 specifically. Treat the standard heuristic (early blocks and early timesteps set layout; mid and late blocks and later timesteps set texture/identity) as a hypothesis [UNVERIFIED for H3], and measure it by the bypass sweep.

### 2.3 What to implement in H3 (given the custom attention hook)
1. KV pull with a logit bias: add +b to the logits of (query in hand mask) x (key in reference gauntlet tokens). Restrict to the hand-region queries so the rest of the frame keeps its look. This is the soft analogue of MagicAnimate's KV concatenation.
2. Block selection by sweep: apply the bias in one block-band at a time (e.g. 10-block windows) and measure gauntlet identity vs edge correlation.
3. Timestep gating: bias only in the first ~30-50% of sampling (AnyV2V convention), or only in the later portion; they achieve different things.
4. Head selection: rank heads by existing attention mass from hand-region queries to reference keys; bias only the top-k heads (head-selective steering). [UNVERIFIED that steering a subset works better than all heads; test.]
5. Optional second reference stream: re-encode the first-pass render (hand crop) as reference to enforce self-consistency over time (the ReferenceNet-style "same identity every frame" idea).
Risk: strong bias degrades geometry because ref tokens carry their own spatial layout (the reference image's own hand pose); mitigate by using a reference with the target pose (plate-aligned stylized keyframe, see 5).

## 3. Render-to-video with exact motion; hand-specific control

- Diffusion as Shader (DaS, arXiv 2501.03847): 3D tracking videos as the control, built on CogVideoX with image + tracking video; claims tracking videos tie frames together and enable mesh-to-video, motion transfer, camera control. https://arxiv.org/abs/2501.03847 . Lesson: per-surface-point tracking (colors encode a persistent 3D point id) gives correspondence that Canny/depth cannot.
- DAR "Video Models as Native 4D Renderers" (arXiv 2608.00094, per search summary): depth alone is a weak condition; rasterize animated mesh to tracking, world-position and normal maps on Wan2.2 via widened control adapter; swapping world-position for depth cost 1.26-1.55 dB PSNR. https://arxiv.org/abs/2608.00094 [numbers from a search summary, UNVERIFIED]. Note these require training/fine-tuning a control adapter; you only have Fun Union, which does not accept tracking maps natively. You can still feed a coordinate/ID-coloured pass through Gray or the RGB-accepting channels, but that is out of distribution [UNVERIFIED].
- Generative Rendering (CVPR 2024): uses mesh-derived correspondences to inject features across frames in a 2D diffusion model. https://arxiv.org/pdf/2312.01409 . Lesson: correspondence-based feature sharing across time works with a pretrained model; relevant to enforcing the same gauntlet texture on the same finger across frames.
- HandRefiner (2311.17957): hand mesh to depth ControlNet inpainting; official advice is control weight 0.4-0.8 since 1.0 loses texture; a "phase transition" in ControlNet strength. https://ar5iv.labs.arxiv.org/html/2311.17957 . Directly relevant: your 1.2 over-control is the opposite of the recommended regime, and a masked inpaint pass over only the hand with a hand-specific depth is the HandRefiner recipe. 2025 follow-up argues full mesh beats depth for palm/back details: https://arxiv.org/pdf/2506.12680
- HANDI (hand-centric video, hand-refinement loss) https://arxiv.org/html/2412.04189v5 and Generated Reality (hand-tracking-conditioned video; hybrid 2D-3D conditioning, best under occlusion) https://hyper.ai/en/papers/2602.18422 . Both train; not drop-in.
- Optical flow / tracks as conditioning: Go-with-the-Flow, MotionPrompting etc. returned nothing in search. [UNVERIFIED]; H3 has no known flow-conditioned adapter.
- Piano-specific hand video generation: nothing found.

Implementable in H3/ComfyUI now: the Fun inpaint mode applied to the hand region (control + inpaint share one branch; uses mask), depth/Canny/HED/Scribble passes of a hand-crop, tight crops at higher resolution (below), and the noise mask.

## 4. Why a refinement pass and more steps change nothing; and what that suggests
Not literature-backed, reasoning only: if a refine pass re-noises the video to a level below the finger scale it will keep the existing layout; if it is a pure re-sample from the same control it reproduces the same prior. Alter what is given, not how long it runs: higher effective resolution on the hands (crop and render the hand region as its own clip at full latent resolution, then composite), different control signal, or reference with matching pose. In 1360x768 latents with 2x2 patching a finger is on the order of a few tokens wide [UNVERIFIED arithmetic: VAE stride and patch size for H3 not checked], so a crop pass is likely the largest single geometric gain.

## 5. Plate preparation ("pixel setup")

Little published evidence on fine fingers; the following combines HandRefiner's depth finding with general ControlNet practice.
- Make the plate carry the answer: model the gauntlet geometry (plates, knuckle guards, segmented fingers) on the mannequin hands in Blender so Canny/depth edges ARE gauntlet edges. A bare mannequin finger plate teaches the model "bare finger". This probably dominates every other plate-side change, and also removes the reason the model reaches for human skin at strength 1.2. [UNVERIFIED but strongly motivated.]
- Near-target materials and lighting: render a grey-metal (chrome-ish matcap or a simple reflective material), dark piano, low-key lighting. Gray/luminance control (v2.0) and the VAE-encoded plate in the video stream both carry tonal information; a plate close to the target tonal range means the sampler needs to change less, so strength can be lowered without losing fidelity.
- Control passes ranking for fine fingers (hypothesis from HandRefiner + coverage): (1) depth or normal of hands (isolated, per-finger separation via distinct depth discontinuities), (2) line art / Scribble with finger-segment edges and occlusion lines, (3) Canny (dependent on shading edges; generates spurious edges on mannequin joints), (4) segmentation/ID pass as Layout (boxes only, too coarse), (5) Pose is too sparse unless using a hand skeleton. Normal maps are not an official Union type, so only via the Gray/RGB channel [UNVERIFIED].
- Separate hand pass: render hands alone on black with a high-contrast depth so the control sees only the hands; composite with the full-scene control or run as a masked hand-only pass.
- Render at 2x the target resolution for edges, downsample; anti-aliased thin edges vanish at latent scale; thicken line passes to >= 2 latent tokens, i.e. about 16 px at 8x VAE and 2x2 patch (assumed stride, UNVERIFIED).
- Stylized keyframe for reference: produce one stylized frame per shot in plate pose (via any method, even manual), use it as `<Picture 1>` identity/style reference and as a guide anchor (`MiniMaxH3AddGuide` at multiple frame_idx). Pose-matched references reduce the pose-conflict risk from section 2.3.

## 6. Ranked options

(a) Hand-region fidelity to the plate (highest first):
1. Gauntlet-correct plate geometry in Blender (removes the prior conflict). High.
2. Hand-crop second pass at high resolution with its own control (depth/Scribble) plus noise mask compositing. High.
3. Switch/add control type for hands (depth or Scribble in addition to Canny), control scale 0.7-1.0. Medium-high.
4. Multi-frame guide anchors from pose-matched stylized keyframes. Medium-high [UNVERIFIED how strongly guides constrain finger layout].
5. Lower plate-injection strength inside hand mask / higher outside (spatially varying noise mask). Medium.
6. KV-pull logit bias with pose-matched reference limited to early steps (also helps geometry only if reference is pose-matched). Medium.
7. Unconditioned changes (steps, refine). Low (observed).

(b) Consistent strong look:
1. Pose-matched stylized keyframes as refs with role-tagged prompt and `ref_image_size=max`. High.
2. KV pull on hand-region queries toward gauntlet reference tokens, mid/late blocks, head-selective. High.
3. Multiple gauntlet crops as separate references (palm side, back, knuckle). Medium.
4. Lower control scale (<=1.0), prompt with explicit "chrome plated gauntlet fingers, no bare skin" and a negative via CFG if CFG is available (guidance distilled; negative prompts may not work, UNVERIFIED). Medium.
5. Plate with near-target materials. Medium.

## 7. Experiments (8-12)

Keep seed, shot, resolution fixed. Remember: same-seed A/B is only valid in-process (cross-process MAD ~37), so run variants in one process.

Metrics: M1 = edge correlation inside the hand mask (your existing metric, Canny of render vs plate); M2 = finger-keypoint error using a hand detector (e.g. MediaPipe/RTMPose on render vs plate, pixels) [tool choice up to you]; M3 = gauntlet-ness: mean CLIP/DINO feature similarity between hand-crop and gauntlet reference crop, plus a skin-pixel fraction in the hand mask (HSV heuristic); M4 = temporal drift: LPIPS or DINO-feature variance of the hand crop across consecutive frames after flow-warp; M5 = whole-frame structure correlation (guard: must not drop).

1. Timestep trace. Dump the x0-prediction hand crop at each step for the current best setting; measure M1 per step. Finding the step at which finger layout freezes tells which injection window matters (settles the "refine changes nothing" puzzle).
2. Control-type bake-off in the hand region: Canny vs Depth vs Scribble vs Gray vs Canny+Depth blended into one video (or chained patches if the node permits), control scale in {0.6, 0.8, 1.0}. Measure M1, M2, M3.
3. Spatial noise-mask schedule: plate strength in hand mask in {0.6, 0.75, 0.88, 0.95} with background fixed at 0.88. Measure M2, M3, M5. Expect fidelity to rise and look to fall as hand-mask strength goes up.
4. Gauntlet-geometry plate: add real gauntlet geometry (even rough) to the mannequin hands, rerun at control 0.8-1.0 and strength 0.88. Measure M1-M3. Predicted largest single move.
5. Hand-crop second pass: crop a hand-region window, render at native H3 resolution with its own depth/Scribble control, composite via feathered mask. Compare with full-frame. M1, M2, M4, and seam visibility.
6. Pose-matched stylized keyframe: one hand-stylized keyframe (any method) as `<Picture 1>` with an explicit role tag, plus as a guide anchor at 1, 3, 5 positions. Measure M2, M3, M4. Compare against a pose-mismatched reference to quantify the pose-conflict risk.
7. KV-pull bias sweep: bias b in {0, 1, 2, 4} on hand-query to ref-key logits; all blocks vs. bands of 10 blocks (0-9, ... 40-49); full run vs. first 40% of steps. Measure M3 (up) vs M1/M2 (must not drop). Log attention mass to ref tokens per block.
8. Per-block bypass sweep (Stable Flow style): bypass or zero the reference-tokens' attention contribution in one block at a time (and for the Fun control, scale individual control skips to 0) and record change in M1 and M3 to find the structural blocks and the appearance blocks. Use result to pick bands for experiment 7.
9. Head-selective steering: from exp 7's best band, rank heads by hand-query attention to ref tokens; apply bias to top 8 / top 25% / all heads. Measure M3, M1, and cross-frame M4.
10. HandRefiner regime check: plate scale 0.4-0.8 with hand-only mask via the Fun inpaint mode (49-channel) on a first-pass render; measure M1/M3 versus the whole-frame result, and whether the human-hand reversion at 1.2 disappears at lower scale combined with the gauntlet plate.
11. Reference quantity and size: 1 vs 3 gauntlet crops, `match` vs `max` ref size, with/without role tags. Measure M3 and wall time.
12. Plate material test: grey matte vs matcap chrome vs full chrome-with-lighting plate at fixed settings, controlling for edges (render both depth and Canny from the same geometry). Measure M1, M3 and tonal match to target; tells whether pixel-setup realism lets you lower strengths.

## 8. Unverified / open items
- Whether `MiniMaxH3FunControlNetApply` exposes start/end percent (not in any page I read).
- Exact VAE stride, patch size and token grid in H3; the finger-scale arithmetic in section 4/5 assumes 8x VAE and 2x2 patching.
- Which blocks/heads carry appearance vs structure in H3: no literature found; experiments 1, 8, 9 are the way to find out.
- Whether chained control patches (two control types at once) are supported or just additive in the node.
- Negative prompting effect given guidance distillation.
- DAR, Jung et al. (ECCV 2026), and numbers quoted from them come from search snippets only.
- I2VEdit, TokenFlow, FreeControl, VideoBooth, Champ, StableAnimator, StableFlow's FLUX layer list, Go-with-the-Flow, MotionPrompting: no content retrieved.

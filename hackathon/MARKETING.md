# Band Aid — copy-ready hackathon kit

## One-line pitch
**Band Aid puts timing into the pixels before MiniMax H3 paints the performance.**

## Short description (submission)
Band Aid is a timing-first toolkit for AI music videos in ComfyUI. A Blender blockout provides explicit performer motion and instrument actions; MiniMax H3 adds a cinematic look using plate and structural conditioning. Custom nodes support drum events, blockout controls, audio-conditioned AV latents and reference-adherence diagnostics. The project makes control signals and failures visible instead of relying on prompts alone. Current demos show the workflow and its trade-offs—not a guarantee of note-perfect execution. Exact piano contact, synchronized skeletal appearance and the new reference-video route remain active work.

## Problem / approach / contribution
- **Problem:** a plausible-looking performance can miss the beat, target the wrong key, or slide through an instrument.
- **Approach:** make an explicit motion source, condition generation on it, then compare output to that source.
- **Contribution:** reusable Comfy nodes, editable workflow graphs, a blockout-to-H3 pipeline and diagnostics that expose drift.
- **Why it matters:** creators can iterate on performance timing and visual appearance separately and see where control fails.

## 90-second demo script
**0–15 s — problem.** “Music-video generation can look convincing while the hands or hits miss the performance. Band Aid starts with timing, not another longer prompt.” Show a labeled blockout.

**15–35 s — pipeline.** Show `FL2VA control` graph: motion plate + control + audio → H3 → comparison. Point to the custom node sockets. Say which runtime dependencies are required.

**35–55 s — evidence.** Show a provenance-checked historical plate/control/H3 triptych. “Here is what was supplied, what H3 generated, and how we inspect drift.” Do not imply the clip was generated live.

**55–75 s — honest result.** “A recent short test preserved motion better but did not produce the desired skeletal appearance. Another had skeletal features but poor motion. We report both; neither is accepted as exact performance.”

**75–90 s — handoff.** Show the new reference-video graph as experimental. “The source, workflows and offline checks are available for the team. Next we test joint appearance/contact fidelity, then speed.”

Fallback with no GPU: run the offline verifier, show the graphs, and play existing labeled clips. Never fake a queued render or a live generation.

## Social launch draft
We’re building **Band Aid**: timing-first control for MiniMax H3 music videos in ComfyUI. Blender blockout → motion/structure conditioning → H3 → reference-adherence checks. Editable graphs and custom nodes are in the repo. Exact contact and skeleton-look fidelity are still research—not solved by a speed demo.

Code: https://github.com/newjordan/h3_band_TFCV
Project page: https://newjordan.github.io/h3_band_TFCV/

## Team update / judge Q&A
**Does it guarantee correct notes?** No. The source performance and generated contacts both need validation. Edge metrics alone do not prove it.

**Did you train H3?** No. Credit MiniMax for H3, the H3/Fun ControlNet contributors and ComfyUI for the runtime. Our work is the performance-control pipeline, integration and diagnostics.

**Is Ref2VA the fix?** Not established. We have a new video-reference graph to test without assuming ControlNet compatibility.

**Can this run now with no model RAM?** The offline kit can be inspected and checked now. Actual inference requires loading models and explicit resource approval.

## Submission checklist
- [ ] Confirm challenge name, deadline, eligibility and required fields against the organizer's current rules.
- [ ] Use this branch's `STATUS.md`, not optimistic historical README language.
- [ ] Pick only rights-cleared, provenance-checked clips; label historical vs new, blockout vs H3.
- [ ] Add team names, demo URL, submitted commit SHA and repository visibility.
- [ ] Record the 90-second demo from existing material; no inference needed.
- [ ] Confirm credits/licenses for models, Blender rigs, music and samples. `NOTICE_FurElise.md` covers only its named source, not all music rights.
- [ ] Publish submission manually; this push does not submit to the competition or social accounts.

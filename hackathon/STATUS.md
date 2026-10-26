# Evidence-based status

## Implemented / demonstrated, with boundaries
| Area | Evidence | Honest boundary |
|---|---|---|
| Comfy custom-node package | Root entry point and seven registered nodes in `comfy_nodes/` | Dependency/runtime availability must be checked in the target Comfy install. |
| FL2VA plate + Fun ControlNet workflow | Existing API workflow; fresh portable handoff copy | Prior renders completed, but completion does not mean exact contact or desired appearance. |
| Audio-conditioned latent and regional control | `H3BandZoneLatent` implementation | Behavior depends on upstream per-token AV mask support; no new runtime qualification in this handoff. |
| Pose controls / edge lock diagnostics | Blockout pose/reference-edge/lock-check nodes | Edge matching is a proxy, not correct notes, finger ownership or acoustic sync. |
| Piano recovery | Two completed 124-frame runs; frozen result summary | Both failed the combined appearance/performance objective. |
| Historical visual demos | Tracked `docs/media` look grids and clips | Label as historical examples; see provenance below. |
| Source-only collaboration kit | Offline verifier, marketing, API graphs and package tooling | A passing offline check is not an inference test. |

## Actual last piano results (recovery handoff, 23 Oct)
- Baseline: ~1180 s wall, motion correlation **0.02**, median reference-edge distance **5.48 px**. Malformed fan-like skeletal hands; ambiguous contact. **Failed.**
- Geometry variant: ~1170 s wall, motion correlation **0.91**, median edge distance **0.96 px**. Solid gray human hands, not skeleton appearance. **Failed appearance; not certified performance.**
- Multiple settings changed together (plate, control, strength); do not credit strength alone.
- Original action has known missed/extra blows. Key-depth reconciliation did not fix finger trajectories. A model cannot be credited with solving this source defect.

## Needs work / not qualified
1. **Joint look + motion fidelity:** skeletal appearance while retaining standard-hand source trajectories and exact key/finger contact.
2. **Ref2VA:** new video-reference API graph follows inspected installed source sockets; no fresh inference or target-server validation. Prior local recipe says Ref2VA ignores Fun ControlNet. The experimental graph therefore does not attach it.
3. **Timing:** pitch-preserving conditioning stretch event alignment still unverified. The 124 dense frames are ~5.17 s at 24 fps; delivery returns to 31 original-clock frames (~1.29 s). Never silently substitute clocks.
4. **Acceleration:** four-step adapter conversion had CPU algebra checks, not serving-kernel/quality qualification. Post benchmarks do not transfer to our Spark workload.
5. **Long shots:** seams, sustained identity, exact performance and full suite remain unaccepted. Do not release the existing suite waiter.
6. **Canvas layouts:** included for editing convenience; not opened/re-saved in the target UI. API graphs are authoritative. Check widget values and dynamic input sockets before execution.
7. **Distribution:** source archive is supported. Registry publication, wheel installation and a fresh environment are not tested by this handoff. No credentials or model weights included.

## Claims policy and demo provenance
Use: “timing-first blockout and measurable reference adherence for H3 music videos.”
Do not use: “perfect sync,” “note-perfect pianist,” “solved skeleton hands,” “Ref2VA fixes it,” or unqualified hardware speedups.

Some older look experiments supplied look-specific geometry; that is not proof of the current standard-hand-input requirement. Before showing a clip as compliant, verify its input provenance. Label plate/control/H3 outputs separately and state limitations on-screen.

## What this push does not do
No model loads, GPU tests, dependencies installed, services started/stopped, media regenerated, or full-suite jobs submitted. Local unfinished experiment code is not automatically included in the clean handoff commit.

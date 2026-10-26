# Install and package handoff

## No-model review now
```sh
git clone --branch team/hackathon-handoff https://github.com/newjordan/h3_band_TFCV.git
cd h3_band_TFCV
python3 scripts/verify_hackathon.py
python3 scripts/package_hackathon.py --output ../band-aid-hackathon.tar.gz
```
The archive tool uses the **committed** source tree, excludes media/research/tests and includes code, license, requirements, handoff docs and example workflows. It streams through git/tar with bounded memory and writes a SHA-256 file. It does not include dirty local changes, weights, reference audio or private source actions. Commit your intended changes before packaging.

## ComfyUI install later (explicit approval required)
Clone the repository into `ComfyUI/custom_nodes/band-aid` using the handoff branch. Install `requirements.txt` with that Comfy environment's Python, then restart ComfyUI only when permitted. Those operations can allocate memory and are **not** part of this no-load handoff.

The root `__init__.py` exports node mappings from `comfy_nodes/`. Keep the whole repo together: the nodes use sibling `blockout/` and `h3band/` modules. This is a source/plugin release, not a promise of a tested wheel or Registry publication. The included `pyproject.toml` carries Comfy metadata; do not claim the Registry entry has been published.

Expected Band Aid node IDs:
- `H3BandZoneLatent`
- `BandAidBlockoutPoseControl`
- `BandAidBlockoutReferenceEdges`
- `BandAidBlockoutLockCheck`
- `H3BandDrumEvents`
- `H3BandDrumHitsMIDI`
- `H3BandDrumBlockout`

Individual modules are optional-loaded, so a server starting is not proof all seven nodes loaded. Inspect the node list/logs after an approved startup.

## External dependencies and assets
- ComfyUI with MiniMax H3, video/audio and advanced sampler nodes.
- H3 video diffusion model, Qwen text encoder and video/audio VAEs; model filenames in graphs are examples, not bundled files or auto-download directives.
- Fun ControlNet Union model patch only for the FL2VA control graph.
- Blender and a suitable rig for new blockout rendering (not needed to inspect graphs).
- Rights-cleared plate/control/audio assets mapped under `input/band_aid/`.
- Upstream per-token AV mask behavior matters for frozen audio and source adherence. Check target runtime; do not assume all revisions are equivalent.

## Publish discipline
Share the branch/commit and archive checksum. Never add model weights, local credentials, generated source-action dumps or unreviewed caches. Historical large media is omitted from the source archive; the repository's tracked docs/media remains available separately. Model, rig and media licenses are distinct from this repo's `LICENSE`.

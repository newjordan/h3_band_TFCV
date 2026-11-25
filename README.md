# Band Aid

Timing for MiniMax H3 music videos. H3 can paint a pianist or a drummer, but a stem by itself does not put a key press or a drum hit on the beat. Band Aid animates a simple 3D performer on the real notes, renders that motion, and has H3 paint the final look over it. The timing is in the pixels before H3 starts.

ComfyUI custom nodes, plus a Comfy API runner and Comfy SDK drivers. The project page is [`docs/index.html`](docs/index.html).

Code is MIT. See [License](#license).

## What it does

- **Piano.** `blockout/piano.py` reads MIDI, chooses a fingering, and solves rigged hands so each finger reaches its key on the note. `blockout/feeling.py` turns a per-bar chart (strength, touch, freedom, rubato, head style) into the performance the motion and the audio both use.
- **Drums.** `blockout/drums.py` sticks a kit from a General MIDI drum file or from a hit list. `h3band/drum_events.py` is a band-energy hit detector for a drum stem. `blockout/drumsynth.py` renders a hit list to a WAV for timing checks.
- **Working piano.** `blockout/action.py` plays a simulated grand action from the animated hands: weighted keys, a hammer with let-off, dampers and pedal. A key sounds only when a finger presses it.
- **Render.** `blockout/blender_piano.py` and `blockout/blender_drums.py` render the motion in Blender (grey plate, normal pass, hand mask). Blender is not required for the solvers or the tests.
- **Lock.** *Blockout Reference Edges* builds a Canny control from the normal pass. *Blockout Lock Check* reports the share of those edges, inside the performer mask, that have a render edge within a tolerance (default 3 px), and draws the reference edges in red over the render.
- **H3.** `h3band/h3_run.py` turns a job file into an API-format graph and can post it to ComfyUI. *H3 Band Zone Latent* freezes a stem in H3's audio stream and can reopen a box of the frame. The drum blockout is also a Comfy node, driven by `h3band/sdk_drum_blockout.py` and `h3band/sdk_drum_h3.py`.
- **Scores.** Lip sync (`sync_score.py`), strum sync (`strum_score2.py`; `strum_score.py` is the earlier whole-box flow), stick sync (`drum_sync_score.py`), and stem dynamic range (`stem_quality.py`). `shot_preflight.py` checks a shot spec and writes the zone jobs. `run_chain.py` queues them in order.

H3 weights, separator checkpoints, face and hand landmark models, and input audio or video are not in this repo. Tests do not download them.

## Setup

Python 3.10 or newer. The base install is numpy, scipy, soundfile and opencv-python-headless. From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Optional extras, same flags as the requirements files:

| extra | requirements file | what it adds |
|---|---|---|
| (base) | `requirements.txt` | solvers, lock check, drum events, H3 graph builder |
| `blockout` | `requirements-blockout.txt` | matplotlib, for the feeling-chart plot |
| `score` | `requirements-score.txt` | mediapipe, for face and hand landmarks |
| `sep` | `requirements-sep.txt` | audio-separator, onnxruntime, torch, torchaudio, librosa |
| `sdk` | `requirements-sdk.txt` | `comfy-sdk` and `comfy-api-proxy` |
| `dev` | — | pytest |

`requirements-sep.txt` expects a CPU torch installed first:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[sep]"
```

ComfyUI nodes: the package is the whole repo, because the nodes import `blockout/` and `h3band/`.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/newjordan/h3_band_TFCV band-aid
# ComfyUI's own Python
pip install -r band-aid/requirements.txt
export H3B_BLENDER=/path/to/blender   # default: blender on PATH
```

Linking only `comfy_nodes/` also works (`ln -s "$PWD/comfy_nodes" ComfyUI/custom_nodes/band_aid`). Use one layout. `pyproject.toml` carries the Comfy Registry metadata (publisher `newjordan`, display name Band Aid).

H3 renders need a ComfyUI with MiniMax H3 and the files named in `h3band/config.py` (`H3B_UNET`, `H3B_CLIP`, `H3B_VAE`, `H3B_AUDIO_VAE`, `H3B_FUN_CONTROLNET`). The tightest audio lock uses per-token noise masks on nested AV latents. Without that, the zone latent still loads and the sampler treats the masks as inpaint blending.

Paths are environment variables. Defaults are relative to the repo (`h3band/config.py`):

| variable | default | meaning |
|---|---|---|
| `H3B_WORK` | `./work` | working directory |
| `H3B_SONG` | `work/song/mix.wav` | full mix |
| `H3B_BEATS` | `work/song/beats.json` | `{"beats": [seconds, ...]}` |
| `H3B_STEMS_DIR` | `work/stems` | `vocals_dry.wav`, `guitar.wav`, `drums.wav`, `backing.wav`, `bass.wav`, `mix.wav` |
| `H3B_GUITAR_EVENTS` | `work/guitar_events.json` | output of `guitar_events.py` |
| `H3B_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI API |
| `H3B_COMFY_INPUT`, `H3B_COMFY_OUTPUT` | `./ComfyUI/input`, `./ComfyUI/output` | ComfyUI directories |
| `H3B_MODEL_DIR` | `./models` | separator checkpoints |
| `H3B_FACE_MODEL` | `models/face_landmarker.task` | MediaPipe face landmarker |
| `H3B_HAND_MODEL` | `models/hand_landmarker.task` | MediaPipe hand landmarker |
| `H3B_BLENDER` | `blender` | Blender executable |
| `H3B_REPO` | this repo | used when the node folder is linked by itself |
| `H3B_LAB_PYTHON` | the current interpreter | lab runner |
| `H3B_LAB_LOCAL_SLOTS` | `1` | local worker threads for `lab/run.py` |
| `H3B_LAB_HOSTS` | empty | extra machines, `name:slots:python:code`, comma-separated |
| `H3B_LAB_SUITE` | `examples/piano` | MIDI and feeling files for the lab |
| `H3B_LAB_SITE` | unset | site directory for `lab/publish.py` |

## Usage

These commands need no weights and no GPU.

```bash
python -m blockout.midi examples/drums/rock_groove.mid --beats
python -m blockout.drums examples/drums/rock_groove.mid work/drums.json --dur 4
python -m blockout.drumsynth work/drum_hits.json work/drums.wav
python -m blockout.synth examples/piano/clair_de_lune.mid work/preview.wav --dur 8
python -m blockout.piano examples/piano/clair_de_lune.mid work/cdl.json \
    --feeling examples/piano/clair_de_lune.feeling.json --dur 4 --notes-out work/cdl_notes.json
python h3band/h3_run.py examples/workflows/band_aid_skeleton_job.json --dump
```

`--speed` on the piano solver is applied once, after a feeling chart, and each note keeps its touch. `--schedule` times are seconds on that same clock. `--notes-out` is `[[start, end, pitch, velocity, track], ...]` for `blockout/sampler.py`.

`python -m blockout.drums examples/drums/rock_groove.mid /tmp/drums.json --dur 1` was run here at 24 fps. The printed check was contact error 0.04 mm median and max, 0 frames with a tip through a head, and the one kick's beater on the head (5 hand hits in that second).

`python -m blockout.midi examples/drums/rock_groove.mid --beats` printed 224 notes, 0.000–32.214 s, 61 beats and 16 downbeats.

`h3_run.py --dump` prints the API graph to stdout and a one-line summary to stderr. For `examples/workflows/band_aid_skeleton_job.json` that graph has 24 nodes. `video_strength` 0.88 is written onto H3 Band Zone Latent as `video_denoise` 0.9921875 (the strength, converted into H3's shifted sigma and rounded up onto the 1/256 grid). `--dump` does not contact the server. Queueing posts the graph to `$H3B_COMFY_URL/prompt` and writes `work/results/<tag>.json`, including when the server cannot be reached.

Shipped example MIDI, counted by `blockout/midi.py`:

| file | notes |
|---|---|
| `examples/piano/clair_de_lune.mid` | 1468 |
| `examples/piano/KV331_3_RondoAllaTurca.mid` | 1614 |
| `examples/piano/brahms_intermezzo_op118_no2.mid` | 1677 |
| `examples/piano/rach_prelude_op3_no2.mid` | 1725 |
| `examples/piano/chopin_nocturne_cs_minor_posth.mid` | 940 |
| `examples/piano/knights_suite.mid` | 2046 |
| `examples/drums/rock_groove.mid` | 224 |

### ComfyUI

| node | class | role |
|---|---|---|
| Blockout Reference Edges | `BandAidBlockoutReferenceEdges` | Canny of a normal pass. Optional mask. |
| Blockout Lock Check | `BandAidBlockoutLockCheck` | recall, precision, report, red overlay |
| H3 Band Zone Latent | `H3BandZoneLatent` | stem frozen in the audio stream; optional zone box |
| H3 Band Drum Events | `H3BandDrumEvents` | drum stem to `DRUM_HITS` |
| H3 Band Drum Hits (MIDI) | `H3BandDrumHitsMIDI` | GM drum `.mid` in the input folder |
| H3 Band Drum Blockout | `H3BandDrumBlockout` | `DRUM_HITS` to grey kit frames, via Blender |

Each module loads on its own, so a missing dependency only drops that module's nodes.

```bash
export H3B_COMFY_URL=http://127.0.0.1:8188
python h3band/h3_run.py examples/workflows/band_aid_skeleton_job.json --dump
python h3band/h3_run.py examples/workflows/band_aid_skeleton_job.json
```

| file | what |
|---|---|
| `examples/workflows/band_aid_skeleton_job.json` | mode B, `video_strength` 0.88, control strength 1.0, `no_first`, 124 frames, 1280×704, seed 7 |
| `examples/workflows/band_aid_h3_lock_api.json` | the lock graph exported from that kind of job |
| `examples/workflows/drum_blockout_api.json` | drum stem to hits to blockout to MP4 |

Drum blockout through the Comfy SDK (a self-hosted ComfyUI speaks API v2 via `comfy-api-proxy`):

```bash
pip install -e ".[sdk]"
comfy-api-proxy run --comfyui http://127.0.0.1:8188 --port 8189 &
cd h3band
python sdk_drum_blockout.py ../examples/drums/rock_groove.mid ../work/groove.mp4 --audio ../work/drums.wav
python sdk_drum_h3.py ../work/C.mp4 --midi ../examples/drums/rock_groove.mid --audio ../work/drums.wav \
    --blockout --denoise 0.8 --start 13
```

`COMFY_BASE_URL` selects the proxy (default `http://127.0.0.1:8189`). `COMFY_API_KEY` is read from the environment when the SDK asks for one. It is not stored in the repo.

### Stems, shots, scores

```bash
cd h3band
python separate.py
python stem_quality.py q.json dry=$H3B_STEMS_DIR/vocals_dry.wav gtr=$H3B_STEMS_DIR/guitar.wav --ref dry
python guitar_events.py $H3B_STEMS_DIR/guitar.wav $H3B_BEATS $H3B_GUITAR_EVENTS --drums $H3B_STEMS_DIR/drums.wav
python shot_preflight.py ../examples/shot_example.json
python run_chain.py demo_duo --dry-run
```

Guitar zones in the chain are scored with `strum_score2.py`. Pass a tight `--box` around the picking hand. Without a beats file, metric `b` is left unset and `sync_margin` is margin `a`. A clip under 13 frames cannot take the 6-frame permutation shift, so `p_a` is unset and the margin is still returned. An audio file shorter than one 24 fps frame, or shorter than the band-pass filter's padding, scores as silence instead of raising.

Face landmarks for `sync_score.py`:

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The piano blockout is not a Comfy node. Render it with Blender after the solver:

```bash
blender -b --factory-startup -P blockout/blender_piano.py -- work/cdl.json work/frames --view three4
```

Sampled playback (`blockout/sampler.py`) uses Salamander Grand Piano V3 (CC-BY 3.0, Alexander Holm), which is not included: <https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html>. `blockout/synth.py` is the dependency-free preview.

## Tests

GitHub Actions (`.github/workflows/ci.yml`) installs `.[dev]` on Python 3.11 and 3.12 and runs pytest. No GPU, no model download.

On this machine, Python 3.12.3, `python3 -m pytest` reported **34 passed in 1.06s**. The suite covers MIDI running status and tempo, the feeling/speed timeline, drum-groove roundtrip, a one-second drum contact check, the additive synths, lock-check overlays (including a frame with no edges), short-clip scorers, the H3 graph dump, and the shot frame grid.

## Layout

| path | role |
|---|---|
| `blockout/` | piano, drums, action, synths, Blender renderers |
| `h3band/` | stems, scores, preflight, chain, H3 and SDK runners |
| `comfy_nodes/` | ComfyUI nodes. Lock math that does not need torch is in `lock_core.py` |
| `h3_tools/` | experiment drivers for H3 looks and lock checks |
| `examples/` | MIDI, feeling charts, shot spec, API graphs |
| `lab/` | piano-hand experiment runner. Hosts come from the environment, not from a personal path |
| `docs/` | project page |
| `tests/` | pytest |

Earlier H3 render writeups live on the project page and in `lab/`. This README does not restate those figures.

## License

Code: MIT, in `LICENSE`.

Example MIDI files have their own terms. See `examples/piano/README.md` and `examples/drums/README.md`. `rock_groove.mid` is original and MIT, like the code. MPFB2 hands: generated meshes CC0, MPFB2 code GPL-3.0. Salamander Grand Piano V3 is CC-BY 3.0 and is not shipped.

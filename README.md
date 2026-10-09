# h3_band_TFCV

Multi-performer music-video shots on **MiniMax H3**: the singer's mouth on the vocal *and* the guitarist's hands on the guitar, in the same shot.

## The problem

H3 generates video and audio jointly, with **one** audio stream and one attention pass over the whole frame. Nothing tells it which sound belongs to which part of the frame. Drive a two-person shot with the full mix and it guesses: everyone sings, or one person performs and the other freezes.

## The approach

1. **Pre-isolate.** Split the song into stems (dry lead vocal, guitar, drums, backing vocals) and define the **zones of action** in the frame (a box per performer).
2. **Isolated runs.** One H3 run per zone. Only that zone regenerates, driven by **its own stem** frozen into H3's audio stream. Everything outside the zone is locked to the previous take.
3. **Merge.** A low-denoise pass over the whole frame under the full mix, to settle the seams.

```
song.wav ──separate──▶ vocals_dry / guitar / drums / backing
                              │
shot spec (zones) ──preflight──▶ cut stems + jobs ──run_chain──▶ base → zone A → zone B → merge
                                                                    (each take scored per zone)
```

## How the lock works (per-token noise masks)

`comfy_nodes/h3_band_zone_latent.py` (**H3 Band Zone Latent**) writes per-stream noise masks onto the AV latent:

- **mask 0**: the token is fed in clean at the condition timestep and pinned to that content to the end. With `audio_denoise=0`, the stem is in H3's audio stream, clean, from the very first step.
- **mask 1**: generated from scratch.
- **fractional mask m**: the token runs at m·σ, which is partial re-noise.

An isolated run is `mode B` + `video_denoise 0` (the take is locked) + a zone box at `zone_denoise`.

**Zone sizing, what we've seen so far:**
- A lip sync works with a head box at about **0.92**, because a mouth moves a few pixels.
- Instrument motion did **not** appear at 0.92. Because of H3's sigma shift, 0.92 leaves the region about half-noised, so the pose survives and only the texture changes.
- At **1.0** with a box over only the lower body and the instrument body, the performer was regenerated but still didn't strum. The locked guitar neck and raised arm above the box pinned the pose.
- **Open:** whether 1.0 with a box over the **whole performer and the whole instrument** (plus a prompt that describes playing, not posing) gets strumming in time. That's hackathon task #1.

## Setup

**Not included:** ComfyUI, the MiniMax H3 weights, separator checkpoints, the face landmark model, and any media.

1. **ComfyUI with MiniMax H3 support and per-token noise masks on nested AV latents** (upstream ComfyUI PR #15375). Copy `comfy_nodes/h3_band_zone_latent.py` into `ComfyUI/custom_nodes/`. H3 file names default to the values in `h3band/config.py` (`H3B_UNET`, `H3B_CLIP`, `H3B_VAE`, `H3B_AUDIO_VAE`).
2. **Two venvs.** They're kept separate because mediapipe and torch pins fight.
   ```bash
   python3 -m venv .venv_score && .venv_score/bin/pip install -r requirements-score.txt
   python3 -m venv .venv_sep   && .venv_sep/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
                               && .venv_sep/bin/pip install -r requirements-sep.txt
   ```
3. **Face model** for `sync_score.py`:
   ```bash
   mkdir -p models && curl -L -o models/face_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```
4. **Paths.** Everything is set by env vars (see `h3band/config.py`):

   | var | default | what |
   |---|---|---|
   | `H3B_WORK` | `./work` | working dir |
   | `H3B_SONG` | `work/song/mix.wav` | full mix |
   | `H3B_BEATS` | `work/song/beats.json` | `{"beats": [s, ...]}` |
   | `H3B_STEMS_DIR` | `work/stems` | `vocals_dry.wav guitar.wav drums.wav backing.wav bass.wav mix.wav` |
   | `H3B_GUITAR_EVENTS` | `work/guitar_events.json` | output of `guitar_events.py` |
   | `H3B_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI API |
   | `H3B_COMFY_INPUT` / `H3B_COMFY_OUTPUT` | `./ComfyUI/input` / `output` | ComfyUI dirs |
   | `H3B_MODEL_DIR` | `./models` | audio-separator checkpoints |
   | `H3B_FACE_MODEL` | `models/face_landmarker.task` | mediapipe model |

## Workflow

```bash
cd h3band
# 1. stems: separate from the mix, or drop generator stem exports (e.g. Suno) into $H3B_STEMS_DIR under the generic names
../.venv_sep/bin/python separate.py                        # vocal -> dry vocal, 6-stem split
../.venv_sep/bin/python stem_quality.py q.json dry=$H3B_STEMS_DIR/vocals_dry.wav gtr=$H3B_STEMS_DIR/guitar.wav --ref dry
# 2. pattern identifiers
../.venv_sep/bin/python guitar_events.py $H3B_STEMS_DIR/guitar.wav $H3B_BEATS $H3B_GUITAR_EVENTS --drums $H3B_STEMS_DIR/drums.wav
../.venv_sep/bin/python align_lyrics.py $H3B_STEMS_DIR/vocals_dry.wav lyrics.json align.json      # optional
../.venv_sep/bin/python vocal_activity.py $H3B_STEMS_DIR/vocals_dry.wav va.json --align align.json  # optional
# 3. pre-flight a shot (cuts stems, checks zones, writes jobs)
../.venv_score/bin/python shot_preflight.py ../examples/shot_example.json
# 4. run the chain (needs ComfyUI up)
../.venv_score/bin/python run_chain.py demo_duo --dry-run
../.venv_score/bin/python run_chain.py demo_duo
```

**Choosing a lead vocal.** Use the stem with real silence between phrases: high per-5-s dynamic range and a high silent fraction in `stem_quality.py`. A reverb-washed or doubled vocal never goes quiet, so the mouth never closes. A de-reverb pass on a separated vocal often beats a generator's own vocal stem, which can keep doubles and gang vocals.

**Starting from an existing take.** Put `"plate": "<input-relative mp4>"` in the spec. The base run is skipped and the first zone run locks to that take. Mind any preroll: `song_start` is where the *render's* frame 0 sits in the song.

## Scorers (each with its own control)

- **`sync_score.py VIDEO AUDIO --box x0,y0,x1,y1`**: mediapipe FaceLandmarker `jawOpen` (plus the inner-lip gap) against the vocal envelope (150–4000 Hz RMS dB). It reports:
  - peak cross-correlation and its lag over ±8 frames
  - correlation at lag 0
  - closed-mouth-in-silence and open-mouth-in-voice rates
  - the fraction of frames with no face found.

  Use `--shift S` for a time-shifted audio control. `jawOpen` is the signal that separates real audio from the control; the lip gap doesn't. Small or mic-occluded faces use a tiled search plus a tracked upscaled crop.
- **`strum_score.py VIDEO EVENTS SONG_START --box ...`**: Farneback optical-flow magnitude in the box (detrended) against the guitar onset train. It reports the peak cross-correlation over ±8 frames, plus controls with the onsets shifted by 1, 2 and 3 s, and `margin_vs_control`. Positive-control check: injecting a 4 px jolt on each onset frame of a real take scores far above the shifted controls.

## Known limits

- **Video tokens are about 4 frames long.** H3's latent frames cover 1, 4, 4, 4, 4 pixel frames per 17-frame clip, and a token is 32 px. 16th-note strums are finer than one video token, so strum timing has to come through the 40 Hz audio tokens.
- **The lyric aligner is weak on sung vocals.** torchaudio MMS_FA is speech-trained: expect about ±0.5 s on half the words, and character spans rather than phonemes. Vocal activity leans on dry-vocal energy.
- **Stroke direction** (down vs up, from low-vs-high band rise timing) is a weak signal; treat individual calls as unreliable.
- **One shared audio stream per run.** The zone runs work around this by freezing one stem per run; they don't solve it.

## Hackathon tasks

- [ ] **Per-zone audio binding.** Bind each stem to its zone inside one run: multiple audio segments plus an attention bias, so stem A's tokens reach only zone A's video tokens.
- [ ] **Staggered per-zone timesteps.** Give zones their own timestep rows (AdaLN row ids per token). Global structure resolves first, then each zone develops in turn while the other stays noisy as context.
- [ ] **Drummer and bassist shots.** Sync to drum hits (kick, snare, crash onsets from the drum stem) and to bass note onsets.
- [ ] **A better strum scorer.** Track the picking hand (keypoints) instead of box flow, score stroke direction, and add a ground-truth clip.
- [ ] **Phoneme-level vocal alignment** for sung vocals, giving visemes for the mouth.
- [ ] **Seam handling in the merge pass.** Release a feathered band around touching zones instead of the whole frame.

## License

MIT, see `LICENSE`.

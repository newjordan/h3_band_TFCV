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
- At **1.0** with a box over the **whole performer and the whole instrument**, plus a prompt that describes playing instead of posing, the guitarist **plays**: the picking arm moves with motion blur. The strokes are **not on the beat**.
- Adding drums under the guitar in the audio stream, or a "hard strum on every beat" prompt, didn't change that.
- **The prompt sets the amount of motion.** "Hard, on every beat" gives big swings; "small, tight strokes, fretting hand slides, gentle sway" gives a calmer performance. It doesn't set the timing.
- **Retiming an H3 take onto the onsets** (time-warping the zone with DTW so its stroke peaks land on the onsets) failed: the generated strokes weren't distinct enough to align.
- **Conclusion:** H3 doesn't place a strum or a key press on a specific onset from the audio alone. Its video tokens span 4 frames, finer than a 16th-note strum, and nothing ties a background performer's hand to the stem. Timing has to be in the pixels before H3 sees them, which is what `blockout/` does (below).

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
- **`strum_score.py VIDEO EVENTS SONG_START --box ...`** (v1): Farneback optical-flow magnitude in the box (detrended) against the guitar onset train, with the onsets shifted by 1, 2 and 3 s as controls. Body bounce and camera moves swamp it; use v2.
- **`strum_score2.py VIDEO EVENTS SONG_START --box ... [--beats BEATS] [--grid-lag S]`** (v2): the picking-stroke signal. It uses MediaPipe hand landmarks when a hand is found in at least half the frames. Otherwise it takes optical flow with the zone's median motion removed (bounce, push-in) and picks the 3x3-cell sub-box with the most motion above 3 Hz. It scores stroke speed against the onset train, with controls (onsets shifted 1/2/3 s, another song section) and a permutation p-value over 200 circular shifts. `sync_margin` = real minus the best control.
  - Validation: a synthetic onset-locked jolt under a ±25 px random-walk body bounce scores +0.59 (p 0.005), where v1 reads 0.00. A static plate scores −0.09.
  - **Pass a tight `--box` around the picking hand and check the reported `subbox`.** Given a whole-performer zone, the sub-box search once locked onto drifting stage smoke beside the guitarist and reported a false "in sync" (+0.10, p 0.005).
  - At 24 fps a 16th note is about 3 frames, so beat-phase locking (metric `b`) can't separate synced from static; it's reported but not used.

## Known limits

- **Video tokens are about 4 frames long.** H3's latent frames cover 1, 4, 4, 4, 4 pixel frames per 17-frame clip, and a token is 32 px. 16th-note strums are finer than one video token, so strum timing has to come through the 40 Hz audio tokens.
- **The lyric aligner is weak on sung vocals.** torchaudio MMS_FA is speech-trained: expect about ±0.5 s on half the words, and character spans rather than phonemes. Vocal activity leans on dry-vocal energy.
- **Stroke direction** (down vs up, from low-vs-high band rise timing) is a weak signal; treat individual calls as unreliable.
- **One shared audio stream per run.** The zone runs work around this by freezing one stem per run; they don't solve it.

## Instrument blockouts (`blockout/`)

Zone runs gave lip sync, but not instruments: H3 doesn't learn a strum or a key press from the audio alone. H3 *is* good at painting over grey 3D blockouts. So we animate primitive performers playing the actual notes, render grey footage, and let H3 re-skin it with the stem locked in its audio stream. The timing is in the pixels.

| file | what |
|---|---|
| `midi.py` | MIDI reader (notes in seconds, beat/downbeat grid from the time signature). No dependencies. |
| `feeling.py` | Feeling chart → performance: per-bar strength, touch (caress..strike), freedom, tempo (rubato) and head style, plus phrase breathing, voicing and seeded per-note variation. |
| `piano.py` | Fingering (Viterbi, Parncutt-style costs), hand glide, per-touch strike/press/release, arm weight, phrase "breaths", finger IK, key travel. Collision: the keyboard is a heightfield (white keys, raised black keys, pressed keys lowered, rail, fallboard); finger pads sit on it by their radius, every finger segment is lifted out of it, and pressing fingers pull the hand within reach. Each run reports `qa`: finger-frames inside a key, strikes in the hit pad, crossed fingers, and `reach` (pressing fingertips that don't get to their key). Plain numpy. |
| `dynamics.py` | Second-order springs (frequency, damping, response) that every motion target runs through: anticipation, overshoot, follow-through. |
| `suite.py` | Splices an original theme and excerpts into one MIDI on bar boundaries, with a per-bar tempo map (the Knight's Suite). |
| `cameras.py` | Music-driven camera edit: cuts on downbeats (on beats at the climax), shot length and shot choice follow the energy (wide, first-person, orbit, hand tracking, close-ups, overhead, low hero angle, grazing keyboard, closing pull-back), operator-smoothed tracking. Render with `--view edit`. |
| `performer.py` | Shared body: torso, neck, head, 2-bone arm IK, and head styles driven by the beat grid (`focused`, `groove`, `wild`, `crowd`, `expressive`). |
| `blender_piano.py` | Blender renderer: grey 88-key piano, capsule performer, `pov` (camera rides the eyes), `three4`, `side`, `edit` views. `--hands mpfb` (default): a real rigged, skinned CC0 hand from MPFB2 on the primitive wrists, each bone posed from the joints (wrist, metacarpals, MCP/PIP/DIP; thumb CMC/MCP/IP). `--hands capsule`: the mannequin. `--hands skin`: a Skin-modifier tube stopgap. |
| `hand_model/` | `mpfb_make_hand.py` makes the hand (Blender 4.2+ with the MPFB2 extension, run on an x86-64 box). `mpfb_hands.blend` is the skinned hands and rig. `mpfb_hands.json` has the rest bones and measured finger radii. `piano.py --hand mpfb` (default) takes its segment lengths, knuckle layout, palm length and radii from this file, and solves the thumb with joint limits (CMC cone, MCP/IP hinges). MPFB2 code is GPL-3.0; the generated meshes are CC0. |
| `sampler.py` | Sampled grand piano (SFZ+FLAC) with automatic pedal and room reverb. |
| `synth.py` | Dependency-free additive piano, for quick timing checks. |

```bash
python3 -m venv .venv_blockout && .venv_blockout/bin/pip install -r requirements-blockout.txt
# piano samples (Salamander Grand Piano V3, CC-BY 3.0, Alexander Holm), into models/:
#   https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html  (SFZ+FLAC)
P=.venv_blockout/bin/python
$P -m blockout.piano examples/piano/clair_de_lune.mid work/cdl.json \
   --feeling examples/piano/clair_de_lune.feeling.json --notes-out work/cdl_notes.json
blender -b --factory-startup -P blockout/blender_piano.py -- work/cdl.json work/frames --view three4
$P -m blockout.sampler work/cdl_notes.json work/cdl.wav --sfz models/.../SalamanderGrandPiano-V3+20200602.sfz
```

Example MIDIs from the Mutopia Project: Clair de Lune, Rondo alla Turca and Brahms Op. 118 No. 2 are Public Domain; Rachmaninoff Op. 3 No. 2 is CC BY-SA 4.0. `knights_suite.mid` (built by `suite.py`) contains an original theme plus excerpts of the Rachmaninoff (CC BY-SA 4.0) and the Brahms.

## Hackathon tasks

- [ ] **Per-zone audio binding.** Bind each stem to its zone inside one run: multiple audio segments plus an attention bias, so stem A's tokens reach only zone A's video tokens.
- [ ] **Staggered per-zone timesteps.** Give zones their own timestep rows (AdaLN row ids per token). Global structure resolves first, then each zone develops in turn while the other stays noisy as context.
- [ ] **H3 over the piano blockout: the go/no-go.** Run about 5 s of a blockout through H3 at zone strengths 0.6 / 0.75 / 0.9 with the piano audio locked, then score whether each struck key lands on its onset.
- [ ] **Guitar strum blockout.** A grey picking arm and guitar strumming down/up on the onsets from `guitar_events.py`, rendered from the shot's camera, composited over the guitarist zone, then H3 at about 0.85–0.9 with the guitar stem locked.
- [ ] **Drum and bass blockouts.** Stick tips on the drums at the drum-stem hits (kick, snare, crash); bass fingers on the bass note onsets.
- [ ] **Comfy nodes for the chain.** A Beat Matrix node (stem or MIDI to events), a Primitive Performer Render node (events + instrument + camera to a grey plate), then the existing H3 Band Zone Latent, driven end to end through the Comfy API.
- [ ] **A ground-truth strum clip** for `strum_score2.py` (real footage with known stroke times), plus stroke-direction scoring.
- [ ] **Phoneme-level vocal alignment** for sung vocals, giving visemes for the mouth.
- [ ] **Seam handling in the merge pass.** Release a feathered band around touching zones instead of the whole frame.

## License

Code: MIT, see `LICENSE`. Example MIDI files carry their own licenses; see `examples/piano/README.md`.

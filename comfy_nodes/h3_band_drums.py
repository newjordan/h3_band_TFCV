"""H3 Band drum nodes: a drum stem or GM drum MIDI -> drum hits -> grey drum-kit blockout frames.

  H3 Band Drum Events       AUDIO (drum stem) -> DRUM_HITS, with h3band/drum_events.py
  H3 Band Drum Hits (MIDI)  a .mid in the ComfyUI input dir -> DRUM_HITS, with its beat grid
  H3 Band Drum Teacher      DRUM_HITS -> DRUM_HITS with techniques (choke, rimshot, cross-stick, bell, flam,
                            ghost) picked by blockout/drum_teacher.py, and the lesson sheet saying why
  H3 Band Drum Synth        DRUM_HITS -> AUDIO: blockout/drumsynth.py, techniques included (a choked crash
                            stops dead), so a MIDI alone can drive the whole graph
  H3 Band Drum Blockout     DRUM_HITS -> IMAGE frames: blockout/drums.py kinematics, rendered by Blender
                            (blockout/blender_drums.py). The frames start at `start` seconds into the hits,
                            so a CreateVideo with the same audio trimmed to `start` lines up. `emotion` is a
                            blockout/drum_style.py timeline ("0:calm,8:intense"); force/range/body/flair >= 0
                            override every preset's slider. The second output is the check and lesson report.

The nodes import the repo's blockout/ and h3band/ code, so they need the repo, not just this file: link the
comfy_nodes folder into ComfyUI/custom_nodes, or set H3B_REPO to the repo root. Blender comes from H3B_BLENDER
(default: `blender` on PATH).
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
import soundfile as sf
import torch
from comfy.utils import ProgressBar
from PIL import Image

REPO = Path(os.environ.get("H3B_REPO") or Path(__file__).resolve().parent.parent)
sys.path.append(str(REPO))
sys.path.append(str(REPO / "h3band"))
from blockout import drum_style, drum_teacher, drums, drumsynth, midi as midi_io  # noqa: E402
import drum_events  # noqa: E402

BLENDER = os.environ.get("H3B_BLENDER", "blender")
VIEWS = ["front", "three4", "side", "over", "top"]
ENGINES = ["WORKBENCH", "CYCLES"]


class H3BandDrumEvents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("DRUM_HITS",)
    FUNCTION = "run"
    CATEGORY = "h3band"

    def run(self, audio):
        wav = audio["waveform"][0].transpose(0, 1).float().cpu().numpy()
        tmp = tempfile.mkdtemp(prefix="h3b_drums_", dir=folder_paths.get_temp_directory())
        try:
            path = os.path.join(tmp, "drums.wav")
            sf.write(path, wav, int(audio["sample_rate"]))
            hits = drum_events.detect(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        counts = {p: sum(h["piece"] == p for h in hits) for p in sorted({h["piece"] for h in hits})}
        logging.info("[H3BandDrumEvents] %d hits %s", len(hits), counts)
        return ({"hits": hits, "beats": None, "downbeats": None},)


class H3BandDrumHitsMIDI:
    @classmethod
    def INPUT_TYPES(cls):
        files = [f for f in os.listdir(folder_paths.get_input_directory()) if f.lower().endswith((".mid", ".midi"))]
        return {"required": {"midi": (sorted(files),)}}

    RETURN_TYPES = ("DRUM_HITS",)
    FUNCTION = "run"
    CATEGORY = "h3band"

    @classmethod
    def VALIDATE_INPUTS(cls, midi):
        if not midi.lower().endswith((".mid", ".midi")) or not folder_paths.exists_annotated_filepath(midi):
            return "Invalid MIDI file: {}".format(midi)
        return True

    def run(self, midi):
        path = folder_paths.get_annotated_filepath(midi)
        hits = drums.hits_from_midi(midi_io.read(path))
        beats, downbeats = midi_io.beats(path)
        return ({"hits": hits, "beats": list(beats), "downbeats": list(downbeats)},)


class H3BandDrumTeacher:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hits": ("DRUM_HITS",),
                             "choke_gap": ("FLOAT", {"default": drum_teacher.CHOKE_GAP, "min": 0.25, "max": 16.0,
                                                     "step": 0.25, "tooltip": "beats of silence after a loud crash "
                                                     "that make the teacher choke it"})}}

    RETURN_TYPES = ("DRUM_HITS", "STRING")
    RETURN_NAMES = ("hits", "lesson")
    FUNCTION = "run"
    CATEGORY = "h3band"

    def run(self, hits, choke_gap):
        taught = [dict(h) for h in hits["hits"]]
        lesson = drum_teacher.teach(taught, hits["beats"], hits["downbeats"], choke_gap)
        sheet = drum_teacher.sheet(lesson)
        logging.info("[H3BandDrumTeacher]\n%s", sheet)
        return ({**hits, "hits": taught}, sheet)


class H3BandDrumSynth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hits": ("DRUM_HITS",),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "h3band"

    def run(self, hits, seed):
        wav = drumsynth.render(hits["hits"], 0.0, None, seed)
        return ({"waveform": torch.from_numpy(wav.T.copy()).unsqueeze(0), "sample_rate": drumsynth.SR},)


class H3BandDrumBlockout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hits": ("DRUM_HITS",),
                             "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                             "frames": ("INT", {"default": 121, "min": 1, "max": 100000}),
                             "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                             "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 32}),
                             "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 32}),
                             "view": (VIEWS, {"default": "front"}),
                             "head_schedule": ("STRING", {"default": "0:groove"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1})},
                "optional": {"camera": ("STRING", {"default": ""}),
                             "engine": (ENGINES, {"default": "WORKBENCH"}),
                             "motion": (list(drums.MOTIONS), {"default": "smooth"}),
                             "emotion": ("STRING", {"default": "", "tooltip": "e.g. 0:calm,8:intense (presets: "
                                                    + ", ".join(drum_style.PRESETS) + "); replaces head_schedule"}),
                             **{k: ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "0..1 on top of every emotion preset; -1 keeps the preset"})
                                for k in drum_style.SLIDERS}}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "run"
    CATEGORY = "h3band"

    def run(self, hits, start, frames, fps, width, height, view, head_schedule, seed, camera="", engine="WORKBENCH",
            motion="smooth", emotion="", **sliders):
        if view not in VIEWS or engine not in ENGINES:
            raise ValueError("view must be one of {} and engine one of {}".format(VIEWS, ENGINES))
        sched = [(float(t), st) for t, st in (x.split(":") for x in head_schedule.split(","))]
        sliders = {k: v for k, v in sliders.items() if k in drum_style.SLIDERS and v >= 0}
        if sliders and not emotion.strip():
            emotion = "0:groove"
        anim = drums.animate(hits["hits"], fps, start, frames / fps, hits["beats"], hits["downbeats"], sched, seed,
                             motion, emotion.strip() or None, sliders)
        report = json.dumps(drums.check(anim))
        if "lesson" in anim:
            report += "\n" + drum_teacher.sheet(anim["lesson"])
        logging.info("[H3BandDrumBlockout] %s", report)
        args = ["--res", "{}x{}".format(width, height), "--view", view, "--engine", engine]
        if camera.strip():
            args += ["--cam", ",".join(str(float(v)) for v in camera.split(","))]
        tmp = tempfile.mkdtemp(prefix="h3b_blockout_", dir=folder_paths.get_temp_directory())
        try:
            anim_path, out_dir = os.path.join(tmp, "anim.json"), os.path.join(tmp, "frames")
            with open(anim_path, "w") as f:
                json.dump(anim, f)
            n = len(anim["frames"])
            pbar = ProgressBar(n)
            cmd = [BLENDER, "-b", "--factory-startup", "-P", str(REPO / "blockout" / "blender_drums.py"), "--",
                   anim_path, out_dir] + args
            log = []
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as p:
                for line in p.stdout:
                    log.append(line)
                    if "Saved:" in line:
                        pbar.update(1)
            if p.returncode != 0:
                raise RuntimeError("Blender exited with {}:\n{}".format(p.returncode, "".join(log[-30:])))
            out = torch.empty((n, height, width, 3), dtype=torch.float32)
            for i in range(n):
                with Image.open(os.path.join(out_dir, "f_{:05d}.png".format(i))) as im:
                    out[i] = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return (out, report)


NODE_CLASS_MAPPINGS = {"H3BandDrumEvents": H3BandDrumEvents, "H3BandDrumHitsMIDI": H3BandDrumHitsMIDI,
                       "H3BandDrumTeacher": H3BandDrumTeacher, "H3BandDrumSynth": H3BandDrumSynth,
                       "H3BandDrumBlockout": H3BandDrumBlockout}
NODE_DISPLAY_NAME_MAPPINGS = {"H3BandDrumEvents": "H3 Band Drum Events (drum stem -> hits)",
                              "H3BandDrumHitsMIDI": "H3 Band Drum Hits (GM drum MIDI)",
                              "H3BandDrumTeacher": "H3 Band Drum Teacher (hits -> techniques + lesson)",
                              "H3BandDrumSynth": "H3 Band Drum Synth (hits -> drum audio)",
                              "H3BandDrumBlockout": "H3 Band Drum Blockout (hits -> grey kit frames)"}

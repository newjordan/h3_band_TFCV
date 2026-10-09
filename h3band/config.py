"""Paths and endpoints, all overridable by environment variables. Defaults are relative to the repo."""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def _p(var, default):
    return pathlib.Path(os.environ.get(var, default)).expanduser()


WORK = _p("H3B_WORK", REPO / "work")                       # stems, shots, results
SONG = _p("H3B_SONG", WORK / "song" / "mix.wav")            # the full mix
STEMS_DIR = _p("H3B_STEMS_DIR", WORK / "stems")             # vocals_dry.wav, guitar.wav, drums.wav, backing.wav, mix.wav
BEATS = _p("H3B_BEATS", WORK / "song" / "beats.json")       # {"beats": [seconds, ...]}
GUITAR_EVENTS = _p("H3B_GUITAR_EVENTS", WORK / "guitar_events.json")
SHOTS_DIR = _p("H3B_SHOTS_DIR", WORK / "shots")
RESULTS_DIR = _p("H3B_RESULTS_DIR", WORK / "results")
MODEL_DIR = _p("H3B_MODEL_DIR", REPO / "models")            # audio-separator checkpoints
FACE_MODEL = _p("H3B_FACE_MODEL", REPO / "models" / "face_landmarker.task")
HAND_MODEL = _p("H3B_HAND_MODEL", REPO / "models" / "hand_landmarker.task")

COMFY_URL = os.environ.get("H3B_COMFY_URL", "http://127.0.0.1:8188")
COMFY_INPUT = _p("H3B_COMFY_INPUT", REPO / "ComfyUI" / "input")
COMFY_OUTPUT = _p("H3B_COMFY_OUTPUT", REPO / "ComfyUI" / "output")
OUTPUT_PREFIX = os.environ.get("H3B_OUTPUT_PREFIX", "h3band")  # subfolder of COMFY_OUTPUT for takes

# MiniMax H3 file names as ComfyUI sees them (weights are not part of this repo)
H3_UNET = os.environ.get("H3B_UNET", "minimax_h3_fl2va_pruned_nvfp4.safetensors")
H3_CLIP = os.environ.get("H3B_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
H3_VAE = os.environ.get("H3B_VAE", "minimax_h3_video_vae_fp16.safetensors")
H3_AUDIO_VAE = os.environ.get("H3B_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")

STEM_FILES = {"vocals": "vocals_dry.wav", "backing": "backing.wav", "guitar": "guitar.wav",
              "drums": "drums.wav", "bass": "bass.wav", "mix": "mix.wav"}

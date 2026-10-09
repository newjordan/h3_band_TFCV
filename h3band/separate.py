#!/usr/bin/env python3
"""Stem separation on CPU or GPU with python-audio-separator.

  separate.py [--mix MIX.wav] [--out DIR] [stage ...]     stages: vocal dry 6s (default: all)

  vocal : BS-Roformer (Viperx 1297) vocal from the full mix
  dry   : BS-Roformer De-Reverb on that vocal -> the dry lead vocal that drives a singer's mouth
  6s    : htdemucs_6s on the full mix (guitar, bass, drums, piano, other, vocals)

The dry vocal matters most: a reverb-washed vocal has no silence between phrases, so nothing tells a
mouth when to close. Writes DIR/<stage>/ and DIR/provenance.json (model per file, runtime), then links
the generic names the pre-flight uses (vocals_dry.wav, guitar.wav, drums.wav, bass.wav) into DIR.
Stems exported from a generator (e.g. Suno stem export) can be dropped into DIR under the same names.
Set SEP_THREADS for CPU threads; CUDA_VISIBLE_DEVICES="" forces CPU.
"""
import argparse, json, os, pathlib, time

import torch

import config

torch.set_num_threads(int(os.environ.get("SEP_THREADS", "12")))
from audio_separator.separator import Separator  # noqa: E402

STAGES = [
    ("vocal", "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "mix"),
    ("dry", "deverb_bs_roformer_8_384dim_10depth.ckpt", "vocal"),
    ("6s", "htdemucs_6s.yaml", "mix"),
]
LINKS = {"dry": {"vocals_dry.wav": "(noreverb)"},
         "6s": {"guitar.wav": "(guitar)", "drums.wav": "(drums)", "bass.wav": "(bass)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=str(config.SONG)); ap.add_argument("--out", default=str(config.STEMS_DIR))
    ap.add_argument("stages", nargs="*")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    for name, model, src in STAGES:
        if a.stages and name not in a.stages:
            continue
        inp = pathlib.Path(a.mix) if src == "mix" else pathlib.Path(prov[src]["primary"])
        stage_dir = out / name
        stage_dir.mkdir(exist_ok=True)
        sep = Separator(output_dir=str(stage_dir), model_file_dir=str(config.MODEL_DIR), output_format="WAV",
                        sample_rate=48000)
        sep.load_model(model_filename=model)
        t0 = time.time()
        files = [str(stage_dir / pathlib.Path(f).name) for f in sep.separate(str(inp))]
        dt = time.time() - t0
        primary = next((f for f in files if "(noreverb)" in f.lower()), None) or \
            next((f for f in files if "(vocals)" in f.lower()), files[0])
        prov[name] = {"model": model, "input": str(inp), "files": files, "primary": primary,
                      "seconds": round(dt, 1), "threads": torch.get_num_threads()}
        prov_path.write_text(json.dumps(prov, indent=1))
        for link, key in LINKS.get(name, {}).items():
            hit = next((f for f in files if key in f.lower()), None)
            if hit:
                dst = out / link
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(pathlib.Path(hit).relative_to(out))
        print(f"[{name}] {model}: {dt:.0f}s -> {files}", flush=True)


if __name__ == "__main__":
    main()

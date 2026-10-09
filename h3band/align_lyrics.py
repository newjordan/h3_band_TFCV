#!/usr/bin/env python3
"""Forced alignment of known lyrics to a vocal stem (torchaudio MMS_FA, CTC).

LYRICS.json = {"lines": [[start_s, end_s, section, text], ...]} with rough line times.
Each lyric line is aligned inside its own window (line times from the lyrics file, padded), so one bad
line cannot drag the rest. Parenthesized text is backing vocals: aligned, flagged backing=True.

Usage: align_lyrics.py VOCAL.wav LYRICS.json OUT.json [--whisper whisper_words.json] [--pad 0.75]
--whisper: optional {"words": [[start, end, text, prob], ...]} to compare against.
Output words: [start, end, text, score, backing] plus per-character spans [start, end, char, score].
"""
import argparse, json, re

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.pipelines import MMS_FA as BUNDLE

SR = BUNDLE.sample_rate


def words_of(line):
    """[(normalized, original, backing)] for one lyric line."""
    out = []
    for m in re.finditer(r"\(([^)]*)\)|([^\s()]+)", line):
        chunk, backing = (m.group(1), True) if m.group(1) is not None else (m.group(2), False)
        for w in chunk.split():
            norm = re.sub(r"[^a-z']", "", w.lower().replace("-", ""))
            if norm.strip("'"):
                out.append((norm, w, backing))
    return out


def number_words(text):
    # the lyric sheet spells numbers out already; a.m. -> "am"
    return text.replace("a.m.", "am").replace("p.m.", "pm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocal"); ap.add_argument("lyrics"); ap.add_argument("out")
    ap.add_argument("--whisper"); ap.add_argument("--pad", type=float, default=0.75)
    a = ap.parse_args()

    x, sr = sf.read(a.vocal, always_2d=True, dtype="float32")
    wav = torch.from_numpy(x.mean(axis=1))[None]
    wav = torchaudio.functional.resample(wav, sr, SR)
    total = wav.shape[1] / SR

    model = BUNDLE.get_model(with_star=False).eval()
    dictionary = BUNDLE.get_dict(star=None)
    lines = json.load(open(a.lyrics))["lines"]

    words_out, flags = [], []
    for li, (t0, t1, section, text) in enumerate(lines):
        ws = words_of(number_words(text))
        ws = [w for w in ws if all(c in dictionary for c in w[0])]
        if not ws:
            continue
        prev_end = lines[li - 1][1] if li else 0.0
        next_start = lines[li + 1][0] if li + 1 < len(lines) else total
        # pad into silence, never past the neighbouring line's own span
        s = max(0.0, min(t0, max(t0 - a.pad, prev_end)))
        e = min(total, max(t1, min(t1 + a.pad, next_start)))
        seg = wav[:, int(s * SR):int(e * SR)]
        with torch.inference_mode():
            emission, _ = model(seg)
        tokens = [dictionary[c] for w in ws for c in w[0]]
        ali, scores = AF.forced_align(emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
        spans = AF.merge_tokens(ali[0], scores[0].exp())
        ratio = seg.shape[1] / SR / emission.shape[1]
        k = 0
        for norm, orig, backing in ws:
            cs = spans[k:k + len(norm)]
            k += len(norm)
            chars = [[round(s + c.start * ratio, 3), round(s + c.end * ratio, 3), ch, round(float(c.score), 3)]
                     for c, ch in zip(cs, norm)]
            ws0, we = chars[0][0], chars[-1][1]
            sc = round(float(np.mean([c.score for c in cs])), 3)
            words_out.append({"start": ws0, "end": we, "text": orig, "score": sc, "backing": backing,
                              "line": li, "section": section, "chars": chars})
            dur = we - ws0
            why = []
            if dur < 0.04:
                why.append(f"short {dur*1000:.0f}ms")
            if dur > max(1.5, 0.35 * len(norm)):
                why.append(f"long {dur:.2f}s")
            if sc < 0.05:
                why.append(f"low score {sc}")
            if why:
                flags.append({"line": li, "text": orig, "start": ws0, "end": we, "why": why})

    report = {"source": f"torchaudio MMS_FA forced alignment of {a.lyrics} lines on {a.vocal}",
              "n_words": len(words_out), "flags": flags,
              "words": [[w["start"], w["end"], w["text"], w["score"]] for w in words_out],
              "detail": words_out}
    if a.whisper:
        wh = json.load(open(a.whisper))["words"]
        offs, missing = [], 0
        for w in words_out:
            key = re.sub(r"[^a-z']", "", w["text"].lower())
            cand = [x for x in wh if re.sub(r"[^a-z']", "", x[2].lower()) == key and abs(x[0] - w["start"]) < 3]
            if cand:
                offs.append(min((x[0] - w["start"] for x in cand), key=abs))
            else:
                missing += 1
        offs = np.array(offs)
        wdur = np.array([x[1] - x[0] for x in wh])
        report["vs_whisper"] = {
            "matched": int(len(offs)), "unmatched": missing, "whisper_words": len(wh),
            "start_offset_median_s": round(float(np.median(offs)), 3) if len(offs) else None,
            "start_offset_abs_p90_s": round(float(np.percentile(np.abs(offs), 90)), 3) if len(offs) else None,
            "whisper_zero_or_lt40ms_durations": int((wdur < 0.04).sum()),
            "whisper_gt1500ms_durations": int((wdur > 1.5).sum()),
        }
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("words", "detail", "flags")}, indent=1))
    print("flags:", len(flags))


if __name__ == "__main__":
    main()

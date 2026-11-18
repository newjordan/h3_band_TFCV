"""Jev (TypeSafe's decision model) as the note classifier for drum_blend.py: asks what each beat of the drum part
is doing and returns the probability of each context, so a drummer blend can follow how sure the call is.

    TYPESAFE_API_KEY=... python -m blockout.drums SONG.mid OUT.json --mocap chad,carey --blend "groove=chad,fill=carey"
    python -m blockout.drum_jev SONG.mid [--downbeats-every 4]       # print the per-beat answers

One request per bar that has hand notes (POST {TYPESAFE_BASE_URL}/v1/systemone, MODEL pinned). The state is
that bar between the bars before and after it, written as text: a line per beat, each note as instrument,
velocity and where in the beat it falls. There is one Choice question per beat with hand notes, and its options are
drum_blend.CONTEXTS with the descriptions in CRITERIA. Jev answers each question with a probability for each
option. Beats with no hand notes aren't asked, and drum_blend holds the weights over them.

Answers are cached by a hash of the request (CACHE, a JSON file; H3B_JEV_CACHE moves it). A song asked once
re-runs offline and the same every time. A cache miss without TYPESAFE_API_KEY raises Unavailable, and
drum_blend then falls back to its rule classifier. 429 and 529 replies are retried with backoff and
Retry-After is honoured.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import drum_mocap

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "jev-1.13.0"            # pinned: jev-latest moves when a release ships
CACHE = os.environ.get("H3B_JEV_CACHE", os.path.join(HERE, "drum_jev_cache.json"))
BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
WORKERS = 4
RETRIES = 5
TIMEOUT_S = 30
SNAP_BEATS = 0.05               # a note this early still belongs to the next beat (humanised timing)
CRITERIA = {                    # by what the beat does, not by its instruments: a tom groove is still a groove
    "groove": "Keeping time: the main beat of the song, a pattern that repeats beat after beat and bar after bar. "
              "Usually a hi-hat, ride or bell over the kick and the snare backbeat, but a floor tom or tom pattern "
              "repeated the same way is a groove too.",
    "fill": "A fill or break: the drummer leaves the time pattern for a run around the snare and toms that changes "
            "from beat to beat instead of repeating, often rolling into the next bar or section.",
    "accent": "An accent: a loud crash, china or splash, usually with the kick, marks the beat, like the start of a "
              "section or a stab.",
    "soft": "A quiet passage: ghost notes and light strokes, the whole beat played softly.",
}
GM_NAMES = {35: "kick", 36: "kick", 37: "cross-stick", 38: "snare", 39: "hand clap", 40: "snare rimshot",
            41: "low floor tom", 42: "hi-hat closed", 43: "floor tom", 44: "hi-hat pedal", 45: "low tom",
            46: "hi-hat open", 47: "mid tom", 48: "high tom", 49: "crash", 50: "high tom", 51: "ride",
            52: "china", 53: "ride bell", 55: "splash", 57: "crash 2", 59: "ride 2"}
HAND_KINDS = ("time", "cymbal", "drum")


class Unavailable(RuntimeError):
    """No TYPESAFE_API_KEY for a request the cache doesn't have."""


def _name(h):
    if h.get("note") in GM_NAMES:
        name = GM_NAMES[h["note"]]
    else:
        name = h["piece"].replace("_", " ")
        if h["piece"].startswith("hihat") and h["piece"] != "hihat_pedal":
            name = "hi-hat open" if h.get("open") else "hi-hat closed"
    tech = h.get("tech")
    return f"{name} ({tech.replace('_', ' ')})" if tech and tech not in name else name


def beat_index(times, beats):
    """The beat each time belongs to: its beat position floored, a note up to SNAP_BEATS early counting as on."""
    return np.floor(drum_mocap.beat_pos(np.asarray(times, float), beats) + SNAP_BEATS).astype(int)


def _beat_lines(hs, b0, b1, pos, kb):
    lines = []
    for k in range(b0, b1):
        here = [(max(p - k, 0.0), h) for p, b, h in zip(pos, kb, hs) if b == k]
        notes = ", ".join(f"{_name(h)} v{h['vel']} @{f:.2f}" for f, h in here) or "rest"
        lines.append(f"  beat {k - b0 + 1}: {notes}")
    return lines


def requests(hits, beats, downbeats=None, kind=None):
    """[({question id: beat index}, request body)], one per bar with hand notes. kind(h) is drum_blend._kind."""
    beats = np.asarray(beats, float)
    allh = sorted(hits, key=lambda h: h["t"])
    pos = drum_mocap.beat_pos(np.array([h["t"] for h in allh]), beats) if allh else np.zeros(0)
    kb = beat_index([h["t"] for h in allh], beats) if allh else np.zeros(0, int)
    hand = {int(b) for b, h in zip(kb, allh) if kind(h) in HAND_KINDS and h.get("piece") != "hihat_pedal"}
    if not hand:
        return []
    if downbeats is not None and len(downbeats):
        starts = sorted({int(round(x)) for x in drum_mocap.beat_pos(np.asarray(downbeats, float), beats)})
    else:
        starts = list(range(0, len(beats), 4))
    starts = [b for b in starts if b <= max(hand)]
    if not starts or starts[0] > min(hand):     # a pickup before the first downbeat is a bar of its own
        starts = [min(hand)] + starts
    last = starts[-1] + (starts[-1] - starts[-2] if len(starts) > 1 else 4)
    bars = list(zip(starts, starts[1:] + [max(last, max(hand) + 1)]))
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    head = (f"A drum part at {60.0 / ibi:.0f} bpm, General MIDI kit. Each line is one beat; each note is the "
            "instrument, its velocity (1-127) and where in the beat it falls (@0.00 on the beat, @0.50 halfway).")
    out = []
    for i, (b0, b1) in enumerate(bars):
        asked = [k for k in range(b0, b1) if k in hand]
        if not asked:
            continue
        lines = [head]
        if i > 0:
            lines += [f"Bar {i} (before):"] + _beat_lines(allh, *bars[i - 1], pos, kb)
        lines += [f"Bar {i + 1} (the bar being asked about):"] + _beat_lines(allh, b0, b1, pos, kb)
        if i + 1 < len(bars):
            lines += [f"Bar {i + 2} (after):"] + _beat_lines(allh, *bars[i + 1], pos, kb)
        qmap = {f"beat{k - b0 + 1}": k for k in asked}
        qs = {q: {"type": "choice", "criteria": CRITERIA,
                  "instructions": f"What is the drummer doing on beat {k - b0 + 1} of bar {i + 1}?"}
              for q, k in qmap.items()}
        out.append((qmap, {"model": MODEL, "state": "\n".join(lines), "questions": qs}))
    return out


def _key(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _post(body, api_key):
    req = urllib.request.Request(BASE_URL + "/v1/systemone", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or attempt == RETRIES:
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"Jev request failed: HTTP {e.code} {detail}") from None
            wait = e.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 0.5 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == RETRIES:
                raise RuntimeError(f"Jev request failed: {e}") from None
            time.sleep(0.5 * 2 ** attempt)


def _load_cache():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def classify(hits, beats, downbeats=None, kind=None, contexts=tuple(CRITERIA)):
    """({beat index: {context: p}}, model id) from Jev, through the cache. Raises Unavailable without a key
    on a cache miss."""
    reqs = requests(hits, beats, downbeats, kind)
    cache = _load_cache()
    todo = [(qmap, body) for qmap, body in reqs if _key(body) not in cache]
    if todo:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise Unavailable(f"{len(todo)} of {len(reqs)} Jev requests not cached and no TYPESAFE_API_KEY")
        with ThreadPoolExecutor(WORKERS) as ex:
            answers = list(ex.map(lambda r: _post(r[1], api_key), todo))
        new = {_key(body): {"model": ans.get("model", MODEL),
                            "probabilities": {q: a["probabilities"] for q, a in ans["answers"].items()}}
               for (_, body), ans in zip(todo, answers)}
        cache = {**_load_cache(), **new}        # re-read: another take may have written the cache meanwhile
        tmp = f"{CACHE}.{os.getpid()}.part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(sorted(cache.items())), f, indent=0, sort_keys=True)
            f.write("\n")
        os.replace(tmp, CACHE)
    out, models = {}, set()
    for qmap, body in reqs:
        c = cache[_key(body)]
        models.add(c["model"])
        for q, k in qmap.items():
            v = np.array([max(float(c["probabilities"][q].get(x, 0.0)), 0.0) for x in contexts])
            out[k] = dict(zip(contexts, (v / v.sum() if v.sum() > 0 else np.full(len(v), 1.0 / len(v))).tolist()))
    return out, "+".join(sorted(models)) or MODEL


if __name__ == "__main__":
    import argparse
    from . import drum_blend, drums, midi
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="GM drum MIDI")
    ap.add_argument("--dry", action="store_true", help="print the requests instead of sending them")
    a = ap.parse_args()
    hits = drums.hits_from_midi(midi.read(a.src))
    bts, dbs = midi.beats(a.src)
    if a.dry:
        for _, body in requests(hits, bts, dbs, drum_blend._kind):
            print(body["state"], "\n  asks:", ", ".join(body["questions"]), "\n")
        sys.exit(0)
    probs, model = classify(hits, bts, dbs, drum_blend._kind)
    rules = {}                  # the rules label each note: every label in the beat, most notes first
    for t, c in drum_blend.classify(hits, bts):
        rules.setdefault(int(beat_index([t], bts)[0]), []).append(c)
    print(f"model {model}")
    for k in sorted(probs):
        p = probs[k]
        top = max(p, key=p.get)
        rk = sorted(set(rules.get(k, ["-"])), key=lambda c: (-rules.get(k, []).count(c), c))
        print(f"beat {k:4d} {bts[k]:7.2f}s  jev {top:6s} " + " ".join(f"{c} {v:.2f}" for c, v in p.items())
              + f"   rules {','.join(rk)}")

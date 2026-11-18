"""Style blending between drummers: a classifier labels what each note is doing, a blend matrix says how much of
each drummer plays that kind of note, and the weights move from note to note, so the blend follows the music.

    python -m blockout.drums SONG.mid OUT.json --mocap chad,carey --blend "groove=chad,fill=carey,accent=carey:0.6+chad:0.4"
    python -m blockout.drums SONG.mid OUT.json --mocap chad,baker --blend "0:chad,8:baker,16:chad:0.5+baker:0.5"
    python -m blockout.drums SONG.mid OUT.json --drummer carey          # one drummer: their mocap, and their kit

Drummers (DRUMMERS) are drum_mocap.py styles built from a drum cam each; --mocap takes their names or style paths.

Classifier: Jev (drum_jev.py, TypeSafe's decision model) when it can answer, else rules. Jev reads each bar of the
part and gives every beat a probability for each context. Each note in the beat then aims for the mix of the matrix
rows those probabilities weight, so a beat Jev isn't sure of lands between its rows. classifier="auto" (the default)
uses Jev when TYPESAFE_API_KEY is set or the song's answers are cached, "jev" insists on it, and "rules" never
asks. The rules (no learned model) label each note from the beat it falls in:
  accent  the note is a crash, splash or china at accent velocity (ACCENT_VEL)
  space   (beats with no hand notes: the weights hold over them)
  fill    two or more drum notes (snare and toms) in the beat and no timekeeping cymbal: the drums carry it
  soft    the beat's hand notes average under SOFT_VEL: ghosted grooves, quiet verses
  groove  everything else: a timekeeper (hi-hat, ride, bell) over the backbeat
A blend matrix is "context=mix,...": a mix is "drummer[:share][+drummer[:share]...]", shares normalised. "*=mix"
sets the rows left out; with no "*" they get equal shares of every drummer. A timeline "t:mix,t:mix" (seconds)
blends by time instead, crossfaded over XFADE_S like drum_style's emotions.

The weights step at each note to its context's row and are smoothed over SMOOTH_BEATS, so a fill turns the drummer
over within about half a beat, and a lone crash in a groove is a short lean toward its row. Each drummer's style is
tiled on the song's beats on its own (drum_mocap.tile, its own chunk order) and the frames mix with equal-power
weights (the square roots of the shares): two drummers' unrelated motion keeps its size instead of averaging down.
"""
import os
import sys

import numpy as np

from . import drum_jev, drum_mocap

HERE = os.path.dirname(os.path.abspath(__file__))
DRUMMERS = {
    "chad": {"mocap": "drum_mocap_chad.json", "kit": "standard",
             "from": "Chad Smith (Red Hot Chili Peppers), a drum cam of 'Otherside'"},
    "carey": {"mocap": "drum_mocap_carey.json", "kit": "big",
              "from": "Danny Carey (Tool), Vic Firth's drum cam of 'Pneuma' live in Boston"},
    "baker": {"mocap": "drum_mocap_baker.json", "kit": "double_pedal",
              "from": "Ginger Baker, a solo at home in South Africa, 2005 (Cream's channel)"},
}
CONTEXTS = ("groove", "fill", "accent", "soft")
TIMEKEEP = {42, 44, 46, 51, 53, 59}                 # GM: hi-hat closed / pedal / open, ride, bell, ride 2
CYMBAL = {49, 52, 55, 57}                           # crash, china, splash, crash 2
DRUM = {37, 38, 39, 40, 41, 43, 45, 47, 48, 50}     # snare and its strokes, the toms
KICKS = {35, 36}
ACCENT_VEL = 96
SOFT_VEL = 64
SMOOTH_BEATS = 0.5
XFADE_S = 0.6
CLASSIFIERS = ("auto", "jev", "rules")
_told = set()


def mocap_path(name):
    """A drummer's style JSON from the drummer's name, or the path as given."""
    if name in DRUMMERS:
        return os.path.join(HERE, DRUMMERS[name]["mocap"])
    return name


def drummer_name(path_or_name):
    if path_or_name in DRUMMERS:
        return path_or_name
    base = os.path.basename(str(path_or_name))
    for k, d in DRUMMERS.items():
        if base == d["mocap"]:
            return k
    return os.path.splitext(base)[0]


def _kind(h):
    """'time', 'cymbal', 'drum', 'kick' or None for a hit (its GM note, or its kit piece for hit lists)."""
    n = h.get("note")
    if n is not None:
        return ("time" if n in TIMEKEEP else "cymbal" if n in CYMBAL else "drum" if n in DRUM
                else "kick" if n in KICKS else None)
    p = h["piece"].split("_")[0].rstrip("0123456789")
    if h["piece"] == "hihat_pedal":
        return "time"
    return {"hihat": "time", "ride": "time", "crash": "cymbal", "splash": "cymbal", "china": "cymbal",
            "snare": "drum", "tom": "drum", "floor": "drum", "kick": "kick"}.get(p)


def classify(hits, beats):
    """[(t, context)] for every hand note: the context of the note in the beat it falls in (see the docstring)."""
    beats = np.asarray(beats, float)
    hand = [h for h in hits if _kind(h) in ("time", "cymbal", "drum") and h.get("piece") != "hihat_pedal"]
    if not hand:
        return []
    b = np.floor(drum_mocap.beat_pos(np.array([h["t"] for h in hand]), beats)).astype(int)
    by = {}
    for h, k in zip(hand, b):
        by.setdefault(int(k), []).append(h)
    ctx = {}
    for k, hs in by.items():
        kinds = [_kind(h) for h in hs]
        if kinds.count("drum") >= 2 and "time" not in kinds:
            ctx[k] = "fill"
        elif np.mean([h["vel"] for h in hs]) < SOFT_VEL:
            ctx[k] = "soft"
        else:
            ctx[k] = "groove"
    out = []
    for h, k in zip(hand, b):
        c = "accent" if _kind(h) == "cymbal" and h["vel"] >= ACCENT_VEL else ctx[int(k)]
        out.append((float(h["t"]), c))
    return out


def parse_mix(s, names):
    """'carey:0.7+chad:0.3' or 'carey' -> {name: share} over names, normalised."""
    w = {}
    for part in s.split("+"):
        part = part.strip()
        name, _, share = part.partition(":")
        name = name.strip()
        if name not in names:
            raise ValueError(f"unknown drummer {name!r} in blend; drummers: {', '.join(names)}")
        w[name] = w.get(name, 0.0) + (float(share) if share else 1.0)
    tot = sum(w.values())
    if tot <= 0:
        raise ValueError(f"blend mix {s!r} has no weight")
    return {n: w.get(n, 0.0) / tot for n in names}


def parse(spec, names):
    """-> ("matrix", {context: mix}) or ("timeline", [(t, mix)]). See the module docstring."""
    items = [x for x in (spec or "").split(",") if x.strip()]
    if not items:
        return "matrix", {}
    if all("=" in x for x in items):
        rows = {}
        for x in items:
            c, mix = x.split("=", 1)
            c = c.strip()
            if c != "*" and c not in CONTEXTS:
                raise ValueError(f"unknown context {c!r}; contexts: {', '.join(CONTEXTS)} (or *)")
            rows[c] = parse_mix(mix, names)
        return "matrix", rows
    tl = []
    for x in items:
        t, mix = x.split(":", 1)
        tl.append((float(t), parse_mix(mix, names)))
    return "timeline", sorted(tl, key=lambda e: e[0])


def row(rows, context, names):
    if context in rows:
        return rows[context]
    if "*" in rows:
        return rows["*"]
    return {n: 1.0 / len(names) for n in names}


def weights(times, hits, beats, names, spec=None, downbeats=None, classifier="auto"):
    """(N, len(names)) shares at `times` (rows sum to 1), the per-note contexts (Jev's likeliest, with Jev), and the
    classifier that labelled them ("rules", or the Jev model id)."""
    times = np.asarray(times, float)
    k = len(names)
    kind, data = parse(spec, names)
    if kind == "timeline":
        W = np.tile(np.array([data[0][1][n] for n in names]), (len(times), 1))
        for t0, mix in data[1:]:
            u = np.clip((times - t0) / XFADE_S, 0, 1)[:, None]
            u = u * u * (3 - 2 * u)
            W = W + (np.array([mix[n] for n in names])[None] - W) * u
        return W, [], "timeline"
    notes = classify(hits, beats)
    if not notes:
        return np.tile(np.array([row(data, "groove", names)[n] for n in names]), (len(times), 1)), [], "rules"
    nt = np.array([t for t, _ in notes])
    P = np.array([[c == x for x in CONTEXTS] for _, c in notes], float)     # (notes, contexts)
    used = "rules"
    if classifier not in CLASSIFIERS:
        raise ValueError(f"classifier must be one of {', '.join(CLASSIFIERS)}")
    if classifier != "rules":
        try:
            probs, used = drum_jev.classify(hits, beats, downbeats, _kind, CONTEXTS)
            kb = drum_jev.beat_index(nt, beats)
            P = np.array([[probs[k][x] for x in CONTEXTS] for k in kb])
            notes = [(t, CONTEXTS[int(np.argmax(p))]) for t, p in zip(nt, P)]
        except drum_jev.Unavailable as e:
            if classifier == "jev":
                raise
            if str(e) not in _told:
                _told.add(str(e))
                print(f"drum_blend: {e}; using the rule classifier", file=sys.stderr)
    targets = P @ np.array([[row(data, c, names)[n] for n in names] for c in CONTEXTS])
    idx = np.clip(np.searchsorted(nt, times, side="right") - 1, 0, len(nt) - 1)
    W = targets[idx]
    if len(times) > 1:      # smooth the steps over SMOOTH_BEATS: a box twice (a triangle) as wide as that
        ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
        dt = float(np.median(np.diff(times)))
        n = max(1, int(round(SMOOTH_BEATS * ibi / dt)))
        ker = np.ones(n) / n
        for _ in range(2):
            pad = np.concatenate([np.repeat(W[:1], n, 0), W, np.repeat(W[-1:], n, 0)])
            W = np.stack([np.convolve(pad[:, j], ker, "same") for j in range(k)], 1)[n:-n]
        W /= W.sum(1, keepdims=True)
    return W, notes, used


def tile(styles, W, beats, times, seed=0, parity=0, pitch=None):
    """(N, 6) mocap channels: each style tiled on its own (drum_mocap.tile, pitch as there), mixed with
    equal-power weights sqrt(W). Returns the channels and each style's chunk plan."""
    out = np.zeros((len(times), len(drum_mocap.CHANNELS)))
    plans = []
    for j, st in enumerate(styles):
        if not W[:, j].any():
            plans.append([])
            continue
        mo, chunks = drum_mocap.tile(st, beats, times, seed, parity, pitch)
        out += np.sqrt(W[:, j])[:, None] * mo
        plans.append(chunks)
    return out, plans

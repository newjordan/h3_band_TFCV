"""Drum technique teacher: reads a hit list in its musical context, picks a technique for the hits that need one,
and writes down why. blockout/drums.py turns each choice into motion; the lesson sheet says what it did.

Rules (each sets h["tech"] and h["why"] on a hit):
  choke        a crash (vel >= CHOKE_VEL) followed by a stop of at least CHOKE_GAP beats with no hand or kick
               hits. The free hand pinches the cymbal's edge before the next hit and the cymbal stops dead.
               The last hit of the song rings instead.
  rimshot      a loud snare (vel >= RIM_VEL) on beat 2 or 4: tip on the head and shaft on the hoop together.
               Without a beat grid, any snare at vel >= RIM_VEL_NO_GRID.
  cross_stick  GM side stick (note 37): the stick lies across the snare and its far end clicks the rim.
  bell         GM ride bell (note 53): the tip comes down steeply on the ride's bell.
  flam         a soft hit FLAM_MIN-FLAM_MAX s before a louder one on the same drum: the grace note is a low
               stroke from the other hand, the main stroke lands just after it.
  ghost        a snare at vel < GHOST_VEL: a wrist stroke a few centimetres off the head.

Hits only need "t", "piece" and "vel"; MIDI hits also carry "note" (the GM pitch), which the cross-stick and bell
rules need. Hits from an audio detector get the context rules (choke, rimshot, flam, ghost).

    python -m blockout.drum_teacher GROOVE.mid [--out lesson.json]
"""
import json

import numpy as np

CHOKE_VEL, CHOKE_GAP = 90, 1.0
RIM_VEL, RIM_VEL_NO_GRID = 112, 118
GHOST_VEL = 50
FLAM_MIN, FLAM_MAX = 0.012, 0.05
CYMBALS = ("crash", "splash", "china")      # piece types (drum_kit.piece_type): crash2 is a crash
DRUMS = ("snare", "tom", "floor")


def _ptype(p):
    return p.split("_")[0].rstrip("0123456789")       # drum_kit.piece_type


def _grid(beats, downbeats):
    beats = np.asarray(beats if beats is not None else [], float)
    downbeats = np.asarray(downbeats if downbeats is not None else beats[::4], float)
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    return beats, downbeats, period


def where(t, beats, downbeats, period):
    """(bar, beat) at time t, both 1-based; beat is fractional (2.5 = the 'and' of 2)."""
    if not len(downbeats):
        return 1 + int(t // (4 * period)), 1 + (t % (4 * period)) / period
    j = max(int(np.searchsorted(downbeats, t + 0.03, side="right")) - 1, 0)   # a hit played a bit early is on the bar
    return j + 1, 1 + (t - downbeats[j]) / period


def teach(hits, beats=None, downbeats=None, choke_gap=CHOKE_GAP):
    """Annotates hits in place (h['tech'], h['why'], and h['choke'] = {'t', 'hold'} for chokes).
    Returns the lesson: [{'t', 'bar', 'beat', 'piece', 'vel', 'tech', 'why'}] sorted by time."""
    beats, downbeats, period = _grid(beats, downbeats)
    hs = sorted(hits, key=lambda h: h["t"])
    for h in hs:
        h.pop("tech", None), h.pop("why", None), h.pop("choke", None)
    times = np.array([h["t"] for h in hs])
    struck = np.array([h["piece"] != "hihat_pedal" for h in hs])

    for i, h in enumerate(hs):
        p, v, note = _ptype(h["piece"]), h["vel"], h.get("note")
        if note == 37:
            h["tech"], h["why"] = "cross_stick", "GM side stick: lay the stick across the snare, click the far rim"
        elif note == 53:
            h["tech"], h["why"] = "bell", "GM ride bell: bring the tip down steeply on the bell"
        elif p in CYMBALS and v >= CHOKE_VEL:
            later = times[(times > h["t"] + 0.03) & struck]
            if not len(later):
                h["tech"], h["why"] = "ring", "last hit: let it ring"
                continue
            gap = float(later[0] - h["t"])
            if gap >= choke_gap * period:
                tc = h["t"] + min(0.5 * period, 0.4 * gap)
                hold = max(0.12, min(0.35, gap - (tc - h["t"]) - 0.3))
                h["tech"] = "choke"
                h["choke"] = {"t": round(tc, 4), "hold": round(hold, 4)}
                h["why"] = (f"the band stops for {gap / period:.1f} beats: pinch the edge "
                            f"{(tc - h['t']) / period:.1f} beats after the hit to cut it off")
        elif p == "snare":
            if v < GHOST_VEL:
                h["tech"], h["why"] = "ghost", "soft snare: wrist only, tip a few cm off the head"
                continue
            if len(beats):
                bar, beat = where(h["t"], beats, downbeats, period)
                on_back = abs(beat - round(beat)) < 0.12 and round(beat) in (2, 4)
                if v >= RIM_VEL and on_back:
                    h["tech"], h["why"] = "rimshot", f"loud backbeat on {round(beat)}: tip and shaft hit head and hoop together"
            elif v >= RIM_VEL_NO_GRID:
                h["tech"], h["why"] = "rimshot", "loud snare: tip and shaft hit head and hoop together"

    for i, h in enumerate(hs):   # flams last: a grace note outranks ghost
        if _ptype(h["piece"]) not in DRUMS:
            continue
        for g in hs[max(i - 4, 0):i]:
            dt = h["t"] - g["t"]
            if g["piece"] == h["piece"] and FLAM_MIN <= dt <= FLAM_MAX and g["vel"] < h["vel"] - 15:
                g["tech"], g["why"] = "flam_grace", f"grace note {1000 * dt:.0f} ms early: low stroke, other hand"
                if h.get("tech") in (None, "ghost"):
                    h["tech"], h["why"] = "flam", "main stroke of a flam, just after the grace note"

    lesson = []
    for h in hs:
        if "tech" in h:
            bar, beat = where(h["t"], beats, downbeats, period)
            lesson.append({"t": round(h["t"], 4), "bar": bar, "beat": round(beat, 2), "piece": h["piece"],
                           "vel": h["vel"], "tech": h["tech"], "why": h["why"]})
    return lesson


def sheet(lesson):
    """The lesson as text, one line per hit."""
    lines = [f"{'time':>7}  {'bar':>3} {'beat':>5}  {'piece':<7} {'hand':<4} {'technique':<12} why"]
    for e in lesson:
        hand = e.get("hand") or "-"
        note = e["why"] + (f" ({e['skipped']})" if e.get("skipped") else "")
        lines.append(f"{e['t']:7.2f}  {e['bar']:3d} {e['beat']:5.2f}  {e['piece']:<7} {hand:<4} {e['tech']:<12} {note}")
    counts = {}
    for e in lesson:
        counts[e["tech"]] = counts.get(e["tech"], 0) + 1
    lines.append("total: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from . import drums, midi
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="GM drum MIDI, or a hit list JSON (then pass --beats)")
    ap.add_argument("--beats", help="beats JSON {'beats': [...], 'downbeats': [...]}")
    ap.add_argument("--out", help="write the lesson as JSON")
    a = ap.parse_args()
    if a.src.lower().endswith((".mid", ".midi")):
        hits = drums.hits_from_midi(midi.read(a.src))
        bts, dbs = midi.beats(a.src)
    else:
        hits = json.load(open(a.src))["hits"]
        bts = dbs = None
    if a.beats:
        bj = json.load(open(a.beats))
        bts, dbs = bj["beats"], bj.get("downbeats")
    lesson = teach(hits, bts, dbs)
    print(sheet(lesson))
    if a.out:
        json.dump(lesson, open(a.out, "w"), indent=1)

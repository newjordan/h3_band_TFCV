"""Build a piano suite as one MIDI file: an original theme, then excerpts of public-domain / CC pieces, spliced on bars.

    python3 -m blockout.suite OUT.mid

The Knight's Suite: "Knight's Theme" (original, A major, 3/4) darkens onto G#7, which is exactly where
Rachmaninoff's Prelude in C# minor Op. 3 No. 2 sits at bar 45; the agitato climb (bars 35-44), the fff
four-stave climax and the pp coda follow (bars 35-61). Its last C# hands over to Brahms' Intermezzo in
A major Op. 118 No. 2, whose melody opens on C# (bars 1-16, then the close, bars 113-124): back home in A.

Output tempo map: one tempo event per output bar so every bar lasts exactly as long as in its source.
Hands: notes are re-split per onset at the widest pitch gap (track 1 = right hand, track 2 = left hand),
since the sources use 2 to 4 staves.
"""
import struct

import numpy as np

from . import midi

DIV = 480
A = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
     "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "B#": 12}


def P(name):
    """'C#5' -> MIDI pitch."""
    n, o = (name[:-1], int(name[-1]))
    return 12 * (o + 1) + A[n]


# ---------------------------------------------------------------- Knight's Theme (original)
THEME_BAR_S = 2.4          # 3/4 at 75 bpm
# melody per bar: (pitch, beats); chords per bar: (root, third, fifth[, seventh]) in the bass octave
MELODY = [
    [("E5", 2), ("C#5", 1)], [("A5", 2), ("F#5", 1)], [("F#5", 1), ("E5", 1), ("D5", 1)], [("B4", 3)],
    [("E5", 2), ("C#5", 1)], [("G#5", 2), ("E5", 1)], [("F#5", 1), ("A5", 1), ("F#5", 1)], [("E5", 2), ("D5", 1)],
    [("C#5", 1), ("B4", 1), ("C#5", 1)], [("E5", 2), ("G#4", 1)], [("A4", 1), ("B4", 1), ("D5", 1)], [("C#5", 3)],
    [("F#5", 1), ("E5", 1), ("D5", 1)], [("D5", 2), ("B4", 1)], [("D#5", 2), ("B#4", 1)], [("G#4", 3)],
]
CHORDS = [
    ("A2", "C#3", "E3"), ("F#2", "A2", "C#3"), ("D2", "F#2", "A2"), ("E2", "G#2", "B2"),
    ("A2", "C#3", "E3"), ("C#2", "E2", "G#2"), ("D2", "F#2", "A2"), ("E2", "G#2", "B2", "D3"),
    ("F#2", "A2", "C#3"), ("C#2", "E2", "G#2"), ("D2", "F#2", "A2"), ("E2", "A2", "C#3"),
    ("D2", "F#2", "A2"), ("B1", "D2", "F#2", "A2"), ("G#1", "B#1", "D#2", "F#2"), ("G#1", "B#1", "D#2", "F#2"),
]


def knights_theme():
    """Returns (notes, bars): notes as midi.Note in seconds from 0; bars = [(start_s, length_s, num, den)]."""
    notes, bars = [], []
    beat = THEME_BAR_S / 3
    for b, (mel, ch) in enumerate(zip(MELODY, CHORDS)):
        t0 = b * THEME_BAR_S
        bars.append((t0, THEME_BAR_S, 3, 4))
        t = t0
        for name, beats in mel:
            notes.append(midi.Note(t, t + beats * beat * 0.98, P(name), 80, 1))
            t += beats * beat
        # left hand: flowing eighths, root - fifth - tenth - fifth - tenth - fifth (last bar: rolled chord, held)
        root, third, fifth = (P(x) for x in ch[:3])
        if b == len(MELODY) - 1:
            for k, p in enumerate((root - 12, root, fifth, P(ch[3]) if len(ch) > 3 else third + 12)):
                notes.append(midi.Note(t0 + 0.06 * k, t0 + THEME_BAR_S, p, 70, 2))
            continue
        pat = [root, fifth, third + 12, fifth, third + 12, fifth]
        if len(ch) > 3:
            pat[3] = P(ch[3]) + 12
        for k, p in enumerate(pat):
            ts = t0 + k * beat / 2
            notes.append(midi.Note(ts, ts + beat / 2 * (2.0 if k == 0 else 0.95), p, 70, 2))
    return notes, bars


# ---------------------------------------------------------------- excerpts
def excerpt(path, bar_lo, bar_hi, num, den):
    """Bars bar_lo..bar_hi (1-based, inclusive) of a MIDI file, shifted to start at 0."""
    notes = midi.read(path)
    _, db = midi.beats(path)
    db = list(db) + [max(n.end for n in notes)]
    t0, t1 = db[bar_lo - 1], db[bar_hi]
    out = [midi.Note(n.start - t0, min(n.end, t1 + 1.5) - t0, n.pitch, n.velocity, n.track)
           for n in notes if t0 - 1e-3 <= n.start < t1 - 1e-3]
    bars = [(db[i - 1] - t0, db[i] - db[i - 1], num, den) for i in range(bar_lo, bar_hi + 1)]
    return out, bars


def resplit(notes):
    """Re-assign hands per onset: above the widest gap -> right hand (1), below -> left (2); max 5 each side kept
    as-is (the extra notes still sound; the fingering keeps the outer ones)."""
    groups = {}
    for n in notes:
        groups.setdefault(round(n.start, 2), []).append(n)
    last_split = 60
    for _, g in sorted(groups.items()):
        ps = sorted({n.pitch for n in g})
        if len(ps) == 1:
            split = last_split
        else:
            gaps = [(ps[i + 1] - ps[i] - 0.15 * abs((ps[i] + ps[i + 1]) / 2 - last_split), i) for i in range(len(ps) - 1)]
            _, i = max(gaps)
            split = (ps[i] + ps[i + 1]) / 2
            last_split = 0.7 * last_split + 0.3 * split
        for n in g:
            n.track = 1 if n.pitch > split else 2
    return notes


# ---------------------------------------------------------------- MIDI writer
def _vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F)); v >>= 7
    return bytes(reversed(out))


def write(path, notes, bars):
    """bars: [(start_s, length_s, num, den)] contiguous. One tempo event per bar keeps every bar's real length."""
    bt = [(b[0], DIV * 4 * b[2] // b[3]) for b in bars]       # (start s, ticks per bar)
    tick0 = np.concatenate([[0], np.cumsum([t for _, t in bt])])
    starts = np.array([b[0] for b in bars] + [bars[-1][0] + bars[-1][1]])

    def tick(s):
        i = int(np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(bars) - 1))
        u = (s - starts[i]) / bars[i][1]
        return int(round(tick0[i] + u * bt[i][1]))

    ev = []
    sig = None
    for i, (s0, L, num, den) in enumerate(bars):
        q = 4 * num / den
        ev.append((int(tick0[i]), 0, b"\xff\x51\x03" + int(round(L / q * 1e6)).to_bytes(3, "big")))
        if (num, den) != sig:
            ev.append((int(tick0[i]), 0, b"\xff\x58\x04" + bytes([num, int(np.log2(den)), 24, 8])))
            sig = (num, den)
    tracks = {1: [], 2: []}
    for n in notes:
        a, b = tick(n.start), max(tick(n.end), tick(n.start) + 1)
        tracks[1 if n.track == 1 else 2] += [(a, 2, bytes([0x90, n.pitch, n.velocity])), (b, 1, bytes([0x80, n.pitch, 0]))]

    def chunk(evts):
        evts.sort(key=lambda e: (e[0], e[1]))
        out, last = b"", 0
        for t, _, data in evts:
            out += _vlq(t - last) + data; last = t
        out += _vlq(0) + b"\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(out)) + out

    data = b"MThd" + struct.pack(">IHHH", 6, 1, 3, DIV) + chunk(ev) + chunk(tracks[1]) + chunk(tracks[2])
    open(path, "wb").write(data)


def knights_suite(rach, brahms):
    parts = [knights_theme(),
             excerpt(rach, 35, 61, 4, 4),
             excerpt(brahms, 1, 16, 3, 4),
             excerpt(brahms, 113, 124, 3, 4)]
    notes, bars, t = [], [], 0.0
    marks = []
    for k, (ns, bs) in enumerate(parts):
        marks.append((len(bars) + 1, t))
        notes += [midi.Note(n.start + t, n.end + t, n.pitch, n.velocity, n.track) for n in ns]
        bars += [(b0 + t, L, num, den) for b0, L, num, den in bs]
        t += sum(b[1] for b in bs)
        if k == 1:      # a silent 3/4 bar of air between the Rachmaninoff coda and the Brahms
            bars.append((t, 2.25, 3, 4)); t += 2.25
    return resplit(notes), bars, marks


if __name__ == "__main__":
    import sys
    rach, brahms = "examples/piano/rach_prelude_op3_no2.mid", "examples/piano/brahms_intermezzo_op118_no2.mid"
    notes, bars, marks = knights_suite(rach, brahms)
    write(sys.argv[1], notes, bars)
    print(f"{len(notes)} notes, {len(bars)} bars, {bars[-1][0] + bars[-1][1]:.1f} s; sections start at bars",
          [m[0] for m in marks], "times", [round(m[1], 1) for m in marks])

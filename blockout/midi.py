"""Minimal Standard MIDI File reader: notes in seconds, with the tempo map applied. No dependencies.

Note = (start_s, end_s, pitch, velocity, track). Running status, sysex and meta events are handled;
only note on/off and set-tempo are used.
"""
import struct
from dataclasses import dataclass


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    track: int


def _varlen(b, i):
    v = 0
    while True:
        c = b[i]; i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def _events(trk):
    """Yield (abs_tick, status, data bytes) for one track chunk."""
    i, tick, status = 0, 0, None
    while i < len(trk):
        d, i = _varlen(trk, i)
        tick += d
        if trk[i] & 0x80:
            status = trk[i]; i += 1
        if status == 0xFF:
            kind = trk[i]; n, i = _varlen(trk, i + 1)
            yield tick, (0xFF, kind), trk[i:i + n]; i += n
        elif status in (0xF0, 0xF7):
            n, i = _varlen(trk, i); i += n
        else:
            n = 1 if status & 0xF0 in (0xC0, 0xD0) else 2
            yield tick, status, trk[i:i + n]; i += n


def _parse(path):
    b = open(path, "rb").read()
    assert b[:4] == b"MThd", "not a MIDI file"
    _, ntrk, div = struct.unpack(">HHH", b[8:14])
    assert not div & 0x8000, "SMPTE time division not supported"
    tracks, i = [], 14
    while len(tracks) < ntrk:
        cid, n = b[i:i + 4], struct.unpack(">I", b[i + 4:i + 8])[0]
        if cid == b"MTrk":
            tracks.append(list(_events(b[i + 8:i + 8 + n])))
        i += 8 + n

    tempo = sorted((t, int.from_bytes(d, "big")) for trk in tracks for t, s, d in trk
                   if s == (0xFF, 0x51))
    tempo = tempo or [(0, 500000)]
    if tempo[0][0] != 0:
        tempo.insert(0, (0, 500000))
    marks, sec = [], 0.0  # (tick, seconds at tick, us per quarter)
    for k, (t, us) in enumerate(tempo):
        if k:
            sec += (t - marks[-1][0]) * marks[-1][2] / 1e6 / div
        marks.append((t, sec, us))

    def to_s(tick):
        m = marks[0]
        for mk in marks:
            if mk[0] <= tick:
                m = mk
        return m[1] + (tick - m[0]) * m[2] / 1e6 / div

    return tracks, div, to_s


def read(path):
    tracks, div, to_s = _parse(path)
    notes = []
    for ti, trk in enumerate(tracks):
        held = {}
        for tick, s, d in trk:
            if isinstance(s, tuple):
                continue
            kind, ch = s & 0xF0, s & 0x0F
            if kind == 0x90 and d[1] > 0:
                held.setdefault((ch, d[0]), []).append((tick, d[1]))
            elif kind == 0x80 or (kind == 0x90 and d[1] == 0):
                if held.get((ch, d[0])):
                    t0, vel = held[(ch, d[0])].pop(0)
                    notes.append(Note(to_s(t0), to_s(tick), d[0], vel, ti))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def beats(path, speed=1.0):
    """Beat and downbeat times (s) from the time-signature map. Compound metres (6/8, 9/8, 12/8) beat in
    dotted quarters. Returns (beats, downbeats) as lists, divided by speed like the notes."""
    tracks, div, to_s = _parse(path)
    sigs = sorted((t, d[0], 2 ** d[1]) for trk in tracks for t, s, d in trk if s == (0xFF, 0x58)) or [(0, 4, 4)]
    if sigs[0][0] != 0:
        sigs.insert(0, (0, 4, 4))
    last = max(t for trk in tracks for t, _, _ in trk)
    bts, dbs = [], []
    for k, (t0, num, den) in enumerate(sigs):
        t1 = sigs[k + 1][0] if k + 1 < len(sigs) else last
        unit = div * 4 // den
        step = unit * 3 if (den == 8 and num % 3 == 0 and num > 3) else unit
        bar = unit * num
        t = t0
        while t < t1:
            dbs.append(to_s(t) / speed)
            for b in range(t, min(t + bar, t1), step):
                bts.append(to_s(b) / speed)
            t += bar
    return bts, dbs


if __name__ == "__main__":
    import sys
    ns = read(sys.argv[1])
    print(f"{len(ns)} notes, {ns[-1].end:.1f} s, pitch {min(n.pitch for n in ns)}-{max(n.pitch for n in ns)}")
    for n in ns[:40]:
        print(f"{n.start:7.3f} {n.end - n.start:6.3f}  p{n.pitch:3d} v{n.velocity:3d} trk{n.track}")

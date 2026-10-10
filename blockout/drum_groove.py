"""Original 16-bar rock groove for the drum blockout, written as a General MIDI drum file (channel 10).

    python -m blockout.drum_groove OUT.mid [--bpm 112] [--seed 0] [--song rock|technique|showcase|toss]

rock: bars 1-4 hi-hat groove with a snare pickup, 5-8 crash into a ride verse with ghost notes and pedal chicks,
ending in a tom fill, 9-11 open-hat lifts, 12 a fill round the kit, 13-15 a crash chorus, 16 the last hit.
technique: 9 bars for the technique teacher (technique_bars()).
showcase: 14 bars (30 s) for a big kit with a double pedal, splash and china (showcase_bars()).
toss: 5 bars with a gap for a stick toss and catch (toss_bars()).
Timing and velocity are humanised a little (seeded). Composed for this repo, MIT like the code.
"""
import struct

import numpy as np

KICK, SNARE, HAT, PEDAL, OPEN, CRASH, CRASH2, RIDE, TOM1, TOM2, FLOOR = 36, 38, 42, 44, 46, 49, 57, 51, 48, 45, 43
DIV = 480


def _bar(**parts):
    """parts: name=[(step, vel), ...] in 16th steps."""
    return [(s, p, v) for p, hits in parts.items() for s, v in hits]


def eighths(vel_on=95, vel_off=70, skip=()):
    return [(s, vel_on if s % 4 == 0 else vel_off) for s in range(0, 16, 2) if s not in skip]


def bars():
    hat = lambda **k: [(s, HAT, v) for s, v in eighths(**k)]
    ride = lambda **k: [(s, RIDE, v) for s, v in eighths(88, 72, **k)]
    back = [(4, SNARE, 112), (12, SNARE, 114)]
    out = []
    # 1-4: hi-hat groove
    out.append(hat() + back + [(0, KICK, 110), (8, KICK, 104)])
    out.append(hat() + back + [(0, KICK, 110), (8, KICK, 104), (10, KICK, 92)])
    out.append(hat() + back + [(0, KICK, 110), (6, KICK, 90), (8, KICK, 104)])
    out.append(hat(skip=(12, 14)) + [(4, SNARE, 112), (0, KICK, 110), (8, KICK, 104)]
               + [(12, SNARE, 72), (13, SNARE, 84), (14, SNARE, 98), (15, SNARE, 116)])
    # 5-8: ride verse, ghost notes, pedal chicks on 2 and 4
    chicks = [(4, PEDAL, 70), (12, PEDAL, 70)]
    out.append([(0, CRASH, 122)] + ride(skip=(0,)) + back + chicks
               + [(7, SNARE, 36), (15, SNARE, 38), (0, KICK, 118), (8, KICK, 104), (10, KICK, 96)])
    out.append(ride() + back + chicks + [(3, SNARE, 34), (11, SNARE, 38), (0, KICK, 110), (6, KICK, 92), (8, KICK, 104)])
    out.append(ride() + back + chicks + [(7, SNARE, 36), (14, SNARE, 40), (0, KICK, 110), (8, KICK, 104), (10, KICK, 96)])
    out.append([(s, RIDE, v) for s, v in eighths(88, 72) if s < 8] + [(4, SNARE, 112), (0, KICK, 110)]
               + [(8, TOM1, 100), (9, TOM1, 104), (10, TOM2, 106), (11, TOM2, 108),
                  (12, FLOOR, 112), (13, FLOOR, 114), (14, FLOOR, 118), (15, FLOOR, 120)] + [(8, KICK, 100), (12, KICK, 104)])
    # 9-11: hi-hat with open lifts on the "and" of 4 (and of 2 in bar 10)
    lift = lambda steps: [(s, OPEN if s in steps else HAT, v + (12 if s in steps else 0)) for s, v in eighths()]
    out.append([(0, CRASH, 120)] + [x for x in lift((14,)) if x[0] != 0] + back + [(0, KICK, 118), (8, KICK, 104), (9, KICK, 90)])
    out.append(lift((6, 14)) + back + [(0, KICK, 110), (8, KICK, 104), (10, KICK, 94)])
    out.append(lift((14,)) + back + [(0, KICK, 110), (6, KICK, 90), (8, KICK, 104)])
    # 12: fill round the kit, quarter kicks under it
    out.append([(s, SNARE, 70 + 8 * s) for s in range(4)] + [(s, TOM1, 100 + 2 * (s - 4)) for s in range(4, 8)]
               + [(s, TOM2, 106 + 2 * (s - 8)) for s in range(8, 12)] + [(s, FLOOR, 112 + 3 * (s - 12)) for s in range(12, 16)]
               + [(s, KICK, 100) for s in (0, 4, 8, 12)])
    # 13-15: chorus on the ride, crashes on the bar, driving kick
    drive = [(0, KICK, 120), (3, KICK, 100), (8, KICK, 112), (11, KICK, 100)]
    out.append([(0, CRASH, 124)] + ride(skip=(0,)) + back + drive + [(8, CRASH2, 110)])
    out.append([(0, CRASH2, 118)] + ride(skip=(0,)) + back + drive)
    out.append([(0, CRASH, 122)] + ride(skip=(0, 12, 14)) + [(4, SNARE, 112), (12, SNARE, 100), (13, SNARE, 108),
                                                           (14, TOM1, 114), (15, FLOOR, 118)] + drive[:3])
    # 16: the last hit
    out.append([(0, CRASH, 127), (0, CRASH2, 124), (0, KICK, 127), (0, FLOOR, 120)])
    return out


SIDE, BELL = 37, 53


def technique_bars():
    """9 bars that give blockout/drum_teacher.py something to teach: cross-stick verse, ride bell with a flam,
    a ghost-to-full snare crescendo, crash stabs with stops (chokes), a rimshot groove, a fill, a stop, the end."""
    hat = lambda **k: [(s, HAT, v) for s, v in eighths(**k)]
    out = []
    # 1-2: cross-stick verse
    for _ in range(2):
        out.append(hat(vel_on=80, vel_off=60) + [(4, SIDE, 100), (12, SIDE, 104), (0, KICK, 100), (6, KICK, 84),
                                                  (8, KICK, 96)])
    # 3-4: ride bell on the beat, bow on the "and", snare backbeats, a flam on the last backbeat
    bell = [(s, BELL if s % 4 == 0 else RIDE, 100 if s % 4 == 0 else 74) for s in range(0, 16, 2)]
    out.append(bell + [(4, SNARE, 104), (12, SNARE, 106), (0, KICK, 108), (8, KICK, 102)])
    out.append(bell + [(4, SNARE, 104), (0, KICK, 108), (8, KICK, 102), (10, KICK, 92), (12, SNARE, 116),
                       (11.8, SNARE, 46)])
    # 5: snare 16ths from ghost to full, quarter kicks
    out.append([(s, SNARE, int(30 + 97 * s / 15)) for s in range(16)] + [(s, KICK, 104) for s in (0, 4, 8, 12)])
    # 6: stabs: crash + kick on 1 and 3, the band stops in between
    out.append([(0, CRASH, 122), (0, KICK, 120), (8, CRASH2, 120), (8, KICK, 118)])
    # 7: rock groove, loud backbeats (rimshots), crash on 1
    out.append([(0, CRASH, 118)] + hat(skip=(0,)) + [(4, SNARE, 122), (12, SNARE, 124), (0, KICK, 112),
                                                       (8, KICK, 104), (10, KICK, 94)])
    # 8: tom fill into a crash stop on 3
    out.append([(s, TOM1 if s < 4 else FLOOR, 104 + 2 * s) for s in range(8)] + [(0, KICK, 104), (4, KICK, 104),
                (8, CRASH, 124), (8, KICK, 122)])
    # 9: the last hit
    out.append([(0, CRASH, 127), (0, CRASH2, 124), (0, KICK, 127), (0, FLOOR, 120)])
    return out


SPLASH, CHINA = 55, 52
TOMS6 = (50, 48, 47, 45, 43, 41)        # GM high tom .. low floor tom


def showcase_bars():
    """14 bars (30 s at 112 bpm) for a big kit (drum_kit.py "big"): rock and technique grooves, a splash, a china
    groove over double-kick 16ths, runs over six toms, stops with chokes, a crash chorus. On a smaller kit the
    extra GM notes fold onto its pieces (drum_kit.gm_map)."""
    hat = lambda **k: [(s, HAT, v) for s, v in eighths(**k)]
    back = [(4, SNARE, 112), (12, SNARE, 114)]
    dbl = lambda steps, v=100: [(s, KICK, v + (8 if s % 4 == 0 else 0)) for s in steps]
    out = []
    out.append(hat() + back + [(0, KICK, 110), (8, KICK, 104)])
    out.append(hat(skip=(14,)) + back + [(0, KICK, 110), (8, KICK, 104), (10, KICK, 92), (14, SPLASH, 112)])
    out.append(hat(vel_on=80, vel_off=60) + [(4, SIDE, 100), (12, SIDE, 104), (0, KICK, 100), (6, KICK, 84),
                                              (8, KICK, 96)])
    bell = [(s, BELL if s % 4 == 0 else RIDE, 100 if s % 4 == 0 else 74) for s in range(0, 16, 2)]
    out.append(bell + [(4, SNARE, 104), (0, KICK, 108), (8, KICK, 102), (10, KICK, 92), (12, SNARE, 116),
                       (11.8, SNARE, 46)])
    out.append([(s, SNARE, int(30 + 97 * s / 15)) for s in range(16)] + [(s, KICK, 104) for s in (0, 4, 8, 12)])
    floors = lambda steps, v: [(s, TOMS6[5 - s % 2], v) for s in steps]     # 16ths across both floors, low on 1
    out.append([(s, TOMS6[s // 2], 104 + s) for s in range(10)] + floors(range(10, 16), 118)
               + [(0, KICK, 104), (4, KICK, 104), (8, KICK, 104), (12, KICK, 110)])
    china = lambda skip=(): [(s, CHINA, 112 if s % 4 == 0 else 96) for s in range(0, 16, 2) if s not in skip]
    out.append([(0, CRASH, 122)] + china(skip=(0,)) + back + dbl(range(8)) + [(8, KICK, 108), (10, KICK, 100)])
    out.append([(0, CRASH2, 120)] + china(skip=(0,)) + back + dbl(range(16)))
    out.append([(0, CRASH, 122), (0, KICK, 120), (8, CRASH2, 120), (8, KICK, 118)])
    out.append([(0, CRASH, 118)] + hat(skip=(0,)) + [(4, SNARE, 122), (12, SNARE, 124), (0, KICK, 112),
                                                       (8, KICK, 104), (10, KICK, 94)])
    lift = [(s, OPEN if s in (6, 14) else HAT, v + (12 if s in (6, 14) else 0)) for s, v in eighths()]
    out.append(lift + back + [(0, KICK, 110), (8, KICK, 104), (10, KICK, 94)])
    ride = [(s, RIDE, v) for s, v in eighths(88, 72) if s != 0]
    out.append([(0, CRASH, 124)] + ride + back + [(0, KICK, 120), (3, KICK, 100), (8, KICK, 112), (11, KICK, 100),
                                                  (8, CRASH2, 110)])
    out.append([(s, SNARE, 80 + 8 * s) for s in range(4)] + [(s, TOMS6[(s - 4) // 2], 106 + s) for s in range(4, 12)]
               + floors(range(12, 16), 122)
               + [(0, KICK, 104), (8, KICK, 104)] + dbl(range(12, 16), 104))
    out.append([(0, CRASH, 127), (0, CHINA, 124), (0, KICK, 127)])
    return out


def toss_bars():
    """5 bars (10.7 s at 112 bpm) with room for a stick toss: bar 3 has no hi-hat, so with a right-hand toss
    the left hand takes the crash on 1 and the backbeat while the right is free until bar 4 (at 112 bpm: release
    about 4.85 s, catch by 5.6 s)."""
    hat = lambda **k: [(s, HAT, v) for s, v in eighths(**k)]
    back = [(4, SNARE, 112), (12, SNARE, 114)]
    out = [hat() + back + [(0, KICK, 110), (8, KICK, 104)],
           hat() + back + [(0, KICK, 110), (8, KICK, 104), (10, KICK, 92)],
           [(0, CRASH, 122), (0, KICK, 118), (4, PEDAL, 70), (12, PEDAL, 70)] + back
           + [(8, KICK, 104), (10, KICK, 94)],
           hat(skip=(12, 14)) + [(4, SNARE, 112), (0, KICK, 110), (8, KICK, 104)]
           + [(12, SNARE, 96), (13, SNARE, 104), (14, TOM1, 110), (15, FLOOR, 116)],
           [(0, CRASH, 127), (0, CRASH2, 124), (0, KICK, 127)]]
    return out


SONGS = {"rock": bars, "technique": technique_bars, "showcase": showcase_bars, "toss": toss_bars}


def notes(bpm=112, seed=0, song="rock"):
    """[(start_s, pitch, vel)] for the whole groove."""
    rng = np.random.default_rng(seed)
    step = 60.0 / bpm / 4
    out = []
    for b, bar in enumerate(SONGS[song]()):
        for s, p, v in bar:
            jitter = 0.0 if (s == 0 and b == 0) else float(rng.normal(0, 0.004))
            out.append((max(0.0, (b * 16 + s) * step + jitter), p, int(np.clip(v + rng.normal(0, 4), 20, 127))))
    return sorted(out)


def _vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F)); v >>= 7
    return bytes(reversed(out))


def write(path, ns, bpm):
    tick = lambda s: int(round(s * bpm / 60.0 * DIV))
    ev = [(0, 0, b"\xff\x51\x03" + int(round(60e6 / bpm)).to_bytes(3, "big")),
          (0, 0, b"\xff\x58\x04" + bytes([4, 2, 24, 8]))]
    for t, p, v in ns:
        ev += [(tick(t), 2, bytes([0x99, p, v])), (tick(t) + DIV // 8, 1, bytes([0x89, p, 0]))]
    ev.sort(key=lambda e: (e[0], e[1]))
    trk, last = b"", 0
    for t, _, data in ev:
        trk += _vlq(t - last) + data; last = t
    trk += _vlq(0) + b"\xff\x2f\x00"
    open(path, "wb").write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, DIV) + b"MTrk" + struct.pack(">I", len(trk)) + trk)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, default=112)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--song", choices=SONGS, default="rock")
    a = ap.parse_args()
    ns = notes(a.bpm, a.seed, a.song)
    write(a.out, ns, a.bpm)
    print(f"{len(ns)} hits, {len(SONGS[a.song]()) * 240 / a.bpm:.1f} s -> {a.out}")

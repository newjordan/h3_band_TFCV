"""Original 16-bar rock groove for the drum blockout, written as a General MIDI drum file (channel 10).

    python -m blockout.drum_groove OUT.mid [--bpm 112] [--seed 0]
                                   [--song rock|technique|showcase|toss|random|tool|baker]

rock: bars 1-4 hi-hat groove with a snare pickup, 5-8 crash into a ride verse with ghost notes and pedal chicks,
ending in a tom fill, 9-11 open-hat lifts, 12 a fill round the kit, 13-15 a crash chorus, 16 the last hit.
technique: 9 bars for the technique teacher (technique_bars()).
showcase: 14 bars (30 s) for a big kit with a double pedal, splash and china (showcase_bars()).
toss: 5 bars with a gap for a stick toss and catch (toss_bars()).
random: 9 bars of a groove made up from the seed and tempo (random_bars()), a different one for every seed.
tool: a song in Danny Carey's idiom from the seed and tempo, odd and shifting meters, for the big kit (tool_bars()).
baker: a song in Ginger Baker's idiom, tom grooves over two bass drums, swing and 12/8 (baker_bars()).
A bar is a list of hits (16th step, GM pitch, velocity[, the main step a grace note leads]) in 4/4, or a Bar, a
list with its own meter: Bar(hits, (7, 8)) is 14 16ths long. notes() lays the bars end to end by their lengths
and write() puts a time signature wherever the meter changes. Timing and velocity are humanised a little
(seeded). Composed for this repo, MIT like the code.
"""
import struct

import numpy as np

KICK, SNARE, HAT, PEDAL, OPEN, CRASH, CRASH2, RIDE, TOM1, TOM2, FLOOR = 36, 38, 42, 44, 46, 49, 57, 51, 48, 45, 43
DIV = 480


class Bar(list):
    """A bar in its own meter: Bar(hits, (7, 8)) is a 7/8 bar, 14 16th steps long (steps 0 .. 13.99). A plain list
    is a 4/4 bar, so every bar list written before meters still works."""

    def __init__(self, hits=(), meter=(4, 4)):
        super().__init__(hits)
        self.meter = (int(meter[0]), int(meter[1]))


def meter(bar):
    return getattr(bar, "meter", (4, 4))


def bar_steps(bar_or_meter):
    """16th steps in a bar (or a (num, den) meter): 4/4 16, 7/8 14, 5/4 20, 6/8 12."""
    num, den = bar_or_meter if isinstance(bar_or_meter, tuple) else meter(bar_or_meter)
    assert (num * 16) % den == 0, f"{num}/{den} is not a whole number of 16ths"
    return num * 16 // den


def beat_steps(m):
    """Steps where the beats of a (num, den) meter fall, as midi.beats counts them: quarters; compound /8 (6/8,
    9/8, 12/8) in dotted quarters; other odd /8 in quarters with the leftover 8th on the last beat (7/8 = 2+2+3)."""
    num, den = m
    n = bar_steps(m)
    if den == 8 and num % 3 == 0 and num > 3:
        return list(range(0, n, 6))
    if den == 8 and num % 2 and num > 3:
        return list(range(0, n - 2, 4))
    return list(range(0, n, 16 // den))


def meter_map(bars):
    """[(start step, num, den)] wherever the meter changes, from the first bar on."""
    out, at = [], 0
    for b in bars:
        if not out or out[-1][1:] != meter(b):
            out.append((at, *meter(b)))
        at += bar_steps(b)
    return out


def seconds(bars, bpm):
    return sum(bar_steps(b) for b in bars) * 15.0 / bpm


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


HAND_PRIORITY = {CRASH: 6, CRASH2: 6, SPLASH: 5, CHINA: 5, SNARE: 4, SIDE: 4, **{t: 4 for t in TOMS6},
                 BELL: 2, RIDE: 2, OPEN: 1, HAT: 1}


def random_bars(seed=0, bpm=112, n=9):
    """n bars of a seeded random groove (drum_random.py plays the drummer on these, beats it wasn't tuned on).
    Each 4-bar phrase picks a timekeeper (closed hi-hat in 8ths or 16ths, open-hat lifts, a shuffle, ride, ride
    bell, china or quarter-note crash), a backbeat (2 and 4, half time or cross-stick), ghost notes or not, and a
    kick pattern off the 16th grid that each bar varies a little. Most phrases end on a fill over random toms
    and the snare (8ths, 16ths or 8th-note triplets) into a crash. Now and then a flam, a splash accent, a
    double-kick run or a stop (a crash, then silence: the teacher chokes it). A human could play every bar: at
    most two hands at once, no hand faster than 16ths at this tempo, 16ths on one hand only up to about 125 bpm."""
    rng = np.random.default_rng([seed, 7])
    s16 = 60.0 / bpm / 4                                       # seconds per 16th
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    chance = lambda p: bool(rng.random() < p)
    vel = lambda v: int(np.clip(v + rng.normal(0, 5), 25, 127))
    out, after_fill = [], False
    for b in range(n):
        if b % 4 == 0:
            keep = pick(["hat8", "hat8", "hat16", "open", "shuffle", "ride", "ride", "bell", "china", "quarter"])
            if keep == "hat16" and s16 < 0.12:
                keep = "hat8"
            back = pick(["back", "back", "back", "half", "side"])
            kicks = {0} | {int(x) for x in rng.choice(16, int(rng.integers(1, 5)), replace=False)}
            ghosts = chance(0.4) and back != "side"
            pedal = keep in ("ride", "bell", "china", "quarter") and chance(0.6)
        bar = []
        if b == n - 1:
            out.append([(0, CRASH, vel(124)), (0, pick([CRASH2, CHINA]), vel(118)), (0, KICK, vel(124))])
            break
        fill = b % 4 == 3 and chance(0.8)
        stop = not fill and b % 4 in (1, 2) and chance(0.08)
        f0 = pick([8, 12, 12, 0 if s16 >= 0.09 else 8]) if fill else 16
        if stop:
            out.append([(0, pick([CRASH, CRASH2]), vel(122)), (0, KICK, vel(118))])
            after_fill = True
            continue
        # timekeeper, up to the fill
        steps = {"hat16": range(16), "quarter": range(0, 16, 4)}.get(keep, range(0, 16, 2))
        if keep == "shuffle":
            steps = [q + d for q in range(0, 16, 4) for d in (0, 8 / 3)]
        piece = {"hat8": HAT, "hat16": HAT, "open": HAT, "shuffle": HAT, "ride": RIDE, "bell": BELL, "china": CHINA,
                 "quarter": pick([CRASH, CRASH2])}[keep]
        lifts = {14} | ({6} if chance(0.5) else set()) if keep == "open" else set()
        for st in steps:
            if st < f0:
                pc = OPEN if st in lifts else (RIDE if keep == "bell" and st % 4 else piece)
                bar.append((st, pc, vel(96 if st % 4 == 0 else 74) + (10 if st in lifts else 0)))
        if after_fill or (b % 4 == 0 and chance(0.4)):
            bar = [x for x in bar if x[0] != 0] + [(0, pick([CRASH, CRASH, CRASH2]), vel(120))]
        if f0 > 14 and chance(0.12):
            bar = [x for x in bar if x[0] != 14] + [(14, SPLASH, vel(112))]
        # backbeat and ghosts
        hits = {"back": [4, 12], "half": [8], "side": [4, 12]}[back]
        for st in hits:
            if st < f0:
                bar.append((st, SIDE if back == "side" else SNARE, vel(112)))
        if back != "side" and chance(0.2) and hits[-1] < f0:   # a flam: both hands, so no cymbal on its beat
            bar = [x for x in bar if x[0] != hits[-1] or x[1] == SNARE]
            bar.append((hits[-1] - round(0.03 / s16, 2), SNARE, vel(48), hits[-1]))   # the grace note 30 ms early
        if ghosts:
            free = [st for st in range(1, min(f0, 16), 2) if all(abs(st - h) > 1 for h in hits)]
            for st in rng.choice(free, min(len(free), int(rng.integers(1, 4))), replace=False) if free else []:
                bar.append((int(st), SNARE, vel(36)))
        # kicks: the phrase's pattern, one step moved now and then, a double-kick run sometimes
        ks = set(kicks)
        if chance(0.4):
            ks ^= {int(rng.integers(1, 16))}
        if chance(0.1) and s16 >= 0.075:
            r0 = pick([8, 12])
            ks |= set(range(r0, 16))
        ks |= {0} if after_fill else set()
        for st in sorted(ks):
            if st < f0 or st % 4 == 0:
                bar.append((st, KICK, vel(110 if st % 4 == 0 else 96)))
        if pedal:
            bar += [(4, PEDAL, vel(70)), (12, PEDAL, vel(70))]
        # fill over random toms and the snare
        if fill:
            rate = pick([2, 1, 4 / 3] if s16 >= 0.075 else [2, 4 / 3])
            if rate == 1 and s16 < 0.075:
                rate = 2
            seq, st, tom = [], float(f0), int(rng.integers(0, 3))
            while st < 16 - 1e-6:
                if chance(0.35):
                    tom = min(5, tom + 1)                       # mostly round the kit, high to low
                seq.append((round(st, 3), pick([SNARE, TOMS6[tom]]) if tom < 2 and chance(0.3) else TOMS6[tom]))
                st += rate
            bar += [(t, pc, vel(100 + 18 * (t - f0) / max(1, 16 - f0))) for t, pc in seq]
        after_fill = fill
        out.append(_two_hands(bar))
    return out


def _two_hands(bar):
    """Drops the least important hand hits wherever a bar asks for more than two at once."""
    by = {}
    for x in bar:
        by.setdefault(round(x[0], 3), []).append(x)
    keep = []
    for st, xs in by.items():
        feet = [x for x in xs if x[1] in (KICK, PEDAL)]
        hands = sorted((x for x in xs if x[1] not in (KICK, PEDAL)), key=lambda x: -HAND_PRIORITY.get(x[1], 3))
        keep += feet + hands[:2]
    return Bar(keep, bar.meter) if isinstance(bar, Bar) else keep


HAND_GAP = 0.115        # s: the soonest one hand strikes again (16ths on one hand up to about 130 bpm)
KICK_GAP = 0.10         # s: the soonest one kick follows another (foot to foot on a double pedal)
FEET = (KICK, PEDAL)


def _playable(bars, s16, gap=HAND_GAP):
    """The bars with one note per drum per step (the loudest), at most two hands at once, no hand striking again
    sooner than gap s and no kick sooner than KICK_GAP after the last. Walks the song in time (across the bar
    lines) with both hands' last strokes, gives each stroke the hand that struck last of those free for it (that
    leaves the other one the most room), and drops the least important notes no hand is free for. A flam (a grace
    note tied to its main stroke) takes both hands, so its beat keeps only its drum; without both hands free it
    is a plain stroke."""
    best, at = {}, 0
    for i, b in enumerate(bars):
        for x in b:     # (song step, bar, note); a grace note's key is its main stroke's
            k = (round(at + x[0], 3), x[1], round(at + x[3], 3) if len(x) > 3 else None)
            if k not in best or x[2] > best[k][2][2]:
                best[k] = (at + x[0], i, x)
        at += bar_steps(b)
    xs = sorted(best.values(), key=lambda e: (e[0], -HAND_PRIORITY.get(e[2][1], 3)))
    graces = {(k[2], k[1]): best[k] for k in best if k[2] is not None}
    keep, kicked = [], -9.0
    for e in xs:
        if e[2][1] == PEDAL:
            keep.append(e)
        elif e[2][1] == KICK and e[0] * s16 - kicked >= KICK_GAP - 1e-9:
            keep.append(e)
            kicked = e[0] * s16
    sl = []
    for e in xs:
        if e[2][1] in FEET or len(e[2]) > 3:
            continue
        if sl and (e[0] - sl[-1][0][0]) * s16 < 0.03:
            sl[-1].append(e)
        else:
            sl.append([e])
    last = [-9.0, -9.0]                     # the hands' last strokes (s), earlier first
    free = lambda t, h: t - last[h] >= gap - 1e-9
    for s in sl:
        s.sort(key=lambda e: -HAND_PRIORITY.get(e[2][1], 3))
        t = min(e[0] for e in s) * s16
        main = next((e for e in s if (round(e[0], 3), e[2][1]) in graces), None)
        if main is not None:
            g = graces[(round(main[0], 3), main[2][1])]
            if free(g[0] * s16, 1):
                keep += [g, main]
                last = [g[0] * s16, t]
                continue
            s = [main] + [e for e in s if e is not main]
        if len(s) > 1 and free(t, 1):
            keep += s[:2]
            last = [t, t]
        elif free(t, 1):
            keep.append(s[0])
            last = [last[0], t]
        elif free(t, 0):
            keep.append(s[0])
            last = [last[1], t]
    out = [Bar([], meter(b)) for b in bars]
    for _, i, x in sorted(keep, key=lambda e: (e[1], e[0])):
        out[i].append(x)
    return out


def _flam(st, pitch, v, s16, grace=46):
    """A flam on one drum: the main stroke at st and a soft grace note 30 ms before it, tied to it."""
    return [(st, pitch, v), (st - round(0.03 / s16, 2), pitch, grace, st)]


def _section(plan, n44):
    """Meters for a section about n44 4/4 bars long: the plan's meters in turn, ending on a whole turn when that
    stays within half a bar of the length."""
    out, total = [], 0
    while total < n44 * 16 - 8 or (len(out) % len(plan) and total < n44 * 16 + 8):
        out.append(plan[len(out) % len(plan)])
        total += bar_steps(out[-1])
    return out


def _lanes(stages, f0, n, rate):
    """[(step, tom)] from f0 to the bar's end every `rate` steps, hand to hand: the right hand's notes on the right
    lane, the left's on the left, both moving through `stages` [(left, right), ...] (high toms to low). The right
    lane always lies right of the left one, so the arms never cross, and the floor toms only take the right hand."""
    out, k, st = [], 0, float(f0)
    total = max(1, int(np.ceil((n - f0) / rate - 1e-6)))
    while st < n - 1e-6:
        left, right = stages[min(len(stages) - 1, k * len(stages) // total)]
        out.append((round(st, 3), right if k % 2 == 0 else left))
        st += rate
        k += 1
    return out


TOOL_PLANS = {"7/8": [(7, 8)], "5/4": [(5, 4)], "7/4": [(7, 4)], "7-7-7-4": [(7, 8)] * 3 + [(4, 4)],
              "5/8-7/8": [(5, 8), (7, 8)], "6/8-4/4": [(6, 8), (6, 8), (4, 4)], "4/4": [(4, 4)]}
TOOL_LANES = [(SNARE, 48), (48, 47), (48, 45), (SNARE, 43), (48, 41)]   # (left, right): the left keeps to its side
# tom ostinato cells for a 4- or 6-step beat: F floor accent with the kick, f floor, t and T rack tom (T accented)
TOOL_CELLS = {4: [[(0, "F"), (2, "f"), (3, "t")], [(0, "F"), (1, "t"), (2, "f")], [(0, "F"), (3, "t")],
                  [(0, "T"), (2, "f"), (3, "t")], [(0, "F"), (2, "t"), (3, "f")]],
              6: [[(0, "F"), (2, "f"), (4, "t")], [(0, "F"), (3, "t"), (4, "f")],
                  [(0, "F"), (2, "t"), (4, "f"), (5, "t")], [(0, "T"), (2, "f"), (3, "t"), (4, "f")]]}


def tool_bars(seed=0, bpm=118):
    """A seeded song in Danny Carey's idiom with Tool, for the big kit (six toms, double pedal, china, splash; on a
    smaller kit the extra notes fold onto its pieces). Original: built from idioms, not taken from any record.
    The meters are odd and shifting: a home plan (7/8, 5/4, 7/4, three 7/8s and a 4/4, or 5/8 + 7/8) and an away
    plan (another of those, 6/8 against 4/4, or 4/4 under a grouping). The song: a quiet intro (soft closed hat
    with side stick or ghosted snare, or a tom ostinato under the hi-hat foot); a build that swells into a crash;
    a groove with ghosted snare where accents grouped in 3s or 5s (china or crash stabs with the kick) run across
    the bar lines over a steady hat or ride; a tribal section, a floor-tom ostinato under the kick with a rack tom
    and the snare answering and flams on the snare; sometimes unison stabs into a stop (the teacher chokes the
    china); a climax of double-kick 16ths under china stabs or grouped unison hits; a big fill round the toms
    (16ths hand to hand, 8th triplets, hand-hand-kick threes, or snare flams into the floors); the last hit. The
    left hand keeps to the snare and the middle rack tom (48), the right takes the toms to its side and the floors:
    on the big kit the 8" tom sits behind the hi-hat, where the left fist catches the hats on the way back to the
    snare. Every bar goes through _playable (two hands, HAND_GAP, KICK_GAP), so a human could play it. Meant for
    95-135 bpm."""
    rng = np.random.default_rng([seed, 46])
    s16 = 15.0 / bpm
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    chance = lambda p: bool(rng.random() < p)
    vel = lambda v, d=1.0: int(np.clip(25 + (v - 25) * d + rng.normal(0, 4), 25, 127))
    home = pick(["7/8", "5/4", "7/4", "7-7-7-4", "5/8-7/8"])
    away = pick([p for p in TOOL_PLANS if p != home])
    hat16 = s16 >= 0.12

    def backbeat(m, mode, late=False):
        n, B = bar_steps(m), beat_steps(m)
        if mode == "half":
            return [B[len(B) // 2]] if len(B) > 1 else [n // 2]
        hits = B[1::2] or [n // 2]
        return hits + [n - 2] if late and m[1] == 8 and m[0] % 2 else hits

    def kick_pattern(m, snares):
        n = bar_steps(m)
        cands = [s for s in range(2, n, 2) if s not in snares]
        ks = {0} | {int(x) for x in rng.choice(cands, min(len(cands), int(rng.integers(1, 4))), replace=False)}
        if chance(0.4):
            ks.add(int(pick([s for s in range(1, n, 2) if s + 1 not in snares and s - 1 not in snares] or [n - 1])))
        return sorted(ks)

    def groove_opts(keep=None, quiet=False):
        keep = keep or pick([HAT, HAT, RIDE, BELL])
        o = {"keep": keep, "rate": pick([1, 2]) if hat16 and keep == HAT else (0 if keep == BELL and quiet else 2),
             "back": pick(["back", "back", "half"]), "late": chance(0.5), "ghosts": chance(0.6), "kicks": {},
             "snare": SIDE if quiet and chance(0.5) else SNARE, "group": 0, "lift": keep == HAT and chance(0.4),
             "stab": pick([CHINA, CHINA, CRASH2]) if keep in (RIDE, BELL) else pick([CRASH, SPLASH]),
             "pedal": keep != HAT and chance(0.7)}
        o["ghosts"] = o["ghosts"] and o["snare"] == SNARE
        return o

    def groove_bar(m, d, at, o):
        """timekeeper, snare, kicks, ghosts and accents grouped across the bar line (at: the bar's first step in
        its section, for the grouping's phase)"""
        n, B = bar_steps(m), beat_steps(m)
        acc = [s for s in range(n) if o["group"] and (at + s) % o["group"] == 0]
        sn = backbeat(m, "half" if acc else o["back"], o["late"])
        bar = []
        for s in (B if o["rate"] == 0 else range(0, n, o["rate"])):
            if any(abs(s - a) <= 1 for a in acc):
                continue
            on = s in B
            pc = (BELL if on else RIDE) if o["keep"] == BELL else o["keep"]
            if pc == HAT and o["lift"] and s == n - 2:
                pc = OPEN
            bar.append((s, pc, vel((94 if on else 68) + (14 if pc == OPEN else 0), d)))
        for a in acc:
            bar += [(a, o["stab"], vel(116, d)), (a, KICK, vel(112, d))]
        bar += [(s, o["snare"], vel(112, d)) for s in sn]
        if m not in o["kicks"]:
            o["kicks"][m] = kick_pattern(m, sn)
        ks = o["kicks"][m]
        bar += [(k, KICK, vel(108 if k in B else 92, d)) for k in ks if k == 0 or not chance(0.15)]
        if o["ghosts"]:
            free = [s for s in range(1, n, 2) if all(abs(s - h) > 1 for h in sn + acc)]
            for s in rng.choice(free, min(len(free), int(rng.integers(1, 4))), replace=False) if free else []:
                bar.append((int(s), SNARE, vel(36)))
        if o["pedal"]:
            bar += [(b, PEDAL, vel(68, max(d, 0.6))) for b in (B[1::2] or B)]
        return bar

    def tribal_opts():
        return {"toms": pick([(48, SNARE), (SNARE, 48)]), "low": pick([FLOOR, FLOOR, 41]),
                "cells": {L: pick(TOOL_CELLS[L]) for L in TOOL_CELLS}, "flams": chance(0.7), "pedal": chance(0.8)}

    def tribal_bar(m, d, o, crash=False, kicks=True):
        """floor-tom ostinato with the kick, the rack tom and snare answering, flams on the snare"""
        n, B = bar_steps(m), beat_steps(m)
        groups = list(zip(B, B[1:] + [n]))
        hi, lo = o["toms"]
        call = int(rng.integers(1, len(groups))) if len(groups) > 1 else 0
        bar = [(0, CRASH, vel(118, d))] if crash else []
        for gi, (b0, b1) in enumerate(groups):
            L = b1 - b0
            if gi == call:      # the snare calls, the toms answer
                cell = [(0, SNARE, 116), (1, hi, 100), (2, lo, 104)] + ([(4, FLOOR, 108)] if L == 6 else [])
                bar += [(b0 + s, p, vel(v, d)) for s, p, v in cell]
                continue
            for s, sym in o["cells"][L if L in TOOL_CELLS else 4]:
                if b0 + s >= b1:
                    continue
                st = b0 + s
                if sym == "F":
                    bar.append((st, o["low"] if s == 0 and gi == 0 else FLOOR, vel(112, d)))
                    bar += [(st, KICK, vel(108, d))] if kicks or st == 0 else []
                elif sym == "f":
                    bar.append((st, FLOOR, vel(80, d)))
                elif sym == "t":
                    bar.append((st, lo, vel(86, d)))
                elif o["flams"] and not (crash and st == 0) and chance(0.6):
                    bar += [x[:2] + (vel(x[2], d),) + x[3:] for x in _flam(st, SNARE, 112, s16)]
                else:
                    bar.append((st, hi, vel(110, d)))
        if o["pedal"]:
            bar += [(b, PEDAL, vel(66, max(d, 0.6))) for b in B]
        return bar

    def climax_opts():
        return {"kick": pick(["run", "run", "burst"]), "hands": pick(["stab", "poly", "bell"]),
                "group": pick([3, 5, 3, 6]), "stab": pick([CHINA, CHINA, CRASH2])}

    def climax_bar(m, d, at, o, crash=False):
        """double-kick 16ths (or gallops) under china stabs, grouped unison hits or a ride bell"""
        n, B = bar_steps(m), beat_steps(m)
        groups = list(zip(B, B[1:] + [n]))
        if o["kick"] == "run":
            ks = list(range(n))
        else:
            ks = [s for b0, b1 in groups for s in [b0] + list(range(b1 - (3 if b1 - b0 == 6 else 2), b1))]
        bar = [(s, KICK, vel(110 if s in B else 96, d)) for s in ks]
        if o["hands"] == "poly":
            for s in range(n):
                if (at + s) % o["group"] == 0:
                    bar += [(s, o["stab"], vel(118, d)), (s, SNARE, vel(116, d))]
        else:
            pc = o["stab"] if o["hands"] == "stab" else BELL
            bar += [(b, pc, vel(116 if pc != BELL else 104, d)) for b in B]
            if o["hands"] == "bell":
                bar += [(b + 2, RIDE, vel(84, d)) for b0, b1 in groups for b in range(b0, b1 - 2, 2) if b + 2 < b1]
            bar += [(s, SNARE, vel(116, d)) for s in backbeat(m, "back")]
        if crash:
            bar.append((0, CRASH, vel(122, d)))
        return bar

    def fill_bar(m, d, kind, f0, pre):
        """a big fill round the six toms from step f0, the bar's groove (pre) before it"""
        n, B = bar_steps(m), beat_steps(m)
        bar = [x for x in pre if x[0] < f0 - 0.5]
        span = max(1, n - f0)
        cresc = lambda st: vel(100 + 24 * (st - f0) / span, d)
        if kind == "threes":    # right, left, kick in 16ths: the accents fall in 3s across the beat
            ng = int(np.ceil(span / 3))
            for j, st in enumerate(range(f0, n, 3)):
                left, right = TOOL_LANES[min(len(TOOL_LANES) - 1, j * len(TOOL_LANES) // ng)]
                bar += [(st, right, cresc(st) + 6), (st + 1, left, cresc(st + 1) - 8)]
                bar += [(st + 2, KICK, vel(104, d))] if st + 2 < n else []
        elif kind == "flams":   # 8th-note flams on the snare, then the floors hand to hand
            last = B[-1] if B[-1] > f0 else n
            bar += [x for st in range(f0, last, 2) for x in _flam(st, SNARE, cresc(st), s16)]
            bar += [(st, p, cresc(st)) for st, p in _lanes(TOOL_LANES[-2:], last, n, 1)]
            bar += [(b, KICK, vel(106, d)) for b in B if b >= f0]
        else:
            rate = 4 / 3 if kind == "trip" else 1
            bar += [(st, p, cresc(st)) for st, p in _lanes(TOOL_LANES[int(rng.integers(0, 2)):], f0, n, rate)]
            bar += [(b, KICK, vel(106, d)) for b in B if b >= f0]
        return bar

    def stabs_bar(m, d):
        """unison stabs into a stop: china, snare and kick, then silence the teacher chokes the china in"""
        n, B = bar_steps(m), beat_steps(m)
        rhythm = pick([[0, 3, 6], [0, 3], [0, 2, 5], [0, 3, 6, 8]])
        rhythm = [s for s in rhythm if s <= n - 6]
        return [x for s in rhythm for x in ((s, CHINA, vel(120, d)), (s, SNARE, vel(114, d)), (s, KICK, vel(116, d)))]

    out = []

    add = out.extend

    # intro: quiet, in the home meter
    ms = _section(TOOL_PLANS[home], 2)
    if chance(0.5):
        o = groove_opts(pick([HAT, HAT, BELL]), quiet=True)
        o["kicks"] = {m: [0] for m in ms}
        add([Bar(groove_bar(m, 0.35 + 0.15 * i / len(ms), 0, o), m) for i, m in enumerate(ms)])
    else:
        o = tribal_opts()
        o["flams"] = False
        add([Bar(tribal_bar(m, 0.35 + 0.15 * i / len(ms), o, kicks=False), m) for i, m in enumerate(ms)])
    # build: the groove swells, a snare pickup into the crash
    ms = _section(TOOL_PLANS[home], 2)
    o = groove_opts()
    bs = [groove_bar(m, 0.5 + 0.4 * (i + 1) / len(ms), 0, o) for i, m in enumerate(ms)]
    m = ms[-1]
    f0 = beat_steps(m)[-1]
    bs[-1] = [x for x in bs[-1] if x[0] < f0 or x[1] in FEET] + \
        [(s, SNARE, vel(70 + 50 * (s - f0) / max(1, bar_steps(m) - f0))) for s in range(f0, bar_steps(m))]
    add([Bar(b, m) for b, m in zip(bs, ms)])
    # groove and tribal, in either order
    parts = ["groove", "tribal"] if chance(0.5) else ["tribal", "groove"]
    for part in parts:
        if part == "groove":
            ms = _section(TOOL_PLANS[away], 4)
            o = groove_opts(pick([HAT, RIDE, RIDE, BELL]))
            o["group"] = pick([3, 5, 6, 0]) if away != "6/8-4/4" else 0
            o["pedal"] = o["pedal"] or (o["keep"] != HAT and o["group"] > 0)
            at, bs = 0, []
            for i, m in enumerate(ms):
                b = groove_bar(m, 0.9, at, o)
                if i == 0:
                    b = [x for x in b if x[0] != 0 or x[1] in FEET or x[1] == SNARE] + [(0, CRASH, vel(122))]
                bs.append(Bar(b, m))
                at += bar_steps(m)
            add(bs)
        else:
            ms = _section(TOOL_PLANS[home], 3)
            o = tribal_opts()
            add([Bar(tribal_bar(m, 0.8 + 0.2 * i / max(1, len(ms) - 1), o, crash=i == 0), m)
                 for i, m in enumerate(ms)])
    if chance(0.5):
        m = TOOL_PLANS[home][0]
        add([Bar(stabs_bar(m, 1.0), m)])
    # climax: double kick under the stabs, then the big fill and the last hit
    cplan = pick([away, home, "4/4"])
    ms = _section(TOOL_PLANS[cplan], 2)
    o = climax_opts()
    at, bs = 0, []
    for i, m in enumerate(ms):
        bs.append(Bar(climax_bar(m, 1.0, at, o, crash=i == 0), m))
        at += bar_steps(m)
    add(bs)
    m = TOOL_PLANS[home][-1]
    B = beat_steps(m)
    f0 = pick([0, B[len(B) // 2], B[1]]) if bar_steps(m) <= 16 else B[len(B) // 2]
    kind = pick(["down", "down", "trip", "threes", "flams"])
    add([Bar(fill_bar(m, 1.0, kind, f0, climax_bar(m, 1.0, at, o)), m)])
    out.append(Bar([(0, CRASH, vel(127)), (0, CHINA, vel(124)), (0, KICK, vel(127))], m))
    return _playable(out, s16)


BAKER_LANES = [(SNARE, SNARE), (48, 47), (48, 43), (48, 41)]     # (left, right): the left keeps to its own side


def baker_bars(seed=0, bpm=140):
    """A seeded song in Ginger Baker's idiom, for a kit with two bass drums (drum_kit "double_pedal": kicks closer
    than drums.DOUBLE_IOI go foot to foot). Original: built from idioms, not taken from any record. Sections: an
    intro (a triplet tom roll that swells, or the tom groove played softly); the tom-tom "jungle" groove, the floor
    tom carrying the beat in place of a hi-hat with a rack tom or the snare on 2 and 4 and the two bass drums under
    it (16th or triplet pairs, running triplets against the hands' 8ths, or 8ths); then one or two of a swung jazz
    ride with feathered kick, the hi-hat foot on 2 and 4 and snare comping on the triplets (4/4 or 5/4), a 12/8
    African bell pattern on the ride bell with the rack tom and snare talking under it over a four-beat kick, or
    the tom groove in 6/8; triplet tom rolls (hand to hand or hand-hand-kick) into sparse crashes; a solo break of
    paradiddles and singles round the snare and toms over a two-bass ostinato; a last hit. The left hand keeps to
    the snare and the near rack tom. Every bar goes through _playable (two hands, HAND_GAP, KICK_GAP). Meant for
    100-170 bpm."""
    rng = np.random.default_rng([seed, 39])
    s16 = 15.0 / bpm
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    chance = lambda p: bool(rng.random() < p)
    vel = lambda v, d=1.0: int(np.clip(25 + (v - 25) * d + rng.normal(0, 4), 25, 127))
    pair = 1 if s16 >= KICK_GAP else 4 / 3          # the two bass drums' quick pair: a 16th, or a triplet at speed

    def kick_steps(m, fig):
        n, B = bar_steps(m), beat_steps(m)
        if fig == "pairs":
            return [s for b in B[::2] for s in (b, b + pair)] + [b for b in B[1::2]]
        if fig == "trips":
            return [b + k * (B[1] - B[0] if len(B) > 1 and B[1] - B[0] == 6 else 4) / 3 for b in B for k in range(3)]
        if fig == "eighths":
            return list(range(0, n, 2))
        return list(B) + [n - pair]                 # four on the floor and a pickup into the next bar

    def jungle_opts():
        return {"off": pick(["floor", "tom", "tom", "skip"]), "acc": pick([48, SNARE, 48]), "low": pick([FLOOR, 41]),
                "kfig": pick(["pairs", "trips", "pairs", "eighths", "four"]), "pick": chance(0.5)}

    def jungle_bar(m, d, o, crash=False, answer=False):
        """the floor tom on the beat, a rack tom (or the snare) on 2 and 4, two bass drums under it"""
        n, B = bar_steps(m), beat_steps(m)
        groups = list(zip(B, B[1:] + [n]))
        bar = [(0, CRASH, vel(120, d))] if crash else []
        for i, (b0, b1) in enumerate(groups):
            bar.append((b0, o["low"] if i == 0 else FLOOR, vel(112, d)))
            if i % 2:
                bar.append((b0, o["acc"], vel(116, d)))
            offs = [b0 + 8 / 3] if o["off"] == "skip" and b1 - b0 == 4 else list(range(b0 + 2, b1, 2))
            if answer and i == len(groups) - 1:     # snare and rack tom answer on the last beat, in triplets
                bar += [(b0 + (b1 - b0) / 3, SNARE, vel(92, d)), (b0 + 2 * (b1 - b0) / 3, 48, vel(98, d))]
                continue
            for s in offs:                          # the right hand's 8ths on the floor, or the left's on the rack tom
                bar.append((s, 48 if o["off"] == "tom" else FLOOR, vel(78, d)))
            if o["pick"] and o["off"] != "tom" and b1 < n and b1 - 1 not in offs and chance(0.5):
                bar.append((b1 - 1, 48, vel(82, d)))
        bar += [(s, KICK, vel(108 if s in B else 94, d)) for s in kick_steps(m, o["kfig"]) if s < n]
        return bar

    def swing_bar(m, d, o, crash=False):
        """jazz ride, ding ding-a ding; the bass drum feathered on every beat, the hi-hat foot on 2 and 4"""
        n, B = bar_steps(m), beat_steps(m)
        bar = [(0, CRASH, vel(112, d))] if crash else []
        for i, b in enumerate(B):
            bar += [(b, RIDE, vel(88 if i % 2 else 80, d)), (b, KICK, vel(44))]
            if i % 2:
                bar += [(b + 8 / 3, RIDE, vel(66, d)), (b, PEDAL, vel(72, d))]
        for _ in range(int(rng.integers(1, 4))):    # snare comping on the triplets
            b = int(pick(B))
            bar.append((b + pick([4 / 3, 8 / 3]), SNARE, vel(pick([40, 46, 64, 76]), d)))
        if chance(o["bombs"]):                      # a bomb: the bass drum speaks on a skip note
            bar.append((int(pick(B[:-1])) + 8 / 3, KICK, vel(104, d)))
        return bar

    def african_bar(m, d, o, crash=False):
        """12/8 bell pattern on the ride bell, rack tom and snare talking in the gaps, the kick on the dotted beats"""
        n = bar_steps(m)
        bell = [0, 2, 4, 5, 7, 9, 11]
        bar = [(0, CRASH, vel(116, d))] if crash else []
        bar += [(2 * e, BELL, vel(100 if e == 0 else 86, d)) for e in bell if 2 * e < n]
        for e in o["talk"]:
            if 2 * e < n:
                bar.append((2 * e, SNARE if e == 6 else o["toms"][e % 2], vel(110 if e == 6 else 84, d)))
        bar += [(s, KICK, vel(104, d)) for s in range(0, n, 6)]
        bar += [(s - 2, KICK, vel(86, d)) for s in range(6, n, 6) if chance(o["pickups"])]
        bar += [(s, PEDAL, vel(70, d)) for s in range(6, n, 12)]
        return bar

    def groove(kind, m, d, o, crash=False):
        if kind == "jungle":
            return jungle_bar(m, d, o, crash)
        if kind == "swing":
            return swing_bar(m, d, o, crash)
        return african_bar(m, d, o, crash)

    def opts(kind):
        if kind == "jungle":
            return jungle_opts()
        if kind == "swing":
            return {"bombs": pick([0.2, 0.4])}
        return {"talk": sorted(set(int(x) for x in rng.choice([1, 3, 6, 8, 10], int(rng.integers(3, 6)),
                                                              replace=False)) | {6}),
                "toms": pick([(48, SNARE), (SNARE, 48)]), "pickups": pick([0.0, 0.3, 0.6])}

    def fill(m, d, f0, pre, kind):
        """a tom roll from f0: 8th triplets (8ths in 12/8) hand to hand, or hand-hand-kick, or 16ths"""
        n, B = bar_steps(m), beat_steps(m)
        bar = [x for x in pre if x[0] < f0 - 0.5]
        compound = len(B) > 1 and B[1] - B[0] == 6
        rate = 1 if kind == "16ths" else (2 if compound else 4 / 3)
        cresc = lambda st: vel(96 + 26 * (st - f0) / max(1, n - f0), d)
        if kind == "hhf":
            k, st = 0, float(f0)
            while st < n - 1e-6:
                left, right = BAKER_LANES[min(3, int(4 * (st - f0) / max(1, n - f0)))]
                bar.append((round(st, 3), KICK, vel(100, d)) if k % 3 == 2 else
                           (round(st, 3), right if k % 3 == 0 else left, cresc(st)))
                st += rate
                k += 1
        else:
            bar += [(st, p, cresc(st)) for st, p in _lanes(BAKER_LANES, f0, n, rate)]
            bar += [(b, KICK, vel(104, d)) for b in B if b >= f0]
        return bar

    def solo_bar(m, d, o):
        """paradiddles or singles round the snare and toms, accents across the beat, the two bass drums under it"""
        n, B = bar_steps(m), beat_steps(m)
        bar, k, st = [], 0, 0.0
        while st < n - 1e-6:
            hand = o["stick"][k % len(o["stick"])]
            side = o["R"] if hand == "R" else o["L"]
            bar.append((round(st, 3), side[(k // len(o["stick"])) % 2], vel(118 if k % o["acc"] == 0 else 86, d)))
            st += o["rate"]
            k += 1
        bar += [(s, KICK, vel(104, d)) for s in kick_steps(m, o["kfig"]) if s < n]
        return bar

    out = []

    add = out.extend

    a_kind = "jungle"
    b_kinds = [pick(["swing", "african", "six"])]
    if chance(0.5):
        b_kinds.append(pick([k for k in ("swing", "african", "jungle") if k != b_kinds[0]]))
    # intro
    oa = jungle_opts()
    if chance(0.5):
        m = (4, 4)
        add([Bar(fill(m, 0.6, 0, [], pick(["trip", "hhf"])), m)])
    else:
        add([Bar(jungle_bar((4, 4), 0.45 + 0.15 * i, dict(oa, kfig="four")), (4, 4)) for i in range(2)])
    # A: the jungle groove
    ms = [(4, 4)] * 4
    add([Bar(jungle_bar(m, 0.85 + 0.05 * (i % 2), oa, crash=i == 0 and chance(0.7), answer=i % 2 == 1), m)
         for i, m in enumerate(ms)])
    # B (and C): swing, 12/8 African, 6/8 toms, or the jungle again, louder; a tom roll ends each
    for j, kind in enumerate(b_kinds):
        if kind == "six":
            kind, ms, o = "jungle", _section([(6, 8)], 4), jungle_opts()
        elif kind == "african":
            ms, o = _section([(12, 8)], 4), opts(kind)
        elif kind == "swing":
            ms, o = _section([pick([(4, 4), (4, 4), (5, 4)])], 4), opts(kind)
        else:
            ms, o = [(4, 4)] * 3, dict(oa, kfig=pick(["trips", "pairs"]))
        d = {"swing": 0.7, "african": 0.85, "jungle": 0.95}[kind]
        bs = [groove(kind, m, d, o, crash=i == 0 and (j == 0 or chance(0.5))) for i, m in enumerate(ms)]
        m = ms[-1]
        B = beat_steps(m)
        f0 = B[len(B) // 2] if bar_steps(m) > 12 else B[-1]
        bs[-1] = fill(m, 1.0, f0, bs[-1], pick(["trip", "trip", "hhf", "16ths"]))
        add([Bar(b, m) for b, m in zip(bs, ms)])
    # solo break over the two bass drums, then the last hit
    for i in range(2):
        m = (4, 4)
        trip = chance(0.5) or s16 < HAND_GAP
        o = {"rate": 4 / 3 if trip else 1, "acc": pick([3, 4, 6]) if trip else pick([3, 4, 5]),
             "stick": pick(["RLRRLRLL", "RL", "RLRRLL"]) if (4 / 3 if trip else 1) * s16 >= HAND_GAP else "RL",
             "R": pick([(FLOOR, 47), (FLOOR, 41), (47, FLOOR)]), "L": pick([(SNARE, 48), (48, SNARE)]),
             "kfig": pick(["pairs", "trips", "four"])}
        add([Bar(solo_bar(m, 0.9 + 0.1 * i, o), m)])
    out.append(Bar([(0, CRASH, vel(127)), (0, FLOOR, vel(120)), (0, KICK, vel(127))], (4, 4)))
    return _playable(out, s16)


SONGS = {"rock": bars, "technique": technique_bars, "showcase": showcase_bars, "toss": toss_bars,
         "random": random_bars, "tool": tool_bars, "baker": baker_bars}
GENERATED = {"random": random_bars, "tool": tool_bars, "baker": baker_bars}     # (seed, bpm) -> bars


def song_bars(song, seed=0, bpm=112):
    return GENERATED[song](seed, bpm) if song in GENERATED else SONGS[song]()


def notes(bpm=112, seed=0, song="rock", bars=None):
    """[(start_s, pitch, vel)] for the whole groove (or for `bars`), the bars laid end to end by their meters."""
    rng = np.random.default_rng(seed)
    step = 60.0 / bpm / 4
    out, at = [], 0
    for b, bar in enumerate(song_bars(song, seed, bpm) if bars is None else bars):
        drawn = []
        for x in bar:
            jitter = 0.0 if (x[0] == 0 and b == 0) else float(rng.normal(0, 0.004))
            drawn.append((x, jitter, int(np.clip(x[2] + rng.normal(0, 4), 20, 127))))
        main = {x[:2]: j for x, j, _ in drawn}
        for x, j, v in drawn:
            if len(x) > 3:      # (step, pitch, vel, main step): a grace note keeps its gap to its main stroke
                j = main.get((x[3], x[1]), j)
            out.append((max(0.0, (at + x[0]) * step + j), x[1], v))
        at += bar_steps(bar)
    return sorted(out)


def _vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F)); v >>= 7
    return bytes(reversed(out))


def _time_sig(num, den):
    clocks = 24 * 4 // den * (3 if den == 8 and num % 3 == 0 and num > 3 else 1)   # MIDI clocks per beat
    return b"\xff\x58\x04" + bytes([num, den.bit_length() - 1, clocks, 8])


def write(path, ns, bpm, meters=None):
    """meters: meter_map() of the bars, a time signature wherever the meter changes (default one 4/4)."""
    tick = lambda s: int(round(s * bpm / 60.0 * DIV))
    ev = [(0, 0, b"\xff\x51\x03" + int(round(60e6 / bpm)).to_bytes(3, "big"))]
    ev += [(int(round(st * DIV / 4)), 0, _time_sig(num, den)) for st, num, den in (meters or [(0, 4, 4)])]
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
    bs = song_bars(a.song, a.seed, a.bpm)
    ns = notes(a.bpm, a.seed, bars=bs)
    write(a.out, ns, a.bpm, meter_map(bs))
    print(f"{len(ns)} hits, {seconds(bs, a.bpm):.1f} s -> {a.out}")

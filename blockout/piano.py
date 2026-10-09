"""Piano blockout kinematics: MIDI notes -> fingering -> per-frame hand/finger joints and key depression.

Plain numpy, no Blender. The output JSON is everything the renderer needs (keyboard layout, camera, and per
frame: key depths, and for each hand the elbow, wrist, palm corners and 5 finger chains), so the renderer
stays dumb and the motion rules live here, testable.

Frame of reference (metres): x along the keyboard (middle C at x=0, pitch rises with x), y away from the
player (white key fronts at y=0), z up (white key tops at z=0).

Rules, in order of application:
1. Hand split: MIDI track 1 = right hand, track 2 = left hand (Mutopia convention); else split by pitch.
2. Fingering: Viterbi over onset slices with a Parncutt-style cost (finger-pair stretch tables, thumb/pinky
   on black keys, same finger on a new key, crossings, hand shifts weighted by how little time there is).
3. Hand x: per slice, the least-squares placement of the five-finger "home" over the struck keys;
   it glides to the next slice's placement shortly before that slice sounds.
4. Fingers, per note touch (0 caress .. 1 strike, from feeling.py): soft notes start on the key and press
   slowly (ease-out); strong notes pull back up to ~4 cm and accelerate in. Hold, then release to a hover
   whose height also follows touch. Strong chords add arm weight (hand rises, drops through the onset);
   in rests the hand floats up and "breathes". Tips are placed by 3-link planar IK from the knuckle.
5. Springs (dynamics.py): every target above (hand centre, hand roll toward the playing fingers, each fingertip,
   the head) is fed through a damped second-order system read slightly ahead in time, so hands anticipate,
   overshoot and settle and fingers trail and catch up, instead of easing A-to-B and stopping dead. A slow
   seeded drift keeps the wrists and head alive. Keys are pushed by the sprung fingertips: a key goes down
   only as far as a finger actually presses it.
"""
import itertools, json, math

import numpy as np

from . import dynamics, performer

# ---------------------------------------------------------------- keyboard
WHITE_W, WHITE_L = 0.0235, 0.150
BLACK_W, BLACK_L, BLACK_H = 0.0137, 0.095, 0.012
KEY_TRAVEL = 0.010
LOW, HIGH = 21, 108
_BLACK_PC = {1, 3, 6, 8, 10}


def is_black(p):
    return p % 12 in _BLACK_PC


def _white_index(p):
    return sum(1 for q in range(LOW, p) if not is_black(q))


_MID_C = _white_index(60) * WHITE_W + WHITE_W / 2


def key_x(p):
    if not is_black(p):
        return _white_index(p) * WHITE_W + WHITE_W / 2 - _MID_C
    return _white_index(p) * WHITE_W - _MID_C  # on the boundary between its white neighbours


def keyboard():
    keys = []
    for p in range(LOW, HIGH + 1):
        b = is_black(p)
        keys.append({"pitch": p, "black": b, "x": key_x(p), "w": BLACK_W if b else WHITE_W,
                     "l": BLACK_L if b else WHITE_L, "y0": WHITE_L - BLACK_L if b else 0.0,
                     "top": BLACK_H if b else 0.0})
    return keys


def contact(p, near_black):
    """Where a fingertip presses the key: black keys mid-length, white keys near the front unless the
    hand is up among the black keys."""
    if is_black(p):
        return np.array([key_x(p), WHITE_L - BLACK_L + 0.045, BLACK_H])
    return np.array([key_x(p), 0.075 if near_black else 0.035, 0.0])


# ---------------------------------------------------------------- hand model
# finger 0 = thumb ... 4 = pinky. Home tip offsets: a five-finger position over adjacent white keys.
HOME = np.array([-2, -1, 0, 1, 2]) * WHITE_W
SEG = np.array([[0.040, 0.032, 0.026],   # thumb: metacarpal, proximal, distal
                [0.044, 0.025, 0.019],
                [0.048, 0.028, 0.020],
                [0.045, 0.027, 0.020],
                [0.036, 0.020, 0.018]])
# knuckle (MCP; thumb CMC) offsets from the hand centre at the knuckle line, right hand
KNUCKLE = np.array([[-0.034, -0.055, -0.020],
                    [-0.027, 0.000, 0.0],
                    [-0.008, 0.004, 0.0],
                    [0.011, 0.000, 0.0],
                    [0.029, -0.010, -0.004]])
KNUCKLE_Z = 0.060          # knuckle line height over white key tops
PALM_LEN = 0.085           # knuckle line -> wrist

# Parncutt-style finger-pair spans in semitones (right hand, finger i < j, signed pitch_j - pitch_i):
# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac)
SPAN = {(0, 1): (-5, -3, 1, 5, 8, 10), (0, 2): (-4, -2, 3, 7, 10, 12), (0, 3): (-3, -1, 5, 9, 12, 14),
        (0, 4): (-1, 1, 7, 10, 13, 15), (1, 2): (1, 1, 1, 2, 3, 5), (1, 3): (1, 1, 3, 4, 5, 7),
        (1, 4): (2, 2, 5, 6, 8, 10), (2, 3): (1, 1, 1, 2, 2, 4), (2, 4): (1, 1, 3, 4, 5, 7),
        (3, 4): (1, 1, 1, 2, 3, 5)}


def split_hands(notes):
    tracks = {n.track for n in notes}
    if len(tracks) >= 2:
        hi, lo = sorted(tracks)[:2]
        return [n for n in notes if n.track == hi], [n for n in notes if n.track != hi]
    return [n for n in notes if n.pitch >= 60], [n for n in notes if n.pitch < 60]


def slices(notes, tol=0.01):
    out = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if out and n.start - out[-1][0].start < tol:
            out[-1].append(n)
        else:
            out.append([n])
    return [s[:5] for s in out]


def _span_cost(fi, pi, fj, pj, hand):
    """Stretch cost for finger fi on pitch pi and fj on pj (any order)."""
    if fi == fj:
        return 0.0
    if fi > fj:
        fi, pi, fj, pj = fj, pj, fi, pi
    d = (pj - pi) if hand == "R" else (pi - pj)
    mn_p, mn_c, mn_r, mx_r, mx_c, mx_p = SPAN[(fi, fj)]
    if d < mn_p or d > mx_p:
        return 50.0
    c = 0.0
    c += 2 * max(0, mn_c - d) + 2 * max(0, d - mx_c)
    c += max(0, mn_r - d) + max(0, d - mx_r)
    return c


def _place(assign, near_black, hand):
    xs = [contact(p, near_black)[0] - (HOME[f] if hand == "R" else -HOME[f]) for p, f in assign]
    return float(np.mean(xs))


def fingering(sl, hand):
    """Viterbi over slices. Returns, per slice, a list of (note, finger)."""
    cands = []
    for s in sl:
        ps = sorted(n.pitch for n in s)
        cs = []
        for fs in itertools.combinations(range(5), len(ps)):
            order = fs if hand == "R" else fs[::-1]
            a = list(zip(ps, order))
            c = sum(_span_cost(f1, p1, f2, p2, hand) for (p1, f1), (p2, f2) in itertools.combinations(a, 2))
            c += sum(2.0 for p, f in a if f == 0 and is_black(p)) + sum(1.0 for p, f in a if f == 4 and is_black(p))
            if c < 50:
                nb = any(is_black(p) for p in ps)
                cs.append((a, c, _place(a, nb, hand)))
        cands.append(cs or [([(p, 2) for p in ps[:1]], 0.0, key_x(ps[0]))])

    INF = float("inf")
    cost = [c for _, c, _ in cands[0]]
    back = []
    for i in range(1, len(sl)):
        dt = max(sl[i][0].start - sl[i - 1][0].start, 0.03)
        held = {n.pitch for n in sl[i - 1] if n.end > sl[i][0].start + 0.02}
        new_cost, bp = [], []
        for b, cb, hb in cands[i]:
            best, arg = INF, 0
            fb = {f for _, f in b}
            for k, (a, ca, ha) in enumerate(cands[i - 1]):
                if cost[k] == INF:
                    continue
                if any(f in fb and p in held and (p, f) not in b for p, f in a):
                    continue  # that finger is still holding a key
                t = 0.0
                for pa, fa in a:
                    for pb, fbb in b:
                        if fa == fbb and pa != pb:
                            t += 4.0 if dt < 0.3 else 1.0
                        elif fa != fbb:
                            sc = _span_cost(fa, pa, fbb, pb, hand)
                            t += 0.5 * sc if sc < 50 else 8.0  # a reach this wide means a hand shift
                            up = (pb - pa) if hand == "R" else (pa - pb)
                            if up > 0 and fbb < fa and fbb != 0:
                                t += 3.0  # non-thumb crossing
                            if up < 0 and fbb > fa and fa != 0:
                                t += 3.0
                t += abs(hb - ha) / WHITE_W * (0.3 + 0.05 / dt)
                if cost[k] + t < best:
                    best, arg = cost[k] + t, k
            new_cost.append(best + cb)
            bp.append(arg)
        cost, back = new_cost, back + [bp]
    k = int(np.argmin(cost))
    path = [k]
    for bp in reversed(back):
        k = bp[k]
        path.append(k)
    path.reverse()
    out = []
    for i, (s, k) in enumerate(zip(sl, path)):
        by_pitch = dict(cands[i][k][0])
        out.append([(n, by_pitch.get(n.pitch, 2)) for n in s])
    return out


# ---------------------------------------------------------------- motion
def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def hand_track(fing, hand, times, rest_x):
    """Hand-centre x (and near-black flag) over frame times."""
    if not fing:
        return np.full(len(times), rest_x), np.zeros(len(times), bool)
    t_on = np.array([s[0][0].start for s in fing])
    hx = np.array([_place([(n.pitch, f) for n, f in s], any(is_black(n.pitch) for n, _ in s), hand) for s in fing])
    nb = np.array([any(is_black(n.pitch) for n, _ in s) for s in fing])
    x = np.empty(len(times))
    for j, t in enumerate(times):
        i = np.searchsorted(t_on, t, side="right") - 1
        if i < 0:
            u = _smooth((t - (t_on[0] - 0.3)) / 0.3)
            x[j] = rest_x + (hx[0] - rest_x) * u
            continue
        x[j] = hx[i]
        if i + 1 < len(t_on):  # glide to the next placement just before it sounds
            gap = t_on[i + 1] - t_on[i]
            dur = min(0.12, 0.7 * gap)
            t1 = t_on[i + 1] - 0.025
            x[j] = hx[i] + (hx[i + 1] - hx[i]) * _smooth((t - (t1 - dur)) / dur)
    idx = np.clip(np.searchsorted(t_on, times + 0.06, side="right") - 1, 0, len(fing) - 1)
    return x, nb[idx]


def finger_events(fing):
    """Per finger: list of [t_on, t_off_eff, pitch, velocity, near_black, touch, lift], sorted by t_on."""
    ev = {f: [] for f in range(5)}
    for s in fing:
        nb = any(is_black(n.pitch) for n, _ in s)
        for n, f in s:
            ev[f].append([n.start, n.end, n.pitch, n.velocity, nb, getattr(n, "touch", 0.3), getattr(n, "lift", 1.0)])
    for f in ev:
        e = ev[f]
        e.sort(key=lambda r: r[0])
        for i in range(len(e) - 1):  # a finger must let go before it strikes again
            e[i][1] = min(e[i][1], e[i + 1][0] - strike_time(e[i + 1][5]) - 0.02)
            e[i][1] = max(e[i][1], e[i][0] + 0.03)
    return ev


def strike_time(touch):
    """Caressed notes are pressed slowly; struck notes are fast."""
    return 0.080 - 0.050 * touch


def finger_tip(t, e, home_tip):
    """Fingertip target for one note event at time t, and the key depth it causes (or None if idle)."""
    t_on, t_off, p, vel, nbk, touch, lift = e
    S = strike_time(touch)
    rel = 0.10 - 0.05 * touch
    approach = 0.10 + 0.08 * touch
    if t < t_on - S - approach or t > t_off + rel:
        return None, 0.0
    c = contact(p, nbk)
    down = c - np.array([0, 0, KEY_TRAVEL])
    # soft: the finger is already on (or a hair above) the key; strong: it pulls back and comes down
    h = (0.002 + 0.034 * touch ** 1.3 + 0.005 * vel / 127) * lift
    above = c + np.array([0, 0, h])
    if t < t_on - S:
        u = _smooth((t - (t_on - S - approach)) / approach)
        return home_tip + (above - home_tip) * u, 0.0
    if t < t_on:
        u = (t - (t_on - S)) / S
        u = (1 - touch) * (1 - (1 - u) ** 2) + touch * u * u      # press (ease-out) .. strike (ease-in)
        tip = above + (down - above) * u
        return tip, float(np.clip((c[2] - tip[2]) / KEY_TRAVEL, 0, 1))
    if t <= t_off:
        return down, 1.0
    u = _smooth((t - t_off) / rel)
    return down + (home_tip - down) * u, 1.0 - u


def _ik(m, target, seg, fwd_hint):
    """3-link planar IK in the vertical plane through knuckle m and the target. Returns 4 points."""
    v = target - m
    hd = np.array([v[0], v[1], 0.0])
    if np.linalg.norm(hd) < 1e-4:
        hd = np.array([fwd_hint[0], fwd_hint[1], 0.0])
    hd /= np.linalg.norm(hd)
    d, h = float(v @ hd), float(v[2])

    def fk(t1, t2):
        a1, a2, a3 = t1, t1 + t2, t1 + t2 + 0.75 * t2
        return np.array([seg[0] * math.cos(a1) + seg[1] * math.cos(a2) + seg[2] * math.cos(a3),
                         -(seg[0] * math.sin(a1) + seg[1] * math.sin(a2) + seg[2] * math.sin(a3))])

    th = np.array([0.4, 0.7])
    goal = np.array([d, h])
    for _ in range(30):
        e = goal - fk(*th)
        if e @ e < 1e-10:
            break
        J = np.empty((2, 2))
        for k in range(2):
            dt = np.zeros(2); dt[k] = 1e-4
            J[:, k] = (fk(*(th + dt)) - fk(*th)) / 1e-4
        th += np.linalg.lstsq(J, e, rcond=None)[0] * 0.8
        th = np.clip(th, [-0.35, 0.0], [1.4, 1.8])
    t1, t2 = th
    pts = [m]
    for L, a in zip(seg, (t1, t1 + t2, t1 + 1.75 * t2)):
        pts.append(pts[-1] + L * (math.cos(a) * hd + np.array([0, 0, -math.sin(a)])))
    return pts


# spring settings (f Hz, damping, response) and how far ahead each target is read to cancel the spring's lag
SPRING_HAND = (2.6, 0.72, 1.3)
SPRING_TIP = (10.0, 0.80, 0.4)
SPRING_ROLL = (3.0, 0.75, 1.0)
SPRING_HEAD = (1.7, 0.50, 0.9)
LEAD_HAND, LEAD_TIP, LEAD_HEAD = 0.07, 0.018, 0.06
ROLL_MAX = 0.12            # rad: the hand rolls toward the fingers that are playing


def _centre(H, x, nb, t):
    """Raw hand-centre target at time t: placement, attack give, arm weight, breathing."""
    yoff = 0.040 if nb else 0.0
    c = np.array([x, -0.012 + yoff, KNUCKLE_Z])
    on = H["on"]
    if not len(on):
        return c, 0.3, yoff
    i = int(np.searchsorted(on, t, side="right")) - 1
    tch = float(H["touch"][max(i, 0)])
    lo, hi = np.searchsorted(on, t - 0.3), np.searchsorted(on, t + 0.3)
    for k in range(lo, hi):
        d = t - on[k]
        c[2] -= (0.001 + 0.004 * H["vel"][k] / 127) * math.exp(-((d - 0.02) / 0.07) ** 2)
        if H["size"][k] >= 3 and H["touch"][k] > 0.3:
            c[2] += 0.045 * (H["touch"][k] - 0.3) * _smooth((d + 0.28) / 0.2) * (1 - _smooth((d + 0.06) / 0.06))
    if 0 <= i < len(on) - 1:
        g0, g1 = H["end"][i], on[i + 1]
        if g1 - g0 > 0.6:
            u = (t - g0) / (g1 - g0)
            if 0 < u < 1:
                c[2] += (0.010 + 0.020 * (1 - tch)) * math.sin(math.pi * u) ** 1.5
    return c, tch, yoff


def _roll_offsets(off, roll):
    """Rotate hand-local offsets (x across, z up) by the hand's roll about its forward axis."""
    c, s_ = math.cos(roll), math.sin(roll)
    o = np.array(off, float)
    x, z = o[..., 0].copy(), o[..., 2].copy()
    o[..., 0], o[..., 2] = x * c - z * s_, x * s_ + z * c
    return o


def animate(notes, fps=24, start=0.0, dur=None, speed=1.0, beats=None, downbeats=None,
            schedule=((0.0, "expressive"),), energy_fn=None, seed=0):
    if speed != 1.0:
        notes = [type(n)(n.start / speed, n.end / speed, n.pitch, n.velocity, n.track) for n in notes]
    end = dur + start if dur else max(n.end for n in notes) + 1.0
    notes = [n for n in notes if n.end > start - 0.5 and n.start < end + 0.5]
    # simulate from a little before the window so the springs are settled when it starts
    pre = 2.0
    times = (start - pre) + np.arange(int(round((end - start + pre) * fps))) / fps
    N = len(times)
    rh, lh = split_hands(notes)
    allx = [key_x(n.pitch) for n in notes]
    cx = float(np.mean([min(allx), max(allx)]))

    hands = {}
    for name, ns, rest in (("R", rh, cx + 0.15), ("L", lh, cx - 0.15)):
        fing = fingering(slices(ns), name) if ns else []
        ev = finger_events(fing)
        sl_end = np.array([max(n.end for n, _ in s) for s in fing]) if fing else np.zeros(0)
        H = {"fing": fing, "ev": ev, "ev_on": {f: np.array([e[0] for e in ev[f]]) for f in ev},
             "on": np.array([s[0][0].start for s in fing]) if fing else np.zeros(0),
             "end": np.maximum.accumulate(sl_end) if len(sl_end) else sl_end,
             "vel": np.array([max(n.velocity for n, _ in s) for s in fing]) if fing else np.zeros(0),
             "touch": np.array([np.mean([getattr(n, "touch", 0.3) for n, _ in s]) for s in fing]) if fing else np.zeros(0),
             "size": np.array([len(s) for s in fing]) if fing else np.zeros(0)}
        H["xh"], H["nbh"] = hand_track(fing, name, times + LEAD_HAND, rest)
        H["xt"], H["nbt"] = hand_track(fing, name, times + LEAD_TIP, rest)
        hands[name] = H

    beats = np.asarray(beats if beats is not None else np.arange(0, end + 1, 0.5))
    downbeats = np.asarray(downbeats if downbeats is not None else beats[::4])
    th = times + LEAD_HEAD
    if energy_fn is None:
        energy = performer.energy_from_onsets(sorted({n.start for n in notes}), th)
    else:
        energy = np.array([energy_fn(t) for t in th])
    sched = sorted((float(t), st) for t, st in schedule)
    body = performer.Body(hips=(cx, -0.50, -0.22), base_gaze_pitch=0.86)

    # ---- pass 1: raw targets (read slightly ahead to cancel spring lag)
    for name, H in hands.items():
        sgn = 1 if name == "R" else -1
        C = np.zeros((N, 3)); ROLL = np.zeros(N); TIP = np.zeros((N, 5, 3)); act = [[None] * 5 for _ in range(N)]
        TCH = np.zeros(N)
        for j in range(N):
            C[j], _, _ = _centre(H, H["xh"][j], H["nbh"][j], times[j] + LEAD_HAND)
            t = times[j] + LEAD_TIP
            ct, tch, yoff = _centre(H, H["xt"][j], H["nbt"][j], t)
            TCH[j] = tch
            hover = 0.003 + 0.012 * tch
            playing = []
            for f in range(5):
                home = np.array([ct[0] + sgn * HOME[f], 0.035 + yoff, hover])
                if f == 0:
                    home += np.array([sgn * 0.004, -0.012, 0.0])
                tip = home
                eo = H["ev_on"][f]
                lo, hi = max(0, int(np.searchsorted(eo, t - 8.0))), int(np.searchsorted(eo, t + 0.3))
                for e in H["ev"][f][lo:hi][::-1]:
                    tp, dep = finger_tip(t, e, home)
                    if tp is not None:
                        tip = tp
                        c = contact(e[2], e[4])
                        act[j][f] = (e[2], float(c[0]), float(c[2]))
                        if e[0] - 0.15 <= t <= e[1]:
                            playing.append(f)
                        break
                TIP[j, f] = tip
            if playing:
                ROLL[j] = sgn * ROLL_MAX * (2 - np.mean(playing)) / 2
        H["C"], H["ROLL"], H["TIP"], H["act"], H["TCH"] = C, ROLL, TIP, act, TCH

    # ---- pass 2: springs + organic drift
    for k, (name, H) in enumerate(hands.items()):
        H["Cf"] = dynamics.filter_track(H["C"], fps, *SPRING_HAND)
        H["Cf"] += np.stack([dynamics.drift(times, a, seed * 10 + 3 * k + i) for i, a in enumerate((0.0015, 0.0015, 0.0012))], 1)
        H["Rf"] = dynamics.filter_track(H["ROLL"], fps, *SPRING_ROLL) + dynamics.drift(times, 0.02, seed * 10 + 7 + k)
        H["Tf"] = dynamics.filter_track(H["TIP"].reshape(N, 15), fps, *SPRING_TIP).reshape(N, 5, 3)
    HEAD = np.array([performer.head_angles(t, beats, downbeats, float(energy[j]), sched) for j, t in enumerate(th)])
    HEAD = dynamics.filter_track(HEAD, fps, *SPRING_HEAD)
    HEAD[:, 1] += dynamics.drift(times, 0.03, seed * 10 + 11)
    HEAD[:, 2] += dynamics.drift(times, 0.02, seed * 10 + 12)

    # ---- pass 3: pose, IK, keys from the sprung fingertips
    frames = []
    j0 = int(round(pre * fps))
    for j in range(j0, N):
        t = times[j]
        fr = {"t": round(float(t - start), 4), "keys": {}, "hands": {}}
        keydepth = {}
        for name, H in hands.items():
            sgn = 1 if name == "R" else -1
            centre, roll = H["Cf"][j], float(H["Rf"][j])
            kn_off = _roll_offsets(KNUCKLE * np.array([sgn, 1, 1]), roll)
            chains = []
            for f in range(5):
                tip = H["Tf"][j, f].copy()
                a = H["act"][j][f]
                if a is not None:
                    p, kx, cz = a
                    tip[2] = max(tip[2], cz - KEY_TRAVEL)            # a key can't be pushed past its bed
                    if abs(tip[0] - kx) < WHITE_W:
                        d = float(np.clip((cz - tip[2]) / KEY_TRAVEL, 0, 1))
                        if d > 0.01:
                            keydepth[p] = max(keydepth.get(p, 0.0), d)
                kn = centre + kn_off[f]
                chains.append([[round(float(c), 5) for c in q] for q in _ik(kn, tip, SEG[f], np.array([0, 1.0, 0]))])
            w_off = _roll_offsets(np.array([sgn * 0.004, -PALM_LEN, -0.006 - 0.004 * H["TCH"][j]]), roll * 0.5)
            fr["hands"][name] = {"wrist": (centre + w_off).round(5).tolist(), "fingers": chains}
        pitch, yaw, roll_h, lean = HEAD[j]
        fr["body"] = body.pose(pitch, yaw, roll_h, lean, {h: fr["hands"][h]["wrist"] for h in fr["hands"]})
        for h in fr["hands"]:
            fr["hands"][h]["elbow"] = fr["body"]["elbows"][h]
        fr["keys"] = {str(p): round(d, 3) for p, d in keydepth.items()}
        frames.append(fr)
    fing_out = {name: [[[n.start, n.pitch, f] for n, f in s] for s in H["fing"]] for name, H in hands.items()}
    return {"fps": fps, "start": start, "speed": speed, "centre_x": cx, "keyboard": keyboard(),
            "schedule": sched, "beats": [b - start for b in beats.tolist() if start <= b <= end],
            "fingering": fing_out, "frames": frames}


if __name__ == "__main__":
    import argparse
    from . import midi
    ap = argparse.ArgumentParser()
    ap.add_argument("mid"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0, help="seconds, after --speed")
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--schedule", default="0:expressive",
                    help="head style timeline, seconds after --speed: '0:focused,8:groove,16:wild,20:crowd'")
    ap.add_argument("--feeling", help="feeling chart JSON (strength/touch/freedom/tempo/head by bar)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes-out", help="write the performed notes (for the audio) as JSON")
    a = ap.parse_args()
    notes = midi.read(a.mid)
    bts, dbs = midi.beats(a.mid, a.speed)
    sched = [(float(t), st) for t, st in (x.split(":") for x in a.schedule.split(","))]
    efn = None
    if a.feeling:
        from . import feeling
        ch = feeling.Chart(a.feeling, dbs)
        notes, tm = feeling.perform(notes, ch, a.seed)
        sched = [(float(tm(t)), h) for t, h in ch.heads]
        efn = lambda t, ch=ch, tm=tm: float(np.clip(1.4 * ch.at("strength", np.interp(t, tm.tp, tm.ts)), 0, 1))
        bts, dbs = [float(tm(b)) for b in bts], [float(tm(b)) for b in dbs]
    if a.notes_out:
        json.dump([[n.start, n.end, n.pitch, n.velocity, n.track] for n in notes], open(a.notes_out, "w"))
    anim = animate(notes, a.fps, a.start, a.dur, a.speed, bts, dbs, sched, efn, a.seed)
    json.dump(anim, open(a.out, "w"))
    print(f"{len(anim['frames'])} frames -> {a.out}")

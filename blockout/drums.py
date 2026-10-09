"""Drum blockout kinematics: drum hits -> sticking -> per-frame sticks, arms, feet, pedals and cymbal swing.

Plain numpy, no Blender, like piano.py: the output JSON holds the kit layout and, per frame, every joint the
renderer needs, so the renderer stays dumb and the motion rules live here, testable.

Frame of reference (metres): x to the drummer's right, y forward (toward the audience), z up from the floor.

Hits come from a General MIDI drum track (pitch -> piece, see GM) or from a hit list JSON
{"hits": [{"t": s, "piece": name, "vel": 1..127, "open": bool}]}, as written by h3band/drum_events.py.

Rules, in order of application:
1. Sticking: beam search over onset slices. Costs: hand preference per piece (the right hand rides the
   hi-hat crossed over the left, the left hand owns the snare), stick-tip travel speed, a hand repeating
   faster than a single stroke allows, and arms crossing anywhere but hi-hat-over-snare. Fills come out
   hand to hand. A slice with more than two hand hits keeps the two that read best (cymbals, snare).
2. Strokes: every hit is drawn on the frame nearest its onset (up to half a frame early or late), so the
   stick is on the head in exactly that frame. Stroke height follows velocity (ghost notes ~5 cm, accents
   ~35 cm). The tip accelerates into the head and rebounds straight into the next stroke's height, or
   settles to a hover in rests. Between pieces the tip travels on an arc during the upstroke.
3. The hand follows the tip through a spring (dynamics.py), lifting about a third as far: the wrist leads
   and the stick whips. The stick keeps its length; near contact the tip is pinned to the strike point.
4. Feet: right foot on the kick pedal (heel up, the leg lifts before each kick, the beater is on the head
   on the onset frame); left foot on the hi-hat pedal (heel taps the beat, releases for open hats,
   presses for pedal chicks).
5. Cymbals swing and drums rock after each hit (damped oscillation scaled by velocity).
6. Body: performer.Body with the beat-driven head styles; the head turns toward where the hands are and
   the torso leans in for far pieces.
"""
import json, math

import numpy as np

from . import dynamics, performer

STICK, TIP = 0.41, 0.28          # stick length; grip -> tip
HIPS = np.array([0.0, -0.30, 0.58])
SLICE_TOL = 0.025                # hits closer than this are one slice (played together)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _toward(deg):
    """Head normal tilted toward the drummer (-y) by deg."""
    a = math.radians(deg)
    return (0.0, -math.sin(a), math.cos(a))


# ---------------------------------------------------------------- kit
# c: centre of the playing surface, r: radius, n: surface normal, depth: shell depth,
# strike: how far from the centre toward the drummer the stick lands (fraction of r),
# pitch: stick angle below horizontal at contact.
KIT = {
    "kick":   {"kind": "kick", "c": (0.06, 0.34, 0.28), "r": 0.28, "n": (0, -1, 0), "depth": 0.40},
    "snare":  {"kind": "drum", "c": (-0.08, 0.15, 0.68), "r": 0.178, "n": _toward(6), "depth": 0.13,
               "strike": 0.30, "pitch": 14},
    "tom1":   {"kind": "drum", "c": (-0.14, 0.49, 0.84), "r": 0.127, "n": _toward(22), "depth": 0.20,
               "strike": 0.30, "pitch": 20},
    "tom2":   {"kind": "drum", "c": (0.17, 0.50, 0.84), "r": 0.152, "n": _toward(22), "depth": 0.22,
               "strike": 0.30, "pitch": 20},
    "floor":  {"kind": "drum", "c": (0.45, 0.10, 0.57), "r": 0.20, "n": _toward(3), "depth": 0.40,
               "strike": 0.35, "pitch": 16},
    "hihat":  {"kind": "hihat", "c": (-0.40, 0.30, 0.92), "r": 0.178, "n": (0, 0, 1), "strike": 0.55, "pitch": 5},
    "crash":  {"kind": "cymbal", "c": (-0.42, 0.60, 1.22), "r": 0.23, "n": _toward(14), "strike": 0.85, "pitch": 2},
    "crash2": {"kind": "cymbal", "c": (0.42, 0.66, 1.25), "r": 0.23, "n": _toward(14), "strike": 0.85, "pitch": 2},
    "ride":   {"kind": "cymbal", "c": (0.63, 0.44, 1.04), "r": 0.26, "n": _toward(10), "strike": 0.55, "pitch": 5},
}
HAND_PIECES = [p for p in KIT if p != "kick"]
HIHAT_OPEN = 0.018               # m the top hat lifts when the pedal is released

# General MIDI drum map (channel 10 pitches). Pedal hi-hat (44) is the left foot.
GM = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
      41: "floor", 43: "floor", 45: "tom2", 47: "tom2", 48: "tom1", 50: "tom1",
      42: "hihat", 44: "hihat_pedal", 46: "hihat", 49: "crash", 55: "crash", 57: "crash2", 52: "crash2",
      51: "ride", 53: "ride", 59: "ride"}
GM_OPEN = {46}

# sticking costs
PREF = {"R": {"snare": 0.6, "hihat": 0.0, "tom1": 0.6, "tom2": 0.2, "floor": 0.0, "crash": 0.5, "crash2": 0.0,
              "ride": 0.0},
        "L": {"snare": 0.0, "hihat": 1.2, "tom1": 0.1, "tom2": 0.5, "floor": 0.8, "crash": 0.0, "crash2": 1.6,
              "ride": 3.0}}
PRIORITY = {"crash": 5, "crash2": 5, "snare": 4, "tom1": 3, "tom2": 3, "floor": 3, "ride": 2, "hihat": 1}
V_EASY = 2.0                     # m/s of tip travel between pieces that costs nothing much
SINGLE_MIN = 0.18                # s: one hand repeating faster than this gets expensive (16ths go hand to hand)
ANCHOR_X = 0.17                  # hands' resting x, either side of the body


def strike_point(p):
    """Where the tip lands on piece p: from the centre toward the drummer, in the surface plane."""
    k = KIT[p]
    c, n = np.array(k["c"], float), _unit(k["n"])
    t = np.array([HIPS[0] - c[0], HIPS[1] - c[1], 0.0])
    u = _unit(t - (t @ n) * n)
    return c + k["strike"] * k["r"] * u


STRIKE = {p: strike_point(p) for p in HAND_PIECES}
NORMAL = {p: _unit(KIT[p]["n"]) for p in KIT}


def grip_home(p, hand):
    """Grip position and stick direction with the tip resting on piece p's strike point."""
    s = STRIKE[p]
    a = np.array([ANCHOR_X if hand == "R" else -ANCHOR_X, HIPS[1] + 0.12, 0.0])
    dh = _unit([s[0] - a[0], s[1] - a[1], 0.0])
    ph = math.radians(KIT[p]["pitch"])
    d = np.array([dh[0] * math.cos(ph), dh[1] * math.cos(ph), -math.sin(ph)])
    return s - TIP * d, d


# ---------------------------------------------------------------- hits
def hits_from_midi(notes):
    out = []
    for n in notes:
        p = GM.get(n.pitch)
        if p:
            out.append({"t": n.start, "piece": p, "vel": int(n.velocity), "open": n.pitch in GM_OPEN})
    return sorted(out, key=lambda h: h["t"])


def _slices(hits):
    sl = []
    for h in hits:
        if sl and h["t"] - sl[-1][0]["t"] < SLICE_TOL:
            if all(o["piece"] != h["piece"] for o in sl[-1]):
                sl[-1].append(h)
        else:
            sl.append([h])
    for s in sl:
        s.sort(key=lambda h: -PRIORITY[h["piece"]])
        for h in s[2:]:
            h["hand"] = None
        del s[2:]
    return sl


def _hand_cost(hand, p, t, prev, other):
    c = PREF[hand][p]
    pp, tp = prev
    if pp is not None:
        dt = max(t - tp, 1e-3)
        v = float(np.linalg.norm(STRIKE[p] - STRIKE[pp])) / dt
        c += 0.4 * (v / V_EASY) ** 2
        if dt < SINGLE_MIN:
            c += ((SINGLE_MIN - dt) / 0.04) ** 2
    po, to = other
    if po is not None and t - to < 0.5:
        pr, pl = (p, po) if hand == "R" else (po, p)
        if STRIKE[pr][0] < STRIKE[pl][0] - 0.03 and pr != "hihat":
            c += 3.0
    return c


def sticking(hits, beam=48):
    """Sets h['hand'] ('R'/'L', or None when dropped) on every hand hit; returns them sorted by time."""
    hh = sorted((h for h in hits if h["piece"] in HAND_PIECES), key=lambda h: h["t"])
    sl = _slices(hh)
    # state: (cost, (pR, tR), (pL, tL), parent index, choice)
    states = [(0.0, (None, -9.0), (None, -9.0), -1, None)]
    hist = []
    for s in sl:
        opts = [("R",), ("L",)] if len(s) == 1 else [("R", "L"), ("L", "R")]
        nxt = {}
        for si, (c0, R, L, _, _) in enumerate(states):
            for o in opts:
                c, st = c0, {"R": R, "L": L}
                for h, hand in zip(s, o):
                    other = st["L" if hand == "R" else "R"]
                    c += _hand_cost(hand, h["piece"], h["t"], st[hand], other)
                for h, hand in zip(s, o):
                    st[hand] = (h["piece"], h["t"])
                key = (st["R"][0], st["L"][0], round(st["R"][1], 2), round(st["L"][1], 2))
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, st["R"], st["L"], si, o)
        hist.append(states)
        states = sorted(nxt.values(), key=lambda x: x[0])[:beam]
    hist.append(states)
    # backtrack from the cheapest final state
    i, choices = 0, []
    for k in range(len(sl), 0, -1):
        st = hist[k][i]
        choices.append(st[4])
        i = st[3]
    for s, o in zip(sl, reversed(choices)):
        for h, hand in zip(s, o):
            h["hand"] = hand
    return hh


# ---------------------------------------------------------------- stroke curves
def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def stroke_height(v):
    """Prep height (m) above the head for a stroke of MIDI velocity v."""
    return 0.015 + 0.34 * (v / 127.0) ** 2


def strike_dur(h):
    return 0.035 + 0.12 * h


def stroke_curve(t, T, H, rest, dur=strike_dur, settle=0.25, lift=0.18, rebound=0.5, link=0.5):
    """Height at time t for strokes that land (height 0) at times T, each prepared from height H[k].
    Returns (height, k, w): k is the last stroke at or before t (-1 before the first), w (0..1) how far the
    hand has moved on toward stroke k+1. Strokes closer than `link` s rebound straight into the next prep
    height (ease-out up, ease-in down); with more time the tip rebounds, settles to `rest`, then lifts."""
    n = len(T)
    k = int(np.searchsorted(T, t, side="right")) - 1
    if n == 0:
        return rest, -1, 0.0
    if k < 0:
        t1, h1 = T[0], H[0]
        S = dur(h1)
        if t >= t1 - S:
            u = (t - (t1 - S)) / S
            return h1 * (1 - u * u), -1, 1.0
        return rest + (h1 - rest) * float(_smooth((t - (t1 - S - lift)) / lift)), -1, 1.0
    t0 = T[k]
    hr = max(rebound * (H[k] if k < n else rest), rest)
    if k == n - 1:
        if t < t0 + 0.12:
            u = (t - t0) / 0.12
            return hr * (1 - (1 - u) ** 2), k, 0.0
        return hr + (rest - hr) * float(_smooth((t - t0 - 0.12) / settle)), k, 0.0
    t1, h1 = T[k + 1], H[k + 1]
    g = t1 - t0
    S = min(dur(h1), 0.45 * g)
    up_end = t1 - S
    if t >= up_end:
        u = (t - up_end) / S
        w = 1.0
        return h1 * (1 - u * u), k, w
    if g - S <= link:
        u = (t - t0) / max(up_end - t0, 1e-6)
        w = float(_smooth((u - 0.15) / 0.85))
        return h1 * (1 - (1 - u) ** 2), k, w
    lift = min(lift, 0.5 * (g - S))
    w = float(_smooth((t - (up_end - lift - 0.08)) / (lift + 0.08)))
    if t < t0 + 0.12:
        u = (t - t0) / 0.12
        return hr * (1 - (1 - u) ** 2), k, w
    if t < up_end - lift:
        return hr + (rest - hr) * float(_smooth((t - t0 - 0.12) / settle)), k, w
    u = (t - (up_end - lift)) / lift
    return rest + (h1 - rest) * float(_smooth(u)), k, w


def _snap(t, start, fps):
    return start + round((t - start) * fps) / fps


# ---------------------------------------------------------------- piece motion
SWING = {  # (amplitude at vel 0, amplitude per unit vel, frequency Hz, decay s)
    "cymbal": (0.05, 0.13, 1.6, 1.5), "ride": (0.02, 0.05, 1.3, 1.2), "hihat": (0.008, 0.02, 4.0, 0.25),
    "hihat_open": (0.03, 0.05, 3.0, 0.6), "drum": (0.006, 0.012, 7.0, 0.15)}


def swing_params(p, h):
    if p == "ride":
        return SWING["ride"]
    kind = KIT[p]["kind"]
    if kind == "hihat":
        return SWING["hihat_open" if h.get("open") else "hihat"]
    return SWING[kind]


def piece_tilt(hits, times):
    """Per piece: (N, 3) axis-angle rotation about the piece centre (edge struck goes down, then swings)."""
    out = {}
    for p in HAND_PIECES:
        hs = [h for h in hits if h["piece"] == p]
        rot = np.zeros((len(times), 3))
        if hs:
            n = NORMAL[p]
            u = _unit(STRIKE[p] - np.array(KIT[p]["c"]))
            axis = _unit(np.cross(n, u))
            ang = np.zeros(len(times))
            for h in hs:
                a0, a1, f, tau = swing_params(p, h)
                d = times - h["t"]
                m = d > 0
                ang[m] += (a0 + a1 * h["vel"] / 127) * np.exp(-d[m] / tau) * np.sin(2 * np.pi * f * d[m] + 0.25)
            rot = ang[:, None] * axis[None, :]
        out[p] = rot
    return out


# ---------------------------------------------------------------- feet
KICK_X = 0.08
PEDAL_HEEL = np.array([KICK_X, -0.02, 0.03])       # kick footboard pivot
BOARD_L = 0.27
BEATER_PIVOT = np.array([KICK_X, 0.24, 0.13])
BEATER_L = 0.165
HAT_HEEL = np.array([-0.34, 0.00, 0.03])
HAT_DIR = _unit([-0.10, 1.0, 0.0])
THIGH, SHIN = 0.44, 0.45


def board(heel, dirxy, angle, s):
    return heel + s * np.array([dirxy[0] * math.cos(angle), dirxy[1] * math.cos(angle), math.sin(angle)])


def beater_angle(p):
    """Beater angle from vertical (forward positive); p 0 = on the head, 1 = pulled fully back."""
    hit = math.atan2(KIT["kick"]["c"][1] - 0.035 - BEATER_PIVOT[1], KIT["kick"]["c"][2] - BEATER_PIVOT[2])
    return hit - 0.80 * p


def leg_ik(hip, ankle, side):
    d = ankle - hip
    L = min(max(float(np.linalg.norm(d)), 1e-4), THIGH + SHIN - 1e-3)
    u = _unit(d)
    a = (THIGH ** 2 - SHIN ** 2 + L ** 2) / (2 * L)
    h = math.sqrt(max(THIGH ** 2 - a * a, 0.0))
    pole = np.array([side * 0.25, 1.0, 0.6])
    pole -= (pole @ u) * u
    return hip + a * u + h * _unit(pole)


def foot_points(heel_pivot, dirxy, angle, ball_s, heel_lift):
    """Ball, heel and ankle of a foot on a pedal whose board is at `angle`."""
    ball = board(heel_pivot, dirxy, angle, ball_s) + np.array([0, 0, 0.025])
    toe = board(heel_pivot, dirxy, angle, ball_s + 0.07) + np.array([0, 0, 0.02])
    heel = np.array([ball[0] - 0.16 * dirxy[0], ball[1] - 0.16 * dirxy[1], heel_pivot[2] + 0.03 + heel_lift])
    ankle = heel + np.array([0.03 * dirxy[0], 0.03 * dirxy[1], 0.075])
    return toe, ball, heel, ankle


# ---------------------------------------------------------------- body
SPRING_HAND = (5.0, 0.60, 1.0)
SPRING_LOOK = (1.6, 0.8, 0.5)
SPRING_HEAD = (1.7, 0.50, 0.9)
SPRING_LEG = (6.0, 0.7, 1.0)
LEAD_HAND, LEAD_HEAD = 0.035, 0.06
HAND_LIFT, HAND_PULL = 0.30, 0.10   # hand rises this fraction of the tip's height, and draws back
REST_H = 0.06                       # hover height of a resting tip


def _snap_curve(t, T, H, rest, fps):
    """Two poses: the tip is on the head on a hit's frame and at the next stroke's prep height on every other
    frame, with no easing. Returns (height, k, w) like stroke_curve; w jumps to the next piece after the hit."""
    k = int(np.searchsorted(T, t + 0.5 / fps, side="right")) - 1
    if k >= 0 and abs(t - T[k]) < 0.5 / fps:
        return 0.0, k, 0.0
    if k + 1 < len(T):
        return H[k + 1], k, 1.0
    return rest, k, 0.0


def _hand_track(hs, times, hand, start, fps, motion="smooth"):
    """Raw targets for one hand: tip target, hand (grip) target, height above the surface, current piece."""
    N = len(times)
    TIPT, HAND, HH = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(N)
    cur = []
    if not hs:
        p = "snare" if hand == "L" else "hihat"
        g, d = grip_home(p, hand)
        for j in range(N):
            TIPT[j] = STRIKE[p] + REST_H * NORMAL[p]
            HAND[j] = g + np.array([0, 0, HAND_LIFT * REST_H])
            HH[j] = REST_H
            cur.append(p)
        return TIPT, HAND, HH, cur, []
    T = np.array([h["tf"] for h in hs])
    Hh = np.array([stroke_height(h["vel"]) for h in hs])
    P = [h["piece"] for h in hs]
    S = [STRIKE[p] + np.array([0, 0, HIHAT_OPEN if h.get("open") else 0.0]) + np.asarray(h.get("aim", (0, 0, 0)))
         for p, h in zip(P, hs)]
    G = [grip_home(p, hand) for p in P]
    for j, t in enumerate(times):
        if motion == "snap":
            h, k, w = _snap_curve(t, T, Hh, REST_H, fps)
        else:
            h, k, w = stroke_curve(t, T, Hh, REST_H)
        a, b = (max(k, 0), min(k + 1, len(T) - 1)) if k >= 0 else (0, 0)
        if k < 0:
            w = 0.0
        s = S[a] + (S[b] - S[a]) * w
        n = _unit(NORMAL[P[a]] + (NORMAL[P[b]] - NORMAL[P[a]]) * w)
        arc = 0.0 if motion == "snap" else 0.25 * math.sin(math.pi * w) * float(np.linalg.norm(S[b] - S[a]))
        g = G[a][0] + (G[b][0] - G[a][0]) * w
        d = _unit(G[a][1] + (G[b][1] - G[a][1]) * w)
        dh = _unit([d[0], d[1], 0.0])
        hh = h + arc
        TIPT[j] = s + hh * n
        HAND[j] = g + np.array([0, 0, HAND_LIFT * hh + 0.4 * arc]) - HAND_PULL * h * dh
        HH[j] = hh
        cur.append(P[a] if w < 0.5 else P[b])
    return TIPT, HAND, HH, cur, T


def _kick_track(kicks, times, start, fps):
    """Pedal state p (0 on the head .. 1 pulled back) and leg lift, per frame."""
    T = np.array([_snap(h["t"], start, fps) for h in kicks])
    if len(T):
        T, idx = np.unique(T, return_index=True)
        vel = np.array([kicks[i]["vel"] for i in idx])
    else:
        vel = np.zeros(0)
    Hk = np.where(vel >= 60, 1.0, 0.6)
    P, LIFT = np.zeros(len(times)), np.zeros(len(times))
    for j, t in enumerate(times):
        h, k, _ = stroke_curve(t, T, Hk, 0.8, dur=lambda hh: 0.045 + 0.03 * hh, lift=0.15, rebound=1.0, link=0.4)
        nv = vel[min(k + 1, len(vel) - 1)] / 127 if len(vel) else 0.5
        P[j] = float(np.clip(h, 0, 1))
        LIFT[j] = (0.015 + 0.045 * nv) * P[j] ** 2
    return P, LIFT, T


def _hat_track(hits, times, start, fps, beats):
    """Hi-hat openness (0 closed .. 1 open) and the left heel's lift, per frame."""
    ev = sorted((_snap(h["t"], start, fps), h) for h in hits if h["piece"] in ("hihat", "hihat_pedal"))
    pedal = [t for t, h in ev if h["piece"] == "hihat_pedal"]
    OPEN, HEEL = np.zeros(len(times)), np.zeros(len(times))
    # open from 80 ms before an open hit until the next closed hit or pedal chick
    spans = []
    for i, (t, h) in enumerate(ev):
        if h["piece"] == "hihat" and h.get("open"):
            close = next((t2 for t2, h2 in ev[i + 1:] if not h2.get("open")), t + 0.5)
            spans.append((t - 0.08, max(close, t + 0.1)))
    for t0, t1 in spans:
        OPEN += _smooth((times - t0) / 0.06) * (1 - _smooth((times - (t1 - 0.05)) / 0.05))
    for t in pedal:   # foot rises before a pedal chick and lands on it
        d = times - t
        OPEN += np.where((d > -0.14) & (d <= 0), 0.6 * np.sin(np.pi * (d + 0.14) / 0.14) ** 2, 0.0)
    OPEN = np.clip(OPEN, 0, 1)
    if len(beats) > 1:     # heel taps the beat while the hat is closed
        for j, t in enumerate(times):
            ph, _ = performer._beat_phase(t + 0.03, beats)
            HEEL[j] = 0.035 * math.sin(math.pi * ph) ** 2
    HEEL = HEEL * (1 - OPEN) + 0.05 * OPEN
    return OPEN, HEEL


# ---------------------------------------------------------------- animate
MOTIONS = ("smooth", "snap", "loose")
LOOSE_FRAMES, LOOSE_AIM = 2, 0.08   # loose: each hand hit moves up to this many frames and aims up to this far off


def animate(hits, fps=24, start=0.0, dur=None, beats=None, downbeats=None, schedule=((0.0, "groove"),), seed=0,
            motion="smooth"):
    """motion: "smooth" (the rules above), "snap" (hands jump between a prep pose and the hit pose, no easing,
    no springs), or "loose" (smooth, but every hand hit lands up to LOOSE_FRAMES early or late and up to
    LOOSE_AIM m off its strike point). snap and loose exist to test how exact a blockout has to be."""
    if motion not in MOTIONS:
        raise ValueError(f"motion must be one of {MOTIONS}")
    rng = np.random.default_rng(seed + 1000)
    hits = [dict(h) for h in hits]
    end = start + dur if dur else max(h["t"] for h in hits) + 1.5
    hits = [h for h in hits if start - 2.0 < h["t"] < end + 1.0]
    pre = 2.0
    times = (start - pre) + np.arange(int(round((end - start + pre) * fps))) / fps
    N = len(times)

    hand_hits = sticking(hits)
    tracks = {}
    for hand in ("R", "L"):
        hs = [h for h in hand_hits if h.get("hand") == hand]
        for h in hs:
            h["tf"] = _snap(h["t"], start, fps)
            if motion == "loose":
                h["tf"] += int(rng.integers(-LOOSE_FRAMES, LOOSE_FRAMES + 1)) / fps
                r, ang = LOOSE_AIM * math.sqrt(rng.random()), 2 * math.pi * rng.random()
                h["aim"] = (r * math.cos(ang), r * math.sin(ang), 0.0)
        hs.sort(key=lambda h: h["tf"])
        keep = []
        for h in hs:     # one stroke per frame per hand: a faster repeat is merged into the stroke before it
            if keep and h["tf"] <= keep[-1]["tf"]:
                keep[-1]["vel"] = max(keep[-1]["vel"], h["vel"])
                h["hand"] = None
                continue
            keep.append(h)
        tracks[hand] = (keep,) + _hand_track(keep, times, hand, start, fps, motion)

    kicks = [h for h in hits if h["piece"] == "kick"]
    KP, KLIFT, kick_T = _kick_track(kicks, times, start, fps)
    beats = np.asarray(beats if beats is not None else np.arange(0, end + 1, 0.5))
    downbeats = np.asarray(downbeats if downbeats is not None else beats[::4])
    OPEN, HEEL_L = _hat_track(hits, times, start, fps, beats)
    tilt = piece_tilt([h for h in hits if h.get("hand")], times)

    # hands: springs on the grip, read ahead to cancel the lag
    lead = int(round(LEAD_HAND * fps))
    Hs = {}
    for k, hand in enumerate(("R", "L")):
        raw = tracks[hand][2]
        if motion == "snap":
            Hs[hand] = raw
            continue
        raw = np.concatenate([raw[lead:], np.repeat(raw[-1:], lead, 0)])
        f = dynamics.filter_track(raw, fps, *SPRING_HAND)
        f += np.stack([dynamics.drift(times, 0.004, seed * 10 + 3 * k + i) for i in range(3)], 1)
        Hs[hand] = f
    # head: beat-driven style + looking toward the hands; torso leans in for far pieces
    th = times + LEAD_HEAD
    onsets = sorted(h["t"] for h in hits)
    energy = performer.energy_from_onsets(onsets, th) if onsets else np.zeros(N)
    sched = sorted((float(t), st) for t, st in schedule)
    mid = 0.5 * (tracks["R"][1] + tracks["L"][1])
    look = np.stack([-0.9 * np.arctan2(mid[:, 0], mid[:, 1] - HIPS[1] + 0.25),
                     0.6 * np.clip(mid[:, 1] - 0.15, 0, None)], 1)
    look = dynamics.filter_track(look, fps, *SPRING_LOOK)
    HEAD = np.array([performer.head_angles(t, beats, downbeats, float(energy[j]), sched, seed)
                     for j, t in enumerate(th)])
    HEAD = dynamics.filter_track(HEAD, fps, *SPRING_HEAD)
    HEAD[:, 1] += look[:, 0] + dynamics.drift(times, 0.03, seed * 10 + 11)
    HEAD[:, 2] += dynamics.drift(times, 0.02, seed * 10 + 12)
    HEAD[:, 3] += look[:, 1]
    body = performer.Body(hips=HIPS, base_gaze_pitch=0.75)
    LEG = dynamics.filter_track(np.stack([KLIFT, HEEL_L], 1), fps, *SPRING_LEG)

    frames = []
    j0 = int(round(pre * fps))
    hat_dir = HAT_DIR
    for j in range(j0, N):
        t = times[j]
        fr = {"t": round(float(t - start), 4), "hands": {}}
        wrists = {}
        for hand in ("R", "L"):
            tipt, hh = tracks[hand][1][j], tracks[hand][3][j]
            hs = Hs[hand][j]
            d = _unit(tipt - hs)
            free = hs + TIP * d
            c = math.exp(-max(hh, 0.0) / 0.012)
            tip = free + c * (tipt - free)
            grip = tip - TIP * d
            butt = grip - (STICK - TIP) * d
            wrist = grip - 0.05 * d - np.array([0, 0, 0.012])
            wrists[hand] = wrist
            fr["hands"][hand] = {"tip": tip, "grip": grip, "butt": butt, "wrist": wrist, "piece": tracks[hand][4][j]}
        pitch, yaw, roll, lean = HEAD[j]
        b = body.pose(pitch, yaw, roll, lean, wrists)
        for hand in ("R", "L"):
            fr["hands"][hand]["elbow"] = b["elbows"][hand]
            fr["hands"][hand]["shoulder"] = b["shoulders"][hand]
        # feet
        kp = KP[j]
        k_ang = 0.08 + 0.20 * kp
        k_toe, k_ball, k_heel, k_ank = foot_points(PEDAL_HEEL, (0, 1), k_ang, 0.20, 0.05 + LEG[j, 0])
        op = OPEN[j]
        h_ang = 0.07 + 0.16 * op
        h_toe, h_ball, h_heel, h_ank = foot_points(HAT_HEEL, hat_dir, h_ang, 0.20, LEG[j, 1])
        feet = {}
        for side, (toe, ball, heel, ank) in (("R", (k_toe, k_ball, k_heel, k_ank)), ("L", (h_toe, h_ball, h_heel, h_ank))):
            sx = 1 if side == "R" else -1
            hip = HIPS + np.array([sx * 0.12, 0.06, -0.05])
            feet[side] = {"hip": hip, "knee": leg_ik(hip, ank, sx), "ankle": ank, "heel": heel, "ball": ball, "toe": toe}
        fr["feet"] = feet
        fr["kick"] = {"board": round(k_ang, 4), "beater": round(beater_angle(kp), 4)}
        fr["hihat"] = {"open": round(float(op), 3), "board": round(h_ang, 4)}
        fr["tilt"] = {p: tilt[p][j] for p in tilt if np.any(tilt[p][j])}
        fr["body"] = b
        frames.append(_round(fr))

    drawn = [{"t": round(h["t"] - start, 4), "tf": round(h["tf"] - start, 4), "piece": h["piece"], "vel": h["vel"],
              "hand": h["hand"]} for h in hand_hits if h.get("hand") and start <= h["tf"] < end]
    kit = {p: {**{k: v for k, v in KIT[p].items()}, "strike_pt": STRIKE.get(p, np.array(KIT[p]["c"])).tolist(),
               "n": NORMAL[p].tolist()} for p in KIT}
    return {"fps": fps, "start": start, "kit": kit, "stick": [STICK, TIP], "hihat_open": HIHAT_OPEN,
            "kick_pedal": {"heel": PEDAL_HEEL.tolist(), "len": BOARD_L, "beater_pivot": BEATER_PIVOT.tolist(),
                           "beater_len": BEATER_L},
            "hat_pedal": {"heel": HAT_HEEL.tolist(), "dir": HAT_DIR.tolist(), "len": BOARD_L},
            "schedule": sched, "beats": [round(float(x) - start, 4) for x in beats if start <= x <= end],
            "hits": drawn, "kicks": [round(float(x) - start, 4) for x in kick_T if start <= x < end],
            "frames": frames}


def _round(o):
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_round(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 4)
    return o


# ---------------------------------------------------------------- checks
def check(anim):
    """Timing and geometry checks on an animation dict. Returns a summary dict."""
    fps, fr = anim["fps"], anim["frames"]
    kit = anim["kit"]
    errs = []
    for h in anim["hits"]:
        j = int(round(h["tf"] * fps))
        if 0 <= j < len(fr):
            s = np.array(kit[h["piece"]]["strike_pt"])
            if h["piece"] == "hihat" and fr[j]["hihat"]["open"] > 0.5:
                s = s + np.array([0, 0, anim["hihat_open"]])
            errs.append(float(np.linalg.norm(np.array(fr[j]["hands"][h["hand"]]["tip"]) - s)))
    pen, reach = 0, 0.0
    for f in fr:
        for hand, hd in f["hands"].items():
            tip = np.array(hd["tip"])
            for p, k in kit.items():
                if k["kind"] in ("kick",):
                    continue
                c, n = np.array(k["c"]), np.array(k["n"])
                d = float((tip - c) @ n)
                rad = float(np.linalg.norm((tip - c) - d * n))
                deep = k.get("depth", 0.02)
                if rad < k["r"] * 0.95 and -deep < d < -0.006:
                    pen += 1
            reach = max(reach, float(np.linalg.norm(np.array(hd["wrist"]) - np.array(hd["shoulder"]))))
    kick_ok = 0
    for t in anim["kicks"]:
        j = int(round(t * fps))
        if 0 <= j < len(fr) and abs(fr[j]["kick"]["beater"] - max(f["kick"]["beater"] for f in fr)) < 1e-3:
            kick_ok += 1
    return {"hand_hits": len(errs), "contact_err_mm_max": round(1000 * max(errs), 2) if errs else None,
            "contact_err_mm_median": round(1000 * float(np.median(errs)), 2) if errs else None,
            "tip_penetrating_frames": pen, "max_reach_m": round(reach, 3),
            "arm_length_m": performer.UPPER_ARM + performer.FOREARM,
            "kicks": len(anim["kicks"]), "kicks_beater_on_head": kick_ok,
            "sticking": {hd: sum(h["hand"] == hd for h in anim["hits"]) for hd in ("R", "L")}}


if __name__ == "__main__":
    import argparse
    from . import midi
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="GM drum MIDI, or a hit list JSON from h3band/drum_events.py")
    ap.add_argument("out")
    ap.add_argument("--beats", help="beats JSON {'beats': [...]} (hit lists; MIDI uses its time signature)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--schedule", default="0:groove",
                    help="head style timeline in seconds: '0:focused,8:groove,16:wild,20:crowd'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion", choices=MOTIONS, default="smooth")
    ap.add_argument("--hits-out", help="write the hit list (for drumsynth / scoring) as JSON")
    a = ap.parse_args()
    bts = dbs = None
    if a.src.lower().endswith((".mid", ".midi")):
        hits = hits_from_midi(midi.read(a.src))
        bts, dbs = midi.beats(a.src)
    else:
        hits = json.load(open(a.src))["hits"]
    if a.beats:
        bj = json.load(open(a.beats))
        bts = bj["beats"]
        dbs = bj.get("downbeats", bts[::4])
    if a.hits_out:
        json.dump({"hits": hits}, open(a.hits_out, "w"))
    sched = [(float(t), st) for t, st in (x.split(":") for x in a.schedule.split(","))]
    anim = animate(hits, a.fps, a.start, a.dur, bts, dbs, sched, a.seed, a.motion)
    json.dump(anim, open(a.out, "w"))
    print(f"{len(anim['frames'])} frames -> {a.out}")
    print(json.dumps(check(anim)))

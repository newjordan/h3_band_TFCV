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

from . import drum_collide, drum_hands, drum_kit, drum_style, drum_teacher, drum_toss, dynamics, performer

STICK, TIP = 0.41, 0.28          # stick length; grip -> tip
# the hips sit 8 cm behind the seat point the kits are laid out around (drum_kit.SEAT): the rigged hand
# (drum_hands) holds the stick's fulcrum about 14 cm ahead of its wrist, and the arms need the room
HIPS = np.array([0.0, -0.38, 0.58])
SLICE_TOL = 0.025                # hits closer than this are one slice (played together)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


# ---------------------------------------------------------------- kit
# The kit is data (drum_kit.py): c: centre of the playing surface, r: radius, n: surface normal, depth: shell
# depth, strike: how far from the centre toward the drummer the stick lands (fraction of r), pitch: stick angle
# below horizontal at contact. use_kit() points the rig at another kit; the tables below are the standard kit's.
KIT = {p: dict(k) for p, k in drum_kit.STANDARD.items()}
HAND_PIECES = [p for p in KIT if p != "kick"]
HIHAT_OPEN = 0.018               # m the top hat lifts when the pedal is released

# General MIDI drum map (channel 10 pitches) onto the standard kit; hits keep their note, and animate(kit=...)
# points them at another kit's pieces by it (drum_kit.remap). Pedal hi-hat (44) is the left foot.
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
STD_PREF = {h: dict(v) for h, v in PREF.items()}
PRIORITY = {"crash": 5, "crash2": 5, "snare": 4, "tom1": 3, "tom2": 3, "floor": 3, "ride": 2, "hihat": 1}
PRIORITY_TYPE = {"crash": 5, "splash": 5, "china": 5, "snare": 4, "tom": 3, "floor": 3, "ride": 2, "hihat": 1}
V_EASY = 2.0                     # m/s of tip travel between pieces that costs nothing much
SINGLE_MIN = 0.18                # s: one hand repeating faster than this gets expensive (16ths go hand to hand)
ANCHOR_X = 0.17                  # hands' resting x, either side of the body


def strike_point(p):
    """Where the tip lands on piece p: from the centre toward the drummer, in the surface plane."""
    k = KIT[p]
    c, n = np.array(k["c"], float), _unit(k["n"])
    t = np.array([HIPS[0] - c[0], HIPS[1] - c[1], 0.0])
    u = _unit(t - (t @ n) * n)
    s = c + k["strike"] * k["r"] * u
    lift = drum_kit.surface_lift(p, k)
    return s + lift * n if lift else s


STRIKE = {p: strike_point(p) for p in HAND_PIECES}
NORMAL = {p: _unit(KIT[p]["n"]) for p in KIT}


YAW_SPAN, YAW_STEP = 60, 3       # deg either side of the anchor's line grip_home turns the stick, in steps
YAW_COST = 1.0                   # per rad the stick turns off the anchor's line
ARM_CLEAR = 0.06                 # m an elbow keeps off the torso capsule
ELBOW_UP = 0.05                  # m below the shoulder an elbow rises to before it costs
KIT_CLEAR = 0.015                # m the fist and stick keep off the other pieces, at the strike and raised ...
KIT_COST = 200.0                 # ... per m closer
LIFT_PITCH = 45                  # deg the stick turns up about the grip for the raised check
_HOME = {}


def _elbows(sh, w, n=24):
    """Elbows around the shoulder-wrist axis."""
    UA, FA = performer.UPPER_ARM, performer.FOREARM
    v = w - sh
    L = min(float(np.linalg.norm(v)), UA + FA - 1e-3)
    u = _unit(v)
    a = (UA ** 2 - FA ** 2 + L ** 2) / (2 * L)
    h = math.sqrt(max(UA ** 2 - a * a, 0.0))
    e1 = _unit(np.cross(u, [0.0, 0.0, 1.0]))
    e2 = np.cross(u, e1)
    for k in range(n):
        t = 2 * math.pi * k / n
        yield sh + a * u + h * (math.cos(t) * e1 + math.sin(t) * e2)


def _arm_awkward(hand, s, d, sh, torso):
    """How awkward the arm is with the stick along d and its tip on s: the best elbow's drum_hands.arm_cost,
    with the elbow kept ARM_CLEAR off the torso and below the shoulder, and the wrist within arm's length."""
    w = drum_hands.wrist(s - TIP * d, d, hand)
    a0, a1 = torso
    ab = a1 - a0
    reach = float(np.linalg.norm(w - sh)) - (performer.UPPER_ARM + performer.FOREARM - 0.02)
    best = None
    for el in _elbows(sh, w):
        t = float(np.clip((el - a0) @ ab / (ab @ ab), 0, 1))
        clear = float(np.linalg.norm(el - (a0 + t * ab))) - drum_collide.TORSO_R
        c = (drum_hands.arm_cost(d, hand, sh, el, w) + 20 * max(0.0, ARM_CLEAR - clear)
             + 3 * max(0.0, el[2] - (sh[2] - ELBOW_UP)))
        best = c if best is None else min(best, c)
    return best + 50 * max(0.0, reach)


def _kit_clash(kit, p, hand, s, d):
    """How far (m) the fist and stick come inside KIT_CLEAR of the pieces other than p (and their stands), with
    the tip on s plus with the stick raised LIFT_PITCH about the grip: a hand turned in under a neighbouring
    cymbal would hit it on every stroke."""
    g = s - TIP * d
    up = _unit(np.array([0.0, 0.0, 1.0]) - d[2] * d)
    a = math.radians(LIFT_PITCH)
    out = 0.0
    for dd in (d, math.cos(a) * d + math.sin(a) * up):
        pts, rad = [], []
        for a0, b0, r0, r1 in [(g - (STICK - TIP) * dd, g + TIP * dd, *drum_collide.STICK_R),
                               *drum_hands.hitbox(g, dd, hand)]:
            x, r, _ = drum_collide.samples(a0, b0, r0, r1, 10)
            pts.append(x), rad.append(r)
        pts, rad = np.concatenate(pts), np.concatenate(rad)
        sd = np.min(np.stack([v for k, v in kit.sd(pts).items() if k[0] != p and not k[0].startswith(p + "_")]),
                    axis=0)
        out += float(np.clip(rad + KIT_CLEAR - sd, 0, None).max())
    return out


def grip_home(p, hand):
    """Grip position and stick direction with the tip resting on piece p's strike point: the stick points from
    the hand's anchor beside the body toward the strike, turned (up to YAW_SPAN) where that leaves the arm
    less awkward and the hand clear of the other pieces: the rigged hand holds the stick across the palm, so
    its wrist sits well behind and inside the stick's line, and a crossed-over or far reach needs the stick
    turned for the elbow to find room."""
    key = (p, hand)
    if key not in _HOME:
        if "kit" not in _HOME:
            _HOME["kit"] = drum_collide.Kit(KIT, HIHAT_OPEN)
        kit = _HOME["kit"]
        s = STRIKE[p]
        a = np.array([ANCHOR_X if hand == "R" else -ANCHOR_X, HIPS[1] + 0.12, 0.0])
        y0 = math.atan2(s[0] - a[0], s[1] - a[1])
        ph = math.radians(KIT[p]["pitch"])
        b = performer.Body(hips=HIPS).pose(0.0, 0.0, 0.0, 0.0, {})
        sh = np.array(b["shoulders"][hand])
        torso = (np.array(b["hips"]) + np.array([0, 0, 0.12]), np.array(b["chest"]))

        def stick(y):
            return np.array([math.sin(y) * math.cos(ph), math.cos(y) * math.cos(ph), -math.sin(ph)])

        k = min(range(-YAW_SPAN, YAW_SPAN + 1, YAW_STEP),
                key=lambda k: _arm_awkward(hand, s, stick(y0 + math.radians(k)), sh, torso)
                + KIT_COST * _kit_clash(kit, p, hand, s, stick(y0 + math.radians(k)))
                + YAW_COST * abs(math.radians(k)))
        d = stick(y0 + math.radians(k))
        _HOME[key] = (s - TIP * d, d)
    g, d = _HOME[key]
    return g.copy(), d.copy()


HOOP_TOP = 0.006 + 0.0075       # hoop top above the head (blender_drums.py) plus the stick's radius


def _in_plane_toward(p, point):
    """Unit vector in piece p's surface plane, from its centre toward `point`."""
    c, n = np.array(KIT[p]["c"], float), NORMAL[p]
    t = np.asarray(point, float) - c
    return _unit(t - (t @ n) * n)


def _anchor(hand):
    return np.array([ANCHOR_X if hand == "R" else -ANCHOR_X, HIPS[1] + 0.12, KIT["snare"]["c"][2]])


def technique_pose(p, hand, tech):
    """(strike point, (grip, stick dir)) for a technique stroke, or None to use the plain strike.
    rimshot: tip near the centre, stick low enough that the shaft lands on the near hoop at the same time.
    bell: tip on the ride's bell, stick steep. cross_stick: the stick lies across the snare, its tip clicks the
    far hoop and the grip rests just above the near half of the head."""
    c, n, r = np.array(KIT[p]["c"], float), NORMAL[p], KIT[p]["r"]
    u = _in_plane_toward(p, _anchor(hand))
    if tech == "rimshot" and KIT[p]["kind"] == "drum":
        s = c + 0.10 * r * u
        a = math.atan2(HOOP_TOP, r + 0.008 - 0.10 * r)
    elif tech == "cross_stick" and drum_kit.piece_type(p) == "snare":
        s = c - (r + 0.008) * u + HOOP_TOP * n
        a = math.atan2(0.03 - HOOP_TOP, TIP)
    elif tech == "bell" and drum_kit.piece_type(p) == "ride":
        s = c + 0.035 * u + 0.016 * n
        a = math.radians(38)
    else:
        return None
    back = _unit(math.cos(a) * u + math.sin(a) * n)
    return s, (s + TIP * back, -back)


def choke_pose(p, hand):
    """Grip and tip targets for a hand pinching cymbal p's edge, the stick pointing back and down from the fist."""
    c, n, r = np.array(KIT[p]["c"], float), NORMAL[p], KIT[p]["r"]
    u = _in_plane_toward(p, _anchor(hand))
    grip = c + 0.97 * r * u + 0.02 * u - 0.02 * n
    d = _unit(_unit([u[0], u[1], 0.0]) + np.array([0, 0, -0.5]))   # flat enough to clear the hi-hat
    return grip, grip + TIP * d


# ---------------------------------------------------------------- hits
def hits_from_midi(notes):
    out = []
    for n in notes:
        p = GM.get(n.pitch)
        if p:
            out.append({"t": n.start, "piece": p, "vel": int(n.velocity), "open": n.pitch in GM_OPEN, "note": n.pitch})
    return sorted(out, key=lambda h: h["t"])


def _slices(hits):
    sl = []
    for h in hits:
        flam = h.get("tech") == "flam" or sl and sl[-1][0].get("tech") == "flam_grace"   # grace + main: two slices
        if sl and h["t"] - sl[-1][0]["t"] < SLICE_TOL and not flam:
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
        opts = [o for o in opts if all(h.get("hand_lock") in (None, hd) for h, hd in zip(s, o))] or opts
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


# Measured strokes (Dahl 2004, accent strokes; marker near the tip): pp 0.15-0.20 m, mf 0.23-0.30 m, ff 0.38-0.50 m
# of prep height, and a downstroke of 60-125 ms that barely changes with height (drum-cam footage, 2-3 frames at
# 24 fps). Heights shrink when strokes come fast (Dahl 2011): below DENSE_IOI s between a hand's strokes the prep
# height scales with sqrt(gap / DENSE_IOI). The physics rig uses these; --no-physics keeps the curves above.
DENSE_IOI = 0.25


def stroke_height_ref(v):
    return 0.03 + 0.42 * v / 127.0


STYLE_SCALE = 0.45 / 0.355     # drum_style's tiers, arm lift and lean were set on stroke_height; the measured ff
                               # stroke is this much taller, so they read heights divided by it


def strike_dur_ref(h):
    return 0.07 + 0.08 * h


def hand_lift_ref(v):
    """Measured hand rise: about 0.3 of the tip's height for ordinary strokes, up to about 0.5 for full accents."""
    return HAND_LIFT + 0.2 * float(np.clip((v - 100) / 27.0, 0.0, 1.0))


def stroke_curve(t, T, H, rest, dur=strike_dur, settle=0.25, lift=0.18, rebound=0.5, link=0.5):
    """Height at time t for strokes that land (height 0) at times T, each prepared from height H[k].
    Returns (height, k, w): k is the last stroke at or before t (-1 before the first), w (0..1) how far the
    hand has moved on toward stroke k+1. rebound is one fraction or one per stroke; so is rest (the hover after it). Strokes closer than `link` s rebound straight into the next prep
    height (ease-out up, ease-in down); with more time the tip rebounds, settles to `rest`, then lifts."""
    n = len(T)
    k = int(np.searchsorted(T, t, side="right")) - 1
    if n == 0:
        return rest, -1, 0.0
    if k < 0:
        rest = rest[0] if np.ndim(rest) else rest
        t1, h1 = T[0], H[0]
        S = dur(h1)
        if t >= t1 - S:
            u = (t - (t1 - S)) / S
            return h1 * (1 - u * u), -1, 1.0
        return rest + (h1 - rest) * float(_smooth((t - (t1 - S - lift)) / lift)), -1, 1.0
    t0 = T[k]
    rb = rebound[k] if np.ndim(rebound) else rebound
    rest = rest[k] if np.ndim(rest) else rest
    hr = max(rb * (H[k] if k < n else rest), rest)
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
# cymbal_wobble's crash and ride: a crash filmed at 1000 fps rings at about 3.0 Hz with tau about 0.9 s; a hard hit
# swings it 0.35 rad (up to 0.63 at the extreme). 2.5 Hz allows for bigger crashes and looser felts. The ride is
# heavier: about 0.8 of the crash frequency (not measured).
SWING_PHYS = dict(SWING, cymbal=(0.06, 0.29, 2.5, 1.0), ride=(0.02, 0.07, 2.0, 1.2))


def swing_params(p, h, table=SWING):
    if drum_kit.piece_type(p) == "ride":
        return table["ride"]
    kind = KIT[p]["kind"]
    if kind == "hihat":
        return table["hihat_open" if h.get("open") else "hihat"]
    return table[kind]


CHOKE_TAU, CHOKE_DIP = 0.04, 0.05   # s for a choked cymbal to stop; rad the pinch pulls the edge down
WOBBLE_SPLIT = 1.12                 # the tilter's two axes differ in stiffness: off-axis hits wobble in an ellipse
CHOKE_F = 6.0                       # Hz: how stiffly a pinching hand holds the cymbal


def _rot(aa):
    a = float(np.linalg.norm(aa))
    if a < 1e-9:
        return np.eye(3)
    k = np.asarray(aa) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def cymbal_wobble(p, hs, times, chokes=(), open_track=None):
    """(N, 3) axis-angle tilt of cymbal p from a 2-axis damped oscillator. Each hit is an angular impulse about
    n x (strike direction), sized so a hit from rest swings to swing_params' amplitude; the two tilt axes ring
    at f and WOBBLE_SPLIT * f, so the wobble precesses. A choke clamps the cymbal to the pinch pose."""
    n = NORMAL[p]
    e1 = _unit(np.cross(n, _unit(STRIKE[p] - np.array(KIT[p]["c"]))))     # the axis a plain hit swings about
    e2 = np.cross(n, e1)
    dt = min(1 / 240, (times[1] - times[0]) / 4) if len(times) > 1 else 1 / 240
    t0, t1 = times[0] - 0.5, times[-1]
    steps = int(math.ceil((t1 - t0) / dt)) + 1
    ts = t0 + dt * np.arange(steps)
    th, om = np.zeros(2), np.zeros(2)
    out = np.zeros((steps, 2))
    events = sorted((h["t"], h) for h in hs)
    ei = 0
    hold = np.zeros(steps)
    for tc, hd, wgt in chokes:
        hold = np.maximum(hold, np.interp(ts, times, wgt))
    _, _, f, tau = swing_params(p, events[0][1], SWING_PHYS) if events else (0, 0, 1.6, 1.5)
    w1, w2 = 2 * math.pi * f, 2 * math.pi * f * WOBBLE_SPLIT
    wc = 2 * math.pi * CHOKE_F
    for i, t in enumerate(ts):
        while ei < len(events) and events[ei][0] <= t:
            h = events[ei][1]
            a0, a1, f_h, tau_h = swing_params(p, h, SWING_PHYS)
            s = np.asarray(h.get("S", STRIKE[p])) - np.array(KIT[p]["c"])
            ax = _unit(np.cross(n, _unit(s - (s @ n) * n)))
            om += (a0 + a1 * h["vel"] / 127) * 2 * math.pi * f_h * np.array([ax @ e1, ax @ e2])
            ei += 1
        tau_t = tau
        if open_track is not None:      # an open hi-hat rings longer than a closed one
            o = float(np.interp(t, times, open_track))
            tau_t = SWING["hihat"][3] + (SWING["hihat_open"][3] - SWING["hihat"][3]) * o
        acc = -np.array([w1 * w1, w2 * w2]) * th - (2 / tau_t) * om
        if hold[i] > 0:
            acc += hold[i] * (-wc * wc * (th - np.array([CHOKE_DIP, 0.0])) - 2 * wc * om)
        om = om + dt * acc
        th = th + dt * om
        out[i] = th
    a = np.stack([np.interp(times, ts, out[:, 0]), np.interp(times, ts, out[:, 1])], 1)
    return a[:, :1] * e1[None, :] + a[:, 1:] * e2[None, :]


def piece_tilt(hits, times, chokes=None, physics=False, open_track=None):
    """Per piece: (N, 3) axis-angle rotation about the piece centre (edge struck goes down, then swings).
    chokes: {piece: [(t, hold, weight array)]}; a choke kills the swing of every earlier hit on that piece.
    physics: cymbals come from cymbal_wobble() instead of one damped sine per hit."""
    out = {}
    chokes = chokes or {}
    for p in HAND_PIECES:
        hs = [h for h in hits if h["piece"] == p]
        rot = np.zeros((len(times), 3))
        if hs and physics and KIT[p]["kind"] in ("cymbal", "hihat"):
            rot = cymbal_wobble(p, hs, times, chokes.get(p, ()), open_track if KIT[p]["kind"] == "hihat" else None)
        elif hs:
            n = NORMAL[p]
            u = _unit(STRIKE[p] - np.array(KIT[p]["c"]))
            axis = _unit(np.cross(n, u))
            ang = np.zeros(len(times))
            for h in hs:
                a0, a1, f, tau = swing_params(p, h)
                d = times - h["t"]
                m = d > 0
                ring = (a0 + a1 * h["vel"] / 127) * np.exp(-d[m] / tau) * np.sin(2 * np.pi * f * d[m] + 0.25)
                tc = min((c[0] for c in chokes.get(p, ()) if c[0] > h["t"]), default=None)
                if tc is not None:
                    ring *= np.exp(-np.clip(times[m] - tc, 0, None) / CHOKE_TAU)
                ang[m] += ring
            for _, _, wgt in chokes.get(p, ()):
                ang += CHOKE_DIP * wgt
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
# double pedal: the left footboard sits between the main pedal and the hi-hat's, its beater beside the main one
SLAVE_DX, SLAVE_BEATER_DX = -0.21, -0.08
DOUBLE_IOI = 0.17       # s: kicks closer than this go foot to foot on a double pedal
DOUBLE_HOLD = 0.6       # s: the left foot stays on the kick pedal through gaps up to this long
FOOT_MOVE = 0.15        # s to slide the left foot between the pedals
_STD_FEET = (PEDAL_HEEL.copy(), BEATER_PIVOT.copy(), HAT_HEEL.copy())


def use_kit(kit=None):
    """Points the rig at a kit dict (drum_kit.build; None: the standard kit): its pieces, sticking preferences and
    priorities, rebound, and pedals under its kick and hi-hat. Every kit needs a kick, a snare and a hi-hat;
    kit["kick"]["double"] adds the double pedal."""
    global KIT, HAND_PIECES, PREF, PRIORITY, REBOUND, STRIKE, NORMAL, PEDAL_HEEL, BEATER_PIVOT, HAT_HEEL, KICK_X
    kit = kit or drum_kit.STANDARD
    missing = [p for p in ("kick", "snare", "hihat") if p not in kit]
    if missing:
        raise ValueError(f"a kit needs {', '.join(missing)}")
    KIT = {p: dict(k) for p, k in kit.items()}
    HAND_PIECES = [p for p in KIT if p != "kick"]
    PREF = {"R": {}, "L": {}}
    for p in HAND_PIECES:
        PREF["R"][p], PREF["L"][p] = _pref(p, KIT[p])
    PRIORITY = {p: PRIORITY_TYPE[drum_kit.piece_type(p)] for p in HAND_PIECES}
    REBOUND = {p: _rebound(p, KIT[p]) for p in HAND_PIECES}
    STRIKE = {p: strike_point(p) for p in HAND_PIECES}
    NORMAL = {p: _unit(KIT[p]["n"]) for p in KIT}
    _HOME.clear()
    dk = np.array([*(np.array(KIT["kick"]["c"][:2]) - drum_kit.STANDARD["kick"]["c"][:2]), 0.0])
    dh = np.array([*(np.array(KIT["hihat"]["c"][:2]) - drum_kit.STANDARD["hihat"]["c"][:2]), 0.0])
    PEDAL_HEEL, BEATER_PIVOT, HAT_HEEL = _STD_FEET[0] + dk, _STD_FEET[1] + dk, _STD_FEET[2] + dh
    KICK_X = float(PEDAL_HEEL[0])


REACH_EASY, REACH_COST = 0.85, 25.0     # m from a shoulder to a strike point; cost per m beyond it


def _pref(p, k):
    """(right, left) hand cost for piece p: the standard kit's table, as lines in x (the right hand takes the
    right side; the ride is the right hand's), and at least REACH_COST per m the strike point lies beyond
    REACH_EASY from that hand's shoulder (on the standard kit this changes nothing)."""
    if p in STD_PREF["R"] and k == drum_kit.STANDARD.get(p):
        return STD_PREF["R"][p], STD_PREF["L"][p]
    t, x = drum_kit.piece_type(p), k["c"][0]
    if t == "snare":
        return 0.6, 0.0
    if t == "hihat":
        return 0.0, 1.2
    if k["kind"] == "drum":
        r, l = float(np.clip(0.42 - 1.29 * x, 0, 0.6)), float(np.clip(0.28 + 1.29 * x, 0, 0.8))
    else:
        r = float(np.clip(0.25 - 0.6 * x, 0, 0.6))
        l = float(np.clip(1.9 * (x + 0.42), 0, 1.6)) + (1.4 if t == "ride" else 0)
    s = drum_kit.strike_point(k, HIPS[:2], p)
    far = [REACH_COST * max(0.0, float(np.linalg.norm(s - sh)) - REACH_EASY) for sh in drum_kit.SHOULDERS]
    return max(r, far[0]), max(l, far[1])


def _rebound(p, k):
    t = drum_kit.piece_type(p)
    if t in ("tom", "floor"):       # smaller toms are tighter: the standard tom1, tom2 and floor tom by radius
        return round(float(np.interp(k["r"], (0.127, 0.152, 0.20), (0.55, 0.52, 0.46))), 4)
    return {"snare": 0.62, "hihat": 0.50, "ride": 0.55}.get(t, 0.38)


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
SPRING_LEAN = (2.5, 0.8, 0.6)
LEAD_HAND, LEAD_HEAD = 0.035, 0.06
HAND_LIFT, HAND_PULL = 0.30, 0.10   # hand rises this fraction of the tip's height, and draws back
REST_H = 0.06                       # hover height of a resting tip
HAND_CLEAR = 0.26                   # m the grip keeps from its shoulder (physics rig)


def _clear_shoulder(p, sh, side):
    """Grip p moved forward (and a little out) until it is HAND_CLEAR from the shoulder, when it is closer.
    The hand lift rises straight up from the grip home, and for pieces near the hips (snare, floor tom) a big
    lift would end with the fist in the shoulder and the elbow folded flat. One fixed direction keeps the move
    continuous wherever the hand is."""
    v = p - sh
    gap = float(v @ v) - HAND_CLEAR ** 2
    if gap >= 0:
        return p
    fwd = _unit(np.array([0.3 * side, 1.0, 0.0]))
    b = float(v @ fwd)
    return p + (math.sqrt(b * b - gap) - b) * fwd


def _snap_curve(t, T, H, rest, fps):
    """Two poses: the tip is on the head on a hit's frame and at the next stroke's prep height on every other
    frame, with no easing. Returns (height, k, w) like stroke_curve; w jumps to the next piece after the hit."""
    k = int(np.searchsorted(T, t + 0.5 / fps, side="right")) - 1
    if k >= 0 and abs(t - T[k]) < 0.5 / fps:
        return 0.0, k, 0.0
    if k + 1 < len(T):
        return H[k + 1], k, 1.0
    return rest, k, 0.0


def _hand_track(hs, times, hand, start, fps, motion="smooth", dur=strike_dur, arc_max=None):
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
    Hh = np.array([h["H"] if "H" in h else stroke_height(h["vel"]) for h in hs])
    LIFT = np.array([h.get("lift", HAND_LIFT) for h in hs])
    RB = np.array([h.get("rb", 0.5) for h in hs])
    REST = np.array([h.get("rest", REST_H) for h in hs])
    P = [h["piece"] for h in hs]
    S = [(h["S"] if "S" in h else STRIKE[p] + np.array([0, 0, HIHAT_OPEN if h.get("open") else 0.0]))
         + np.asarray(h.get("aim", (0, 0, 0))) for p, h in zip(P, hs)]
    G = [h["G"] if "G" in h else grip_home(p, hand) for p, h in zip(P, hs)]
    NP = [np.asarray(h["N"]) if "N" in h else NORMAL[p] for p, h in zip(P, hs)]
    for j, t in enumerate(times):
        if motion == "snap":
            h, k, w = _snap_curve(t, T, Hh, REST_H, fps)
        else:
            h, k, w = stroke_curve(t, T, Hh, REST, dur=dur, rebound=RB)
        a, b = (max(k, 0), min(k + 1, len(T) - 1)) if k >= 0 else (0, 0)
        if k < 0:
            w = 0.0
        s = S[a] + (S[b] - S[a]) * w
        n = _unit(NP[a] + (NP[b] - NP[a]) * w)
        arc = 0.0 if motion == "snap" else 0.25 * math.sin(math.pi * w) * float(np.linalg.norm(S[b] - S[a]))
        hand_arc = 0.4 * arc
        if arc_max is not None:
            arc, hand_arc = min(arc, arc_max * math.sin(math.pi * w)), 0.0
        g = G[a][0] + (G[b][0] - G[a][0]) * w
        d = _unit(G[a][1] + (G[b][1] - G[a][1]) * w)
        dh = _unit([d[0], d[1], 0.0])
        hh = h + arc
        # a stroke under a cymbal stays under it, rising as the tip moves out past its rim (checked under the
        # point it rises from and under where it would get to: a tilted head's stroke leans toward the player)
        room = min(_overhead_room(s, (P[a], P[b])), _overhead_room(s, (P[a], P[b]), s + hh * n))
        if hh * n[2] > room:
            hh = max(room, 0.0) / max(n[2], 0.3)
        lift = LIFT[a] + (LIFT[b] - LIFT[a]) * w
        TIPT[j] = s + hh * n
        HAND[j] = g + np.array([0, 0, lift * hh + hand_arc]) - HAND_PULL * h * dh
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


def _split_kicks(kicks):
    """Double pedal: (right foot's kicks, left foot's). A kick closer than DOUBLE_IOI after a right-foot kick
    goes to the left foot, so fast runs alternate R L R L; everything else stays on the right. Layered kicks
    (two notes within SLICE_TOL) are one stroke of the same foot."""
    R, L, last, prev = [], [], None, -9.0
    for h in sorted(kicks, key=lambda h: h["t"]):
        if last and h["t"] - prev < SLICE_TOL:
            (R if last == "R" else L).append(h)
            continue
        if last == "R" and h["t"] - prev < DOUBLE_IOI:
            L.append(h); last = "L"
        else:
            R.append(h); last = "R"
        prev = h["t"]
    return R, L


def _slave_weight(T, times):
    """0 (left foot on the hi-hat pedal) .. 1 (on the second kick pedal), per frame: over before a run of left
    kicks, back after it; gaps under DOUBLE_HOLD keep it there."""
    w = np.zeros(len(times))
    runs = []
    for t in T:
        if runs and t - runs[-1][1] < DOUBLE_HOLD:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    for t0, t1 in runs:
        w = np.maximum(w, _smooth((times - (t0 - FOOT_MOVE - 0.08)) / FOOT_MOVE) *
                       (1 - _smooth((times - (t1 + 0.12)) / FOOT_MOVE)))
    return w


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


CHOKE_IN, CHOKE_OUT, CHOKE_LEAN = 0.16, 0.22, 0.6    # s to reach the edge, s to let go, torso lean while pinching
TECH_HEIGHT = {"ghost": 0.045, "flam_grace": 0.035, "cross_stick": 0.07}   # m, the most these strokes lift


def _bump(d, rise=0.10, fall=0.22):
    """0..1 envelope around d = 0: eases in before, decays after."""
    return np.where(d < 0, np.exp(-(d / rise) ** 2), np.exp(-d / fall))


def _style_hit(h, hand, style, physics=False):
    """Stroke height, arm lift, rebound and technique pose for one hand hit (drum_style.py, drum_teacher.py)."""
    p, tech = h["piece"], h.get("tech")
    base = stroke_height
    if physics:
        base = stroke_height_ref
        h["rb"] = REBOUND[p]
        h["H"] = stroke_height_ref(h["vel"])
        h["lift"] = hand_lift_ref(h["vel"])
    if style:
        s = drum_style.at(h["t"], style)
        h["H"] = drum_style.height(h["vel"], s, base)
        hs = h["H"] / STYLE_SCALE if physics else h["H"]
        h["lift"] = drum_style.arm_lift(hs)
        h["rb"] = drum_style.rebound(p, s, REBOUND[p] if physics else 0.5)
        h["lean"] = drum_style.lean_kick(hs, s)
    if tech in TECH_HEIGHT:
        h["H"] = min(h.get("H", base(h["vel"])), TECH_HEIGHT[tech])
    if tech == "cross_stick":
        h["lift"] = 0.0
    pose = technique_pose(p, hand, tech) if tech else None
    if pose:
        h["S"], h["G"] = pose
    if physics and KIT[p]["kind"] == "cymbal" and tech != "choke":
        # a cymbal is struck with a glancing blow from the player's side (stick 20-35 deg to the cymbal in the
        # 1000 fps clip), not from high above it: the prep leans back toward the grip and stays lower
        s = np.asarray(h.get("S", STRIKE[p]), float)
        g = np.asarray(h["G"][0] if "G" in h else grip_home(p, hand)[0], float)
        back = np.array([g[0] - s[0], g[1] - s[1], 0.0])
        h["N"] = _unit(NORMAL[p] + CYM_BACK * _unit(back))
        h["H"] = min(h["H"], CYM_H_MAX)
        h["rest"] = CYM_REST


# rebound: fraction of the next prep height the tip bounces back to. A tight snare throws the stick back; toms and
# the floor tom are slacker; a crash gives way under the stick.
REBOUND = {"snare": 0.62, "tom1": 0.55, "tom2": 0.52, "floor": 0.46, "hihat": 0.50, "ride": 0.55,
           "crash": 0.38, "crash2": 0.38}
GIVE = (0.0015, 0.0035)             # m a drum head dips under the tip: at vel 0, extra at vel 127
GIVE_SIGMA = 0.35                   # dimple width, fraction of the head radius (blender_drums.py)


ARC_MAX = 0.06      # physics: m a moving tip arcs above the straight path between pieces (the hitboxes clear the rest)
CYM_BACK, CYM_H_MAX = 0.8, 0.32     # physics: how far a cymbal prep leans toward the player; its highest prep (m)
CYM_REST = 0.16                     # m: the stick settles this far off a cymbal it struck, clear of its swing
LEAN_MAX = 0.3      # physics: total forward lean saturates here (about 11 cm of head travel, performer.Body.pose)
CROSS_R, CROSS_CLEAR, CROSS_MIN = 0.13, 0.09, 0.10   # m: reach of a hand over a strike point, room left under it,
                                                     # lowest a capped stroke goes


OVERHEAD_CLEAR = 0.035      # m a tip keeps under a cymbal (or the hi-hat) hanging over what it plays ...
OVERHEAD_SLOPE = 1.5        # ... a ceiling that rises this much per m the tip is out past the cymbal's rim
UNDERSIDE = {"hihat": -0.012, "cymbal": -0.005}     # m below its centre plane a cymbal's underside sits (piece_parts)


def _overhead_room(s, skip, at=None):
    """How far (m, up) above surface point s a tip at `at`'s x, y (default s) can rise before meeting a cymbal
    or the hi-hat hanging above, OVERHEAD_CLEAR under its underside; inf if nothing is. Pieces in skip (the
    ones being played) don't count, nor any cymbal s is not under."""
    x, y = (s if at is None else at)[:2]
    room = math.inf
    for p, k in KIT.items():
        if k["kind"] not in UNDERSIDE or p in skip:
            continue
        c, n = np.asarray(k["c"], float), NORMAL[p]
        under = c[2] + (UNDERSIDE[k["kind"]] - n[0] * (x - c[0]) - n[1] * (y - c[1])) / n[2]
        if s[2] > under - OVERHEAD_CLEAR:
            continue
        out = max(0.0, math.hypot(x - c[0], y - c[1]) - k["r"])
        room = min(room, under - OVERHEAD_CLEAR - s[2] + OVERHEAD_SLOPE * out)
    return room


def _under_other_hand(tracks, times):
    """Crossed hands (the right over the hi-hat, the left on the snare): a stroke that rises under the other hand's
    grip is played lower, CROSS_CLEAR under it, instead of driving the stick into that fist. Caps H in place;
    True if any stroke changed."""
    changed = False
    for hand, other in (("R", "L"), ("L", "R")):
        grip = tracks[other][2]
        for h in tracks[hand][0]:
            if h.get("tech") in TECH_HEIGHT:
                continue
            p = h["piece"]
            s = np.asarray(h["S"] if "S" in h else STRIKE[p], float)
            m = (times > h["tf"] - 0.3) & (times <= h["tf"])
            near = m & (np.hypot(grip[:, 0] - s[0], grip[:, 1] - s[1]) < CROSS_R)
            if not near.any():
                continue
            cap = max(float(grip[near, 2].min() - s[2]) - CROSS_CLEAR, CROSS_MIN)
            if h["H"] > cap:
                h["H"], h["crossed"] = cap, True
                changed = True
    return changed


def _follow_surfaces(tracks, tilt, times, start):
    """Cymbal hits land where the cymbal is at that moment, not where it rests: each hand's tip and grip targets
    get the offset of the moving strike point, eased from hit to hit. Drum hits sink GIVE into the head on their
    frame. Returns {frame index: {piece: [give, x, y, z]}} for the renderer's head dimple."""
    give = {}
    for hand, tr in tracks.items():
        keep, TIPT, HAND = tr[0], tr[1], tr[2]
        if not keep:
            continue
        T, D = [], []
        for h in keep:
            p = h["piece"]
            j = int(np.argmin(np.abs(times - h["tf"])))
            s = h["S"] if "S" in h else STRIKE[p] + np.array([0, 0, HIHAT_OPEN if h.get("open") else 0.0])
            s = np.asarray(s, float) + np.asarray(h.get("aim", (0, 0, 0)))
            d = np.zeros(3)
            if KIT[p]["kind"] in ("cymbal", "hihat") and p in tilt:
                c = np.array(KIT[p]["c"], float)
                d = c + _rot(tilt[p][j]) @ (s - c) - s
            elif KIT[p]["kind"] == "drum" and not h.get("tech") == "cross_stick":
                g = GIVE[0] + GIVE[1] * h["vel"] / 127
                TIPT[j] -= g * NORMAL[p]
                h["give"] = g
                give.setdefault(j, {})[p] = [round(g, 4)] + [round(float(x), 4) for x in s]
            h["S_hit"] = s + d
            T.append(h["tf"]); D.append(d)
        T, D = np.array(T), np.array(D)
        k = np.clip(np.searchsorted(T, times, side="right") - 1, 0, len(T) - 1)
        k1 = np.minimum(k + 1, len(T) - 1)
        u = np.where(k1 > k, (times - T[k]) / np.maximum(T[k1] - T[k], 1e-6), 0.0)
        u = _smooth(np.clip(u, 0, 1))[:, None]
        off = D[k] + (D[k1] - D[k]) * u
        TIPT += off
        HAND += off
    return give


def _place_chokes(hand_hits, tracks, times, blocked=()):
    """Moves a free hand onto the cymbal's edge for every choke. Returns {piece: [(t, hold, weight)]}, the torso
    lean and twist toward the cymbal, and each hand's choke weight (0..1) per frame. blocked: (hand, t0, t1)
    spans a hand is away (a stick toss)."""
    out, lean, twist = {}, np.zeros(len(times)), np.zeros(len(times))
    on = {"R": np.zeros(len(times)), "L": np.zeros(len(times))}
    used = {"R": [], "L": []}
    for hand, a, b in blocked:
        used[hand].append((a, b))
    for h in sorted((h for h in hand_hits if h.get("tech") == "choke" and h.get("hand")), key=lambda h: h["t"]):
        tc, hold, p = h["choke"]["t"], h["choke"]["hold"], h["piece"]
        lo, hi = tc - CHOKE_IN - 0.06, tc + hold + CHOKE_OUT + 0.06
        chosen = None
        for hand in (h["hand"], "L" if h["hand"] == "R" else "R"):
            busy = any(lo < k["tf"] < hi for k in tracks[hand][0] if k is not h)
            busy = busy or any(a < hi and lo < b for a, b in used[hand])
            if not busy:
                chosen = hand
                break
        if chosen is None:
            h["choke_skipped"] = "both hands busy"
            continue
        used[chosen].append((lo, hi))
        wgt = _smooth((times - (tc - CHOKE_IN)) / CHOKE_IN) * (1 - _smooth((times - (tc + hold)) / CHOKE_OUT))
        grip, tip = choke_pose(p, chosen)
        _, TIPT, HAND, HH, cur, _ = tracks[chosen]
        HAND += (grip - HAND) * wgt[:, None]
        TIPT += (tip - TIPT) * wgt[:, None]
        TIPT[:, 2] += 0.5 * wgt * (1 - wgt)     # arc over the hi-hat on the way to and from the edge
        HH += (0.3 - HH) * wgt
        for j in np.nonzero(wgt > 0.5)[0]:
            cur[j] = p
        out.setdefault(p, []).append((tc, hold, wgt))
        lean += CHOKE_LEAN * wgt
        twist += -math.atan2(grip[0], grip[1] - HIPS[1]) * wgt
        on[chosen] = np.maximum(on[chosen], wgt)
        h["choke_hand"], h["choke_win"] = chosen, (tc - CHOKE_IN, tc + hold + CHOKE_OUT)
    return out, lean, twist, on


REACH_LEAN = 7.0    # physics: lean (performer units) per m a strike point lies beyond REACH_EASY from the shoulder


def _reach_lean(hand_hits, times):
    """Torso lean that brings the shoulder toward strike points past easy reach, eased in before each such hit."""
    lean = np.zeros(len(times))
    for h in hand_hits:
        if not h.get("hand") or h.get("tech") == "choke":
            continue
        s = np.asarray(h["S"] if "S" in h else STRIKE[h["piece"]], float)
        sh = drum_kit.SHOULDERS[0 if h["hand"] == "R" else 1]
        deficit = float(np.linalg.norm(s - sh)) - REACH_EASY
        if deficit > 0:
            lean = np.maximum(lean, REACH_LEAN * deficit * _bump(times - h["tf"], 0.15, 0.2))
    return lean


def _lesson(hits, start, end, beats, downbeats):
    bts, dbs, period = drum_teacher._grid(beats, downbeats)
    out = []
    for h in sorted(hits, key=lambda h: h["t"]):
        if "tech" not in h or not start <= h["t"] < end:
            continue
        bar, beat = drum_teacher.where(h["t"], bts, dbs, period)
        e = {"t": round(h["t"], 4), "bar": bar, "beat": round(beat, 2), "piece": h["piece"], "vel": h["vel"],
             "tech": h["tech"], "why": h["why"], "hand": h.get("choke_hand") or h.get("hand")}
        if h.get("choke_skipped"):
            e["skipped"] = h["choke_skipped"]
        elif h["piece"] in HAND_PIECES and not h.get("hand"):
            e["skipped"] = "no free hand"
        out.append(e)
    return out


def animate(hits, fps=24, start=0.0, dur=None, beats=None, downbeats=None, schedule=((0.0, "groove"),), seed=0,
            motion="smooth", emotion=None, sliders=None, physics=True, kit=None, tosses=None):
    """motion: "smooth" (the rules above), "snap" (hands jump between a prep pose and the hit pose, no easing,
    no springs), or "loose" (smooth, but every hand hit lands up to LOOSE_FRAMES early or late and up to
    LOOSE_AIM m off its strike point). snap and loose exist to test how exact a blockout has to be.
    emotion: a drum_style.py timeline ("0:calm,8:intense"); it sets stroke heights, arm lift, torso lean,
    cymbal follow-through, and the head schedule (replacing `schedule`). sliders: {slider: value} on top.
    Hits annotated by drum_teacher.teach() get their techniques, and the result carries a "lesson".
    physics: hitboxes (drum_collide.py moves sticks, fists and arms out of the kit, the body and each other),
    cymbals as 2-axis damped oscillators that the stick meets where they are, per-surface rebound and drum-head
    give. False turns all of that off, for comparison; the hands and grip homes are the same.
    kit: a kit dict (drum_kit.build) instead of the standard kit; hits are pointed at its pieces (drum_kit.remap).
    tosses: stick tosses and catches (drum_toss.py); hand strokes near one go to the other hand.
    The module is back on the standard kit when this returns, so a kit never carries over to later calls."""
    use_kit(kit)
    try:
        return _animate(hits, fps, start, dur, beats, downbeats, schedule, seed, motion, emotion, sliders, physics,
                        kit, tosses)
    finally:
        use_kit(None)


def _animate(hits, fps, start, dur, beats, downbeats, schedule, seed, motion, emotion, sliders, physics, kit,
             tosses):
    if motion not in MOTIONS:
        raise ValueError(f"motion must be one of {MOTIONS}")
    if kit is not None:
        hits = drum_kit.remap(hits, KIT)
    style = drum_style.parse(emotion, sliders) if emotion else None
    if style:
        schedule = drum_style.head_schedule(style)
    rng = np.random.default_rng(seed + 1000)
    hits = [dict(h) for h in hits]
    end = start + dur if dur else max(h["t"] for h in hits) + 1.5
    hits = [h for h in hits if start - 2.0 < h["t"] < end + 1.0]
    pre = 2.0
    times = (start - pre) + np.arange(int(round((end - start + pre) * fps))) / fps
    N = len(times)

    tosses = [ts for ts in tosses or () if ts["t1"] > start and ts["t0"] < end]
    drum_toss.check(tosses)
    for ts in tosses:
        a, b = drum_toss.window(ts)
        for h in hits:
            if h["piece"] in HAND_PIECES and a < h["t"] < b:
                h["hand_lock"] = "L" if ts["hand"] == "R" else "R"
    hand_hits = sticking(hits)
    for ts in tosses:
        a, b = drum_toss.window(ts)
        busy = [round(h["t"], 3) for h in hand_hits if h.get("hand") == ts["hand"] and a < h["t"] < b]
        if busy:
            raise ValueError(f"toss at {ts['t0']}: hand {ts['hand']} still has strokes at {busy} "
                             "(two hits at once there)")
    tracks = {}
    track_kw = {"dur": strike_dur_ref, "arc_max": ARC_MAX} if physics else {}
    for hand in ("R", "L"):
        hs = [h for h in hand_hits if h.get("hand") == hand]
        for h in hs:
            h["tf"] = _snap(h["t"], start, fps)
            if motion == "loose":
                h["tf"] += int(rng.integers(-LOOSE_FRAMES, LOOSE_FRAMES + 1)) / fps
                r, ang = LOOSE_AIM * math.sqrt(rng.random()), 2 * math.pi * rng.random()
                h["aim"] = (r * math.cos(ang), r * math.sin(ang), 0.0)
            _style_hit(h, hand, style, physics)
        hs.sort(key=lambda h: h["tf"])
        keep = []
        for h in hs:     # one stroke per frame per hand: a faster repeat is merged into the stroke before it
            if keep and h["tf"] <= keep[-1]["tf"]:
                keep[-1]["vel"] = max(keep[-1]["vel"], h["vel"])
                h["hand"] = None
                continue
            keep.append(h)
        if physics:
            for a, b in zip(keep, keep[1:]):
                if b.get("tech") not in TECH_HEIGHT:
                    b["H"] *= min(1.0, (b["tf"] - a["tf"]) / DENSE_IOI) ** 0.5
        tracks[hand] = (keep,) + _hand_track(keep, times, hand, start, fps, motion, **track_kw)
    if physics and motion != "snap" and _under_other_hand(tracks, times):
        for hand in ("R", "L"):
            tracks[hand] = (tracks[hand][0],) + _hand_track(tracks[hand][0], times, hand, start, fps, motion,
                                                            **track_kw)

    kicks = [h for h in hits if h["piece"] == "kick"]
    double = bool(KIT["kick"].get("double"))
    if double:
        kicks, kicks_l = _split_kicks(kicks)
        KP2, KLIFT2, kick_T2 = _kick_track(kicks_l, times, start, fps)
        ON2 = _slave_weight(kick_T2, times)
    KP, KLIFT, kick_T = _kick_track(kicks, times, start, fps)
    beats = np.asarray(beats if beats is not None else np.arange(0, end + 1, 0.5))
    downbeats = np.asarray(downbeats if downbeats is not None else beats[::4])
    hat_hits = hits
    if double:      # the left foot is on the second kick pedal: no hi-hat pedal chick then
        at = lambda t: ON2[int(np.clip(round((t - times[0]) * fps), 0, N - 1))]
        hat_hits = [h for h in hits if h["piece"] != "hihat_pedal" or at(h["t"]) < 0.5]
    OPEN, HEEL_L = _hat_track(hat_hits, times, start, fps, beats)
    chokes, LEAN, TWIST, CHOKING = _place_chokes(hand_hits, tracks, times,
                                                 [(ts["hand"], *drum_toss.window(ts)) for ts in tosses])
    if physics:     # lean in to reach a far piece: like a choke's lean, it is not capped by LEAN_MAX
        LEAN += _reach_lean(hand_hits, times)
    CHOKE_LEAN_T = LEAN.copy()
    tilt = piece_tilt([h for h in hits if h.get("hand")], times, chokes, physics, OPEN)
    give = _follow_surfaces(tracks, tilt, times, start) if physics else {}
    for h in hand_hits:
        if h.get("lean") and h.get("hand"):
            LEAN += h["lean"] * _bump(times - h["tf"])

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
    if LEAN.any():
        HEAD[:, 3] += np.clip(dynamics.filter_track(LEAN[:, None], fps, *SPRING_LEAN)[:, 0], 0, 0.6)
    if TWIST.any():     # head turns to the cymbal; the torso follows with 0.3 of it (performer.Body.pose)
        HEAD[:, 1] += dynamics.filter_track(TWIST[:, None], fps, *SPRING_LEAN)[:, 0]
    if physics:         # the face stays out of the air the sticks travel through, except to reach a choke
        cl = dynamics.filter_track(CHOKE_LEAN_T[:, None], fps, *SPRING_LEAN)[:, 0]
        x = HEAD[:, 3] - cl
        HEAD[:, 3] = np.where(x > 0, LEAN_MAX * np.tanh(x / LEAN_MAX), x) + cl
    body = performer.Body(hips=HIPS, base_gaze_pitch=0.75)
    LEG = dynamics.filter_track(np.stack([KLIFT, HEEL_L], 1), fps, *SPRING_LEG)
    if double:
        LEG2 = dynamics.filter_track(KLIFT2[:, None], fps, *SPRING_LEG)[:, 0]
        slave_heel = PEDAL_HEEL + np.array([SLAVE_DX, 0.0, 0.0])

    frames = []
    j0 = int(round(pre * fps))
    hat_dir = HAT_DIR
    for j in range(j0, N):
        t = times[j]
        fr = {"t": round(float(t - start), 4), "hands": {}}
        wrists, fwds = {}, {}
        pitch, yaw, roll, lean = HEAD[j]
        if physics:
            shoulders = body.pose(pitch, yaw, roll, lean, {})["shoulders"]
        for hand in ("R", "L"):
            tipt, hh = tracks[hand][1][j], tracks[hand][3][j]
            hs = Hs[hand][j]
            if physics:
                hs = _clear_shoulder(hs, np.array(shoulders[hand]), 1 if hand == "R" else -1)
            d = _unit(tipt - hs)
            free = hs + TIP * d
            c = math.exp(-max(hh, 0.0) / 0.012)
            tip = free + c * (tipt - free)
            grip = tip - TIP * d
            butt = grip - (STICK - TIP) * d
            wrist = drum_hands.wrist(grip, d, hand)
            wrists[hand], fwds[hand] = wrist, drum_hands.forward(d, hand)
            fr["hands"][hand] = {"tip": tip, "grip": grip, "butt": butt, "wrist": wrist, "piece": tracks[hand][4][j],
                                 "pin": c, "side": 1 if hand == "R" else -1}
        b = body.pose(pitch, yaw, roll, lean, wrists, hand_fwd=fwds)
        for hand in ("R", "L"):
            if not physics and (CHOKING[hand][j] > 0 or style):   # past arm's length: stop at full reach
                hd, sh = fr["hands"][hand], np.array(b["shoulders"][hand])
                v = hd["wrist"] - sh
                over = float(np.linalg.norm(v)) - (performer.UPPER_ARM + performer.FOREARM - 0.002)
                if over > 0:
                    shift = over * _unit(v)
                    for key in ("tip", "grip", "butt", "wrist"):
                        hd[key] = hd[key] - shift
                    b["elbows"][hand] = performer.arm_ik(sh, hd["wrist"], 1 if hand == "R" else -1,
                                                         fwds[hand]).tolist()
            fr["hands"][hand]["elbow"] = b["elbows"][hand]
            fr["hands"][hand]["shoulder"] = b["shoulders"][hand]
        # feet
        kp = KP[j]
        k_ang = 0.08 + 0.20 * kp
        k_toe, k_ball, k_heel, k_ank = foot_points(PEDAL_HEEL, (0, 1), k_ang, 0.20, 0.05 + LEG[j, 0])
        op = OPEN[j]
        h_ang = 0.07 + 0.16 * op
        h_toe, h_ball, h_heel, h_ank = foot_points(HAT_HEEL, hat_dir, h_ang, 0.20, LEG[j, 1])
        if double:
            k2_ang = 0.08 + 0.20 * KP2[j]
            w = float(ON2[j])
            if w > 0:       # the left foot slides over to the second kick pedal, lifting on the way
                s_pts = foot_points(slave_heel, (0, 1), k2_ang, 0.20, 0.05 + LEG2[j])
                up = np.array([0, 0, 0.24 * w * (1 - w)])
                h_toe, h_ball, h_heel, h_ank = (a + w * (b - a) + up for a, b in
                                                zip((h_toe, h_ball, h_heel, h_ank), s_pts))
        feet = {}
        for side, (toe, ball, heel, ank) in (("R", (k_toe, k_ball, k_heel, k_ank)), ("L", (h_toe, h_ball, h_heel, h_ank))):
            sx = 1 if side == "R" else -1
            hip = HIPS + np.array([sx * 0.12, 0.06, -0.05])
            feet[side] = {"hip": hip, "knee": leg_ik(hip, ank, sx), "ankle": ank, "heel": heel, "ball": ball, "toe": toe}
        fr["feet"] = feet
        fr["kick"] = {"board": round(k_ang, 4), "beater": round(beater_angle(kp), 4)}
        if double:
            fr["kick"].update(board2=round(k2_ang, 4), beater2=round(beater_angle(KP2[j]), 4))
        fr["hihat"] = {"open": round(float(op), 3), "board": round(h_ang, 4)}
        fr["tilt"] = {p: tilt[p][j] for p in tilt if np.any(tilt[p][j])}
        if j in give:
            fr["give"] = give[j]
        fr["body"] = b
        frames.append(fr)

    choke_list = [{"piece": h["piece"], "hand": h["choke_hand"], "t0": round(h["choke_win"][0] - start, 4),
                   "t1": round(h["choke_win"][1] - start, 4)} for h in hand_hits if h.get("choke_hand")]
    resolved = None
    if physics:
        allow = drum_collide.allowed_windows(
            [{"tf": h["tf"] - start, "hand": h["hand"], "piece": h["piece"], "tech": h.get("tech")}
             for h in hand_hits if h.get("hand")], choke_list, fps)
        resolved = drum_collide.resolve(frames, KIT, HIHAT_OPEN, allow, fps, STICK, TIP,
                                        performer.UPPER_ARM + performer.FOREARM, performer.arm_ik)
    for fr in frames:
        for hd in fr["hands"].values():
            hd.pop("pin"), hd.pop("side")
    frames = [_round(fr) for fr in frames]
    toss_info = [drum_toss.apply(frames, ts, fps, start, STICK, TIP) for ts in tosses]
    if toss_info:
        if physics:
            drum_collide.place_elbows(frames, drum_collide.Kit(KIT, HIHAT_OPEN), fps)
        frames = [_round(fr) for fr in frames]
    drum_hands.add(frames, [dict(ts, t0=ts["t0"] - start, t1=ts["t1"] - start) for ts in tosses])

    drawn = []
    for h in hand_hits:
        if h.get("hand") and start <= h["tf"] < end:
            d = {"t": round(h["t"] - start, 4), "tf": round(h["tf"] - start, 4), "piece": h["piece"], "vel": h["vel"],
                 "hand": h["hand"]}
            if "tech" in h:
                d["tech"] = h["tech"]
            if "S_hit" in h:
                d["s"] = [round(float(x), 5) for x in h["S_hit"]]
            elif "S" in h:
                d["s"] = [round(float(x), 5) for x in h["S"] + np.asarray(h.get("aim", (0, 0, 0)))]
            if h.get("give"):
                d["give"] = round(h["give"], 4)
            if style:
                d["tier"] = drum_style.tier(h["H"] / STYLE_SCALE if physics else h["H"])[0]
            drawn.append(d)
    extra = {}
    if style:
        extra["emotion"] = [{"t": t, **sl, "head": head} for t, sl, head in style]
    if any("tech" in h for h in hits):
        extra["lesson"] = _lesson(hits, start, end, beats, downbeats)
    if choke_list:
        extra["chokes"] = choke_list
    if resolved:
        extra["hitbox_push"] = resolved
    if toss_info:
        extra["tosses"] = toss_info
    kick_pedal = {"heel": PEDAL_HEEL.tolist(), "len": BOARD_L, "beater_pivot": BEATER_PIVOT.tolist(),
                  "beater_len": BEATER_L}
    if double:
        kick_pedal.update(heel2=slave_heel.tolist(),
                          beater_pivot2=(BEATER_PIVOT + np.array([SLAVE_BEATER_DX, 0, 0])).tolist())
        extra["kicks_left"] = [round(float(x) - start, 4) for x in kick_T2 if start <= x < end]
    stands = [[n, _round(a), _round(b), r] for n, a, b, r in drum_collide.stands(KIT)]
    kit = {p: {**{k: v for k, v in KIT[p].items()}, "strike_pt": STRIKE.get(p, np.array(KIT[p]["c"])).tolist(),
               "n": NORMAL[p].tolist()} for p in KIT}
    return {**extra, "fps": fps, "start": start, "kit": kit, "stick": [STICK, TIP], "hihat_open": HIHAT_OPEN,
            "kick_pedal": kick_pedal, "stands": stands,
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
            s = np.array(h["s"] if "s" in h else kit[h["piece"]]["strike_pt"])
            if h["piece"] == "hihat" and "s" not in h and fr[j]["hihat"]["open"] > 0.5:
                s = s + np.array([0, 0, anim["hihat_open"]])
            if "give" in h:
                s = s - h["give"] * np.array(kit[h["piece"]]["n"])
            errs.append(float(np.linalg.norm(np.array(fr[j]["hands"][h["hand"]]["tip"]) - s)))
    pen, reach = 0, 0.0
    for f in fr:
        for hand, hd in f["hands"].items():
            tip = np.array(hd["tip"])
            for p, k in kit.items():
                if k["kind"] in ("kick",):
                    continue
                c, n = np.array(k["c"]), np.array(k["n"])
                if p in f.get("tilt", {}):
                    n = _rot(f["tilt"][p]) @ n
                d = float((tip - c) @ n)
                rad = float(np.linalg.norm((tip - c) - d * n))
                deep = k.get("depth", 0.02)
                if rad < k["r"] * 0.95 and -deep < d < -0.006:
                    pen += 1
            reach = max(reach, float(np.linalg.norm(np.array(hd["wrist"]) - np.array(hd["shoulder"]))))
    kick_ok = 0
    for key, ts in (("beater", anim["kicks"]), ("beater2", anim.get("kicks_left", ()))):
        top = max(f["kick"][key] for f in fr) if ts else 0
        for t in ts:
            j = int(round(t * fps))
            if 0 <= j < len(fr) and abs(fr[j]["kick"][key] - top) < 1e-3:
                kick_ok += 1
    return {"hand_hits": len(errs), "contact_err_mm_max": round(1000 * max(errs), 2) if errs else None,
            "contact_err_mm_median": round(1000 * float(np.median(errs)), 2) if errs else None,
            "tip_penetrating_frames": pen, "max_reach_m": round(reach, 3),
            "arm_length_m": performer.UPPER_ARM + performer.FOREARM,
            "kicks": len(anim["kicks"]) + len(anim.get("kicks_left", ())), "kicks_beater_on_head": kick_ok,
            **({"kicks_left_foot": len(anim["kicks_left"])} if "kicks_left" in anim else {}),
            "sticking": {hd: sum(h["hand"] == hd for h in anim["hits"]) for hd in ("R", "L")},
            **({"techniques": _count(e["tech"] for e in anim["lesson"] if not e.get("skipped")),
                "skipped": _count(e["tech"] for e in anim["lesson"] if e.get("skipped"))} if "lesson" in anim else {}),
            **({"tiers": _count(h["tier"] for h in anim["hits"])} if "emotion" in anim else {})}


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


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
    ap.add_argument("--emotion", help="drum_style.py timeline, e.g. '0:calm,8:intense' (replaces --schedule)")
    for k in drum_style.SLIDERS:
        ap.add_argument(f"--{k}", type=float, help=f"{k} slider 0..1 on top of every emotion preset")
    ap.add_argument("--teach", action="store_true", help="pick techniques with drum_teacher.py and print the lesson")
    ap.add_argument("--hits-out", help="write the hit list (for drumsynth / scoring) as JSON, named for the "
                                       "standard kit (pass drumsynth the same --kit)")
    ap.add_argument("--no-physics", action="store_true",
                    help="no hitboxes, cymbal dynamics, surface rebound or head give (for comparison)")
    ap.add_argument("--clips", action="store_true", help="also print drum_collide.py's clip report")
    ap.add_argument("--kit", help=f"drum_kit.py preset ({', '.join(drum_kit.PRESETS)}) or kit spec JSON")
    ap.add_argument("--toss", help="stick tosses (drum_toss.py), 'HAND:release_s:catch_s[:spins],...'")
    a = ap.parse_args()
    kit = None
    if a.kit:
        kit, notes = drum_kit.load(a.kit)
        for n in notes:
            print(n)
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
    if a.teach:
        drum_teacher.teach(hits, bts, dbs)
    if a.hits_out:
        json.dump({"hits": hits}, open(a.hits_out, "w"))
    sched = [(float(t), st) for t, st in (x.split(":") for x in a.schedule.split(","))]
    sliders = {k: getattr(a, k) for k in drum_style.SLIDERS if getattr(a, k) is not None}
    anim = animate(hits, a.fps, a.start, a.dur, bts, dbs, sched, a.seed, a.motion, a.emotion, sliders,
                   physics=not a.no_physics, kit=kit, tosses=drum_toss.parse(a.toss))
    json.dump(anim, open(a.out, "w"))
    for ts in anim.get("tosses", ()):
        print(f"toss {ts}")
    if "lesson" in anim:
        print(drum_teacher.sheet(anim["lesson"]))
    print(f"{len(anim['frames'])} frames -> {a.out}")
    print(json.dumps(check(anim)))
    if a.clips:
        r = drum_collide.report(anim)
        print(json.dumps({k: r[k] for k in ("frames", "frames_clipping", "pairs")}))

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
import itertools, json, math, os

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


# Hit pads (y range from the white key fronts, metres). A key is a lever pivoting at the back: struck near
# its front edge it goes down with the least force; struck mid-key it is heavier and tires the player. So
# pianists strike with the fingertip in a small zone near the front of each key.
HITPAD = {"white": (0.010, 0.026), "black": (WHITE_L - BLACK_L + 0.004, WHITE_L - BLACK_L + 0.016)}


def contact(p, near_black=False, thumb=False):
    """Fingertip strike point on a key's surface: inside its hit pad (white keys at the back of the pad
    when the hand is up among the black keys, still in front of them; a thumb takes a black key on its
    front lip, since it reaches from further back)."""
    if is_black(p):
        lo, hi = HITPAD["black"]
        return np.array([key_x(p), lo + 0.002 if thumb else 0.5 * (lo + hi), BLACK_H])
    lo, hi = HITPAD["white"]
    return np.array([key_x(p), hi - 0.002 if near_black else 0.5 * (lo + hi), 0.0])


def leverage(p, y):
    """Relative force to press key p at depth y (1 = at the front edge): the lever shortens toward the pivot."""
    y0 = WHITE_L - BLACK_L if is_black(p) else 0.0
    arm = 0.40                                   # key-lever length from front edge to the balance point
    return arm / max(arm - (y - y0), 0.05)


# ---------------------------------------------------------------- collision: the keyboard as a heightfield
_WHITES = [p for p in range(LOW, HIGH + 1) if not is_black(p)]
_BLACKS = np.array([p for p in range(LOW, HIGH + 1) if is_black(p)])
_BLACK_X = np.array([key_x(p) for p in _BLACKS])
RAIL_Z, FALLBOARD_Y, FALLBOARD_Z = -0.012, 0.153, 0.055


def surface(x, y, depth):
    """Top of whatever is under (x, y): a white key, a black key, the front rail or the fallboard.
    depth = {pitch: 0..1} lowers pressed keys (they pivot at the back, so the front drops the most)."""
    if y < 0:
        return RAIL_Z
    if y > FALLBOARD_Y:
        return FALLBOARD_Z
    wi = int(np.clip((x + _MID_C) // WHITE_W, 0, len(_WHITES) - 1))
    pw = _WHITES[wi]
    z = -depth.get(pw, 0.0) * KEY_TRAVEL * (WHITE_L - y) / WHITE_L
    if y >= WHITE_L - BLACK_L:
        k = int(np.argmin(np.abs(_BLACK_X - x)))
        if abs(_BLACK_X[k] - x) < BLACK_W / 2:
            pb = int(_BLACKS[k])
            z = max(z, BLACK_H - depth.get(pb, 0.0) * KEY_TRAVEL * (WHITE_L - y) / BLACK_L)
    return z


def surface_under(x, y, r, depth):
    """Highest surface under a disc of radius r (a finger pad seen from above)."""
    return max(surface(x + dx, y + dy, depth) for dx, dy in ((0, 0), (0.8 * r, 0), (-0.8 * r, 0), (0, 0.8 * r), (0, -0.8 * r)))


# ---------------------------------------------------------------- hand model
# finger 0 = thumb ... 4 = pinky. Home tip offsets: a five-finger position over adjacent white keys.
# Home tip offsets across the hand, from reference footage of pianists: the fingers fan out from the wrist
# and the thumb sits well out to the side, so a relaxed playing hand spans about a seventh to an octave.
HOME = np.array([-0.078, -0.036, 0.000, 0.030, 0.058])
SEG = np.array([[0.040, 0.032, 0.026],   # thumb: metacarpal, proximal, distal
                [0.044, 0.025, 0.019],
                [0.048, 0.028, 0.020],
                [0.045, 0.027, 0.020],
                [0.036, 0.020, 0.018]])
# knuckle (MCP; thumb CMC) offsets from the hand centre at the knuckle line, right hand
KNUCKLE = np.array([[-0.046, -0.040, -0.022],      # thumb CMC: low, out to the side
                    [-0.038, 0.000, 0.0],               # knuckle breadth ~8 cm, as on an adult hand
                    [-0.013, 0.005, 0.0],
                    [0.012, 0.001, 0.0],
                    [0.036, -0.010, -0.004]])
FINGER_R = np.array([(0.0115, 0.0105, 0.0095, 0.0085), (0.0095, 0.0088, 0.0080, 0.0070),
                     (0.0098, 0.0090, 0.0082, 0.0072), (0.0093, 0.0085, 0.0078, 0.0068),
                     (0.0085, 0.0078, 0.0070, 0.0062)])
KNUCKLE_Z = 0.064          # knuckle line height over white key tops
HAND_MODEL = "mannequin"
MPFB_JSON = os.path.join(os.path.dirname(__file__), "hand_model", "mpfb_hands.json")


def hand_frame(wrist, mcps, side):
    """Hand frame from the wrist and the four finger MCPs: origin at the knuckle-line centre, rows = across
    (thumb side -> pinky side), forward (wrist -> knuckles), up (back of the hand). Anatomical, so the same
    local coordinates serve both hands. blender_piano.py builds the identical frame from the animated joints."""
    mcps = np.asarray(mcps, float)
    c = mcps.mean(0)
    fwd = c - np.asarray(wrist, float); fwd /= np.linalg.norm(fwd)
    ac = mcps[3] - mcps[0]; ac -= fwd * (ac @ fwd); ac /= np.linalg.norm(ac)
    up = np.cross(ac, fwd) * (1 if side == "R" else -1)
    return c, np.stack([ac, fwd, up])


def mpfb_frame(B, side):
    """hand_frame of an MPFB rig hand at rest."""
    return hand_frame(B["wrist"]["head"], [B[f"finger{f}-1"]["head"] for f in range(2, 6)], side)


def _swing(a, b):
    """Rotation matrix taking unit vector a onto unit vector b by the shortest arc (no twist)."""
    v, cth = np.cross(a, b), float(a @ b)
    if cth < -0.999999:
        return -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + cth)


# Thumb joints (radians). The CMC is a saddle joint: its metacarpal swings within a cone around the rig's rest
# direction (flexion/extension ~50 deg and ab/adduction ~45 deg of total arc, so about +-30 deg here); the MCP and
# IP are hinges on an axis carried by the metacarpal.
THUMB_CONE = math.radians(30)
THUMB_MCP = (math.radians(-10), math.radians(55))
THUMB_IP = (math.radians(-10), math.radians(80))
THUMB = None                 # rest data, set by set_hand_model("mpfb")


def _swings(a, B):
    """_swing(a, b) for every row b of B, as a (K, 3, 3) stack."""
    v = np.cross(a, B)
    cth = B @ a
    K = np.zeros((len(B), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    return np.eye(3)[None] + K + (K @ K) / (1 + np.maximum(cth, -0.999999))[:, None, None]


def _cone_dirs(n0, half, k=600, seed=0):
    """k unit vectors spread over the spherical cap of half-angle `half` around n0 (n0 itself first)."""
    rng = np.random.default_rng(seed)
    z = 1 - rng.random(k - 1) * (1 - math.cos(half))
    ph = rng.random(k - 1) * 2 * math.pi
    r = np.sqrt(1 - z * z)
    cap = np.vstack([[0, 0, 1], np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)])
    return cap @ _swing(np.array([0, 0, 1.0]), n0).T


def _thumb_fk(m, th1, th2):
    """Local MCP, IP, tip offsets from the CMC for metacarpal directions m (K,3) and hinge angles (K,)."""
    T = THUMB
    L1, L2, L3 = T["seg"]
    ax = np.einsum("kij,j->ki", T["swings"](m), T["axis"])        # hinge axis carried by the metacarpal
    e2 = np.cross(ax, m)                                           # direction a positive (flexing) rotation moves m
    d2 = np.cos(th1)[:, None] * m + np.sin(th1)[:, None] * e2
    d3 = np.cos(th1 + th2)[:, None] * m + np.sin(th1 + th2)[:, None] * e2
    mcp = L1 * m
    ip = mcp + L2 * d2
    return mcp, ip, ip + L3 * d3


def _thumb_solve(t, dirs):
    """Best thumb pose reaching local target t (from the CMC) over candidate metacarpal directions."""
    T = THUMB
    L1, L2, L3 = T["seg"]
    m = dirs
    ax = np.einsum("kij,j->ki", T["swings"](m), T["axis"])
    e2 = np.cross(ax, m)
    v = t[None, :] - L1 * m
    v = v - ax * np.sum(v * ax, 1, keepdims=True)                   # into the hinge plane
    D = np.clip(np.linalg.norm(v, axis=1), abs(L2 - L3) + 1e-6, L2 + L3 - 1e-6)
    phi = np.arctan2(np.sum(v * e2, 1), np.sum(v * m, 1))
    th2 = np.arccos(np.clip((D * D - L2 * L2 - L3 * L3) / (2 * L2 * L3), -1, 1))
    th1 = phi - np.arctan2(L3 * np.sin(th2), L2 + L3 * np.cos(th2))
    th1 = np.clip(th1, *THUMB_MCP)
    th2 = np.clip(th2, *THUMB_IP)
    _, _, tip = _thumb_fk(m, th1, th2)
    dev = np.arccos(np.clip(m @ T["n0"], -1, 1))
    miss = np.linalg.norm(tip - t, axis=1)
    cost = miss ** 2 + 1e-5 * dev ** 2                              # reach first, then stay near neutral
    return cost, th1, th2, miss


def thumb_ik(cmc, target, wrist, mcps, side, depth=None, skip_tip=False):
    """Thumb chain [CMC, MCP, IP, tip] (world) reaching for target within the joint limits above, and the miss
    (m) when the target is out of the thumb's range. The chain has one spare degree of freedom (the CMC is a
    2-axis joint): with depth (key depths) it is spent keeping the thumb out of the neighbouring keys."""
    c, Mh = hand_frame(wrist, mcps, side)
    cmc = np.asarray(cmc, float)
    t = Mh @ (np.asarray(target, float) - cmc)
    cost, *_ = _thumb_solve(t, THUMB["dirs"])
    m0 = THUMB["dirs"][int(np.argmin(cost))]                        # refine around the best coarse direction
    fine = _cone_dirs(m0, math.radians(6), 300, 1)
    fine = np.vstack([THUMB["dirs"], m0[None], fine[np.arccos(np.clip(fine @ THUMB["n0"], -1, 1)) <= THUMB_CONE]])
    cost, th1, th2, miss = _thumb_solve(t, fine)
    order = np.argsort(cost)
    k = int(order[0])
    if depth is not None:
        ok = order[miss[order] < miss[k] + 0.001][:60]              # every pose that reaches as well as the best
        mcp, ip, tip = _thumb_fk(fine[ok], th1[ok], th2[ok])
        best = None
        for i, kk in enumerate(ok):
            pts = [cmc] + [cmc + Mh.T @ q[i] for q in (mcp, ip, tip)]
            pen = _penetration(pts, FINGER_R[0], depth, skip_tip)
            if best is None or pen < best[0] - 0.0002:
                best = (pen, int(kk))
            if pen <= 0.0005:
                break
        k = best[1]
    mcp, ip, tip = _thumb_fk(fine[k:k + 1], th1[k:k + 1], th2[k:k + 1])
    pts = [cmc] + [cmc + Mh.T @ q[0] for q in (mcp, ip, tip)]
    return pts, float(miss[k])


def set_hand_model(name):
    """'mpfb': segment lengths, knuckle layout, palm length and finger radii measured off the CC0 MPFB2 hand
    (blockout/hand_model/), so the IK chains and the rendered skinned hand agree bone for bone.
    'mannequin': the hand-tuned proportions the capsule/skin hands were built with."""
    global SEG, KNUCKLE, FINGER_R, TIP_R, PALM_LEN, WRIST_OFF, HAND_MODEL, THUMB
    HAND_MODEL = name
    if name == "mannequin":
        SEG, KNUCKLE, FINGER_R = _SEG0, _KNUCKLE0, _FINGER_R0
        PALM_LEN = 0.085                         # knuckle line -> wrist
        WRIST_OFF = np.array([0.004, -PALM_LEN, -0.006])
        THUMB = None
    else:
        hd = json.load(open(MPFB_JSON))["hands"]["R"]
        B, rad = hd["bones"], hd["radius"]
        c, M = mpfb_frame(B, "R")
        SEG = np.array([[np.linalg.norm(np.array(B[f"finger{f}-{s}"]["tail"]) - B[f"finger{f}-{s}"]["head"])
                         for s in (1, 2, 3)] for f in range(1, 6)])
        KNUCKLE = np.array([M @ (np.array(B[f"finger{f}-1"]["head"]) - c) for f in range(1, 6)])
        WRIST_OFF = M @ (np.array(B["wrist"]["head"]) - c)
        PALM_LEN = float(-WRIST_OFF[1])
        r = []
        for f in range(1, 6):
            b = [rad[f"finger{f}-{s}"] for s in (1, 2, 3)]
            if f == 1:
                b[0] = b[1]                      # the thumb metacarpal's verts are the thenar pad, not the thumb
            j = [b[0], (b[0] + b[1]) / 2, (b[1] + b[2]) / 2, b[2] * 0.9]
            r.append(np.minimum.accumulate(j))   # a finger only narrows toward the tip
        FINGER_R = np.array(r)
        loc = lambda n, k: M @ (np.array(B[n][k]) - c)
        d = [loc(f"finger1-{s}", "tail") - loc(f"finger1-{s}", "head") for s in (1, 2)]
        n0, d2 = d[0] / np.linalg.norm(d[0]), d[1] / np.linalg.norm(d[1])
        axis = np.cross(n0, d2); axis /= np.linalg.norm(axis)        # the rest MCP bend fixes the hinge axis
        THUMB = {"seg": SEG[0], "n0": n0, "axis": axis, "dirs": _cone_dirs(n0, THUMB_CONE),
                 "swings": lambda m, n0=n0: _swings(n0, m)}
    TIP_R = FINGER_R[:, 3]


_SEG0, _KNUCKLE0, _FINGER_R0 = SEG, KNUCKLE, FINGER_R
set_hand_model("mpfb")

# Parncutt-style finger-pair spans in semitones (right hand, finger i < j, signed pitch_j - pitch_i):
# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac)
SPAN = {(0, 1): (-5, -3, 1, 4, 5, 6), (0, 2): (-4, -2, 3, 6, 7, 8), (0, 3): (-3, -1, 5, 8, 9, 10),
        (0, 4): (3, 4, 7, 10, 13, 15), (1, 2): (1, 1, 1, 2, 3, 5), (1, 3): (1, 1, 3, 4, 5, 7),
        (1, 4): (2, 2, 5, 6, 8, 10), (2, 3): (1, 1, 1, 2, 2, 4), (2, 4): (1, 1, 3, 4, 5, 7),
        (3, 4): (1, 1, 1, 2, 3, 5)}


def split_hands(notes):
    tracks = {n.track for n in notes}
    if len(tracks) >= 2:
        hi, lo = sorted(tracks)[:2]
        return [n for n in notes if n.track == hi], [n for n in notes if n.track != hi]
    return [n for n in notes if n.pitch >= 60], [n for n in notes if n.pitch < 60]


SLICE_TOL = float(__import__("os").environ.get("SLICE_TOL", "0.05"))   # s: notes this close are one chord (rolled chords)
HAND_SPAN = 0.19          # m, thumb to pinky tip at full stretch (about a ninth)


def playable(chord, hand):
    """What one hand can actually take of a chord: at most 5 notes within HAND_SPAN, anchored on the outer
    voice (top for the right hand, bottom for the left), keeping the chord's outline. The other notes still
    sound in the audio; this only decides what the fingers do."""
    ch = sorted(chord, key=lambda n: n.pitch, reverse=(hand == "R"))
    anchor = key_x(ch[0].pitch)
    reach = [n for n in ch if abs(key_x(n.pitch) - anchor) <= HAND_SPAN]
    if len(reach) > 5:
        reach = [reach[0]] + reach[1:-1][: 3] + [reach[-1]]   # outer notes plus the nearest inner ones
    return sorted(reach, key=lambda n: n.pitch)


def slices(notes, tol=SLICE_TOL, hand="R"):
    out = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if out and n.start - out[-1][0].start < tol:
            out[-1].append(n)
        else:
            out.append([n])
    return [playable(s, hand) for s in out]


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
            c += sum(3.0 for p, f in a if f == 0 and is_black(p)) + sum(1.0 for p, f in a if f == 4 and is_black(p))
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
                for pa, fa in a:                      # keys still held fix the finger order around them
                    if pa not in held:
                        continue
                    for pb, fbb in b:
                        if fbb == fa or pb == pa:
                            continue
                        above = (pb > pa) if hand == "R" else (pb < pa)
                        if above != (fbb > fa) and not (fbb == 0 or fa == 0):
                            t += 40.0
                for pa, fa in a:
                    for pb, fbb in b:
                        if fa == fbb and pa != pb:
                            t += 4.0 if dt < 0.3 else 1.0
                        elif fa != fbb:
                            sc = _span_cost(fa, pa, fbb, pb, hand)
                            t += 0.5 * sc if sc < 50 else 8.0  # a reach this wide means a hand shift
                            up = (pb - pa) if hand == "R" else (pa - pb)
                            if up > 0 and fbb < fa and fbb != 0:
                                t += 25.0  # finger crossing (only the thumb passes under)
                            if up < 0 and fbb > fa and fa != 0:
                                t += 25.0
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
    return _repair(sl, cands, path, hand)


def _conflicts(assign, held, hand):
    """Does this chord fingering clash with keys still held (finger reuse, or order across a held key)?"""
    bad = 0
    for pb, fb in assign:
        for pa, fa, _ in held:
            if fa == fb and pa != pb:
                bad += 1
            elif fa != fb and pa != pb and fa != 0 and fb != 0:
                above = (pb > pa) if hand == "R" else (pb < pa)
                bad += above != (fb > fa)
    return bad


def _repair(sl, cands, path, hand):
    """Walk the music tracking every key still held (from any earlier chord, not just the last): where the
    chosen fingering would reuse a holding finger or cross a held key, switch to the best fingering that fits;
    if none fits, the held key is let go early (a finger substitution under the pedal; it still sounds)."""
    from .midi import Note
    out, held, prev = [], [], None                    # held: [pitch, finger, note]
    for i, (s, k) in enumerate(zip(sl, path)):
        t0 = s[0].start
        held = [h for h in held if h[2].end > t0 + 0.02]
        hv = [(p, f, n) for p, f, n in held]
        choice = cands[i][k][0]
        if _conflicts(choice, hv, hand):
            ok = [c for c in cands[i] if not _conflicts(c[0], hv, hand)]
            if ok:
                choice = min(ok, key=lambda c: c[1] + abs(c[2] - (prev[2] if prev else c[2])) / WHITE_W)[0]
            else:                                     # let the clashing held keys go just before this chord
                for h in held:
                    if _conflicts(choice, [h], hand):
                        h[2].end = max(h[2].start + 0.05, t0 - 0.03)
                held = [h for h in held if h[2].end > t0 + 0.02]
        by_pitch = dict(choice)
        row = [(n, by_pitch.get(n.pitch, 2)) for n in s]
        out.append(row)
        prev = next((c for c in cands[i] if c[0] == choice), None)
        held += [[n.pitch, f, n] for n, f in row]
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


REACH_LEAD = float(__import__("os").environ.get("REACH_LEAD", "0.35"))
REACH_D = float(__import__("os").environ.get("REACH_D", "0.04"))   # m: how far a held finger may be left behind


def release_far(fing, ev, hand, D=None):
    """A finger that holds a key while the hand moves on lets go before the hand leaves (the pedal / the
    sound carries the note), instead of being dragged off its key: no key stays down without a finger on it."""
    D = REACH_D if D is None else D
    if not fing:
        return ev
    sgn = 1 if hand == "R" else -1
    t_on = np.array([s[0][0].start for s in fing])
    hx = np.array([_place([(n.pitch, f) for n, f in s], any(is_black(n.pitch) for n, _ in s), hand) for s in fing])
    for f, evs in ev.items():
        for e in evs:
            kx = contact(e[2], e[4], f == 0)[0]
            for i in np.nonzero((t_on > e[0] + 0.02) & (t_on < e[1]))[0]:
                if abs(kx - (hx[i] + sgn * HOME[f])) > D:
                    e[1] = max(e[0] + 0.05, min(e[1], t_on[i] - REACH_LEAD))
                    break
    return ev


def strike_time(touch):
    """Caressed notes are pressed slowly; struck notes are fast."""
    return 0.080 - 0.050 * touch


def finger_tip(t, e, home_tip, r, thumb=False):
    """Fingertip (pad centre) target for one note event at time t, and the key depth it causes (or None if
    idle). The pad centre sits one pad radius r above the key surface it touches."""
    t_on, t_off, p, vel, nbk, touch, lift = e
    S = strike_time(touch)
    rel = 0.10 - 0.05 * touch
    approach = 0.10 + 0.08 * touch
    if t < t_on - S - approach or t > t_off + rel:
        return None, 0.0
    c = contact(p, nbk, thumb) + np.array([0, 0, r])
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


def _seg_dist(p0, p1, q0, q1):
    """Closest distance between segments p0-p1 and q0-q1 (Ericson)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            s, t = np.clip(-c / a, 0, 1), 0.0
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0: t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1: t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p0 + d1 * s - q0 - d2 * t))


PALM_R, ARM_R = 0.015, 0.030


def _capsules(h):
    """A posed hand as capsules (a, b, radius): finger segments, palm wrist->each MCP, forearm wrist->elbow."""
    w, caps = np.asarray(h["wrist"], float), []
    for f, ch in enumerate(h["fingers"]):
        ch = np.asarray(ch, float)
        for i in range(3):
            caps.append((ch[i], ch[i + 1], 0.5 * (FINGER_R[f][i] + FINGER_R[f][i + 1])))
        caps.append((w, ch[0], PALM_R))
    caps.append((w, np.asarray(h["elbow"], float), ARM_R))
    return caps


def _hands_detail(frames, thresh=0.003):
    """qa.hands: frames where the two hands' capsule volumes interpenetrate by more than thresh (m)."""
    worst, bad, wf = 0.0, 0, None
    for j, fr in enumerate(frames):
        if "R" not in fr["hands"] or "L" not in fr["hands"]:
            continue
        A, B = _capsules(fr["hands"]["L"]), _capsules(fr["hands"]["R"])
        d = 0.0
        for a0, a1, ra in A:
            for b0, b1, rb in B:
                if min(np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b1)) > 0.25:
                    continue
                d = max(d, ra + rb - _seg_dist(a0, a1, b0, b1))
        if d > thresh:
            bad += 1
        if d > worst:
            worst, wf = d, j
    return {"frames": len(frames), "overlap_frames_over_3mm": bad, "max_depth_mm": round(worst * 1000, 2), "worst_frame": wf}


def _proxy_caps(centre, kn_off, tip, wrist):
    """Cheap hand volume before the IK: palm wrist->MCPs, straight finger rods knuckle->tip."""
    caps = []
    for f in range(5):
        kn = centre + kn_off[f]
        caps.append((wrist, kn, PALM_R))
        caps.append((kn, tip[f], 0.5 * (FINGER_R[f][0] + FINGER_R[f][3])))
    return caps


COLL_MARGIN = 0.004        # hands try to keep this much air between their volumes
COLL_FALL = 0.92
COLL_MAX_X, COLL_MAX_Y, COLL_MAX_Z, COLL_MAX_TIP = 0.020, 0.030, 0.030, 0.04
COLL_ON = False            # hand-to-hand avoidance: experimental, did not reduce qa.hands (see commit message)


def _separate_hands(hands, st, tips, j, t, fps):
    """Hand-to-hand collision: when the two hands' volumes meet, the hand that is not sounding yields. It slides
    away along the keys and, if it has no finger on a key, lifts over the other. The correction is itself a
    smoothed state (fast in, slow out), so it never pops."""
    if len(st) < 2 or not COLL_ON:
        return
    sound = {}
    for name, H in hands.items():
        sound[name] = sum(1 for f in range(5) if H["act"][j][f] is not None)
    yl = "L" if sound["L"] < sound["R"] or (sound["L"] == sound["R"] and st["L"][0][1] <= st["R"][0][1]) else "R"
    ys, os_ = (yl, "R" if yl == "L" else "L")
    def shift(name, v):
        st[name][0][:] += v
        for f in range(5):
            if hands[name]["act"][j][f] is None:
                tips[name][f] += v
    for name in st:                                  # last frame's correction fades out; the projection re-adds what is needed
        corr = hands[name].setdefault("coll", np.zeros(3))
        if np.any(corr):
            new = corr * COLL_FALL
            shift(name, new - corr)
            corr[:] = new
    free = sound[ys] == 0
    share = 1.0 if sound[os_] > sound[ys] else 0.6
    for _ in range(3):
        caps = {}
        for name in st:
            c, roll, psi, kn_off, sgn = st[name]
            wr = c + _yaw(_roll_offsets(WRIST_OFF * np.array([sgn, 1, 1]), roll * 0.5), psi)
            caps[name] = _proxy_caps(c, kn_off, tips[name], wr)
        d = -1.0
        for a0, a1, ra in caps["L"]:
            for b0, b1, rb in caps["R"]:
                d = max(d, ra + rb + COLL_MARGIN - _seg_dist(a0, a1, b0, b1))
        if d <= 0:
            break
        away = 1.0 if st[ys][0][0] >= st[os_][0][0] else -1.0
        for name, w, zl in ((ys, share, 0.5 if free else 0.0), (os_, 1.0 - share, 0.0)):
            if w > 0:
                v = np.array([(away if name == ys else -away) * 0.8 * d * w, 0.0, zl * d])
                lim = np.array([COLL_MAX_X, COLL_MAX_Y, COLL_MAX_Z])
                v = np.clip(hands[name]["coll"] + v, -lim, lim) - hands[name]["coll"]
                shift(name, v)
                hands[name]["coll"] += v

    # free fingertips back out of the other hand's volume (thumbs tuck, fingers lift) -- sprung like the rest
    for name in st:
        other = "R" if name == "L" else "L"
        oc = _proxy_caps(st[other][0], st[other][3], tips[other], st[other][0] + _yaw(_roll_offsets(
            WRIST_OFF * np.array([st[other][4], 1, 1]), st[other][1] * 0.5), st[other][2]))
        to = hands[name].setdefault("tipoff", np.zeros((5, 3)))
        for f in range(5):
            if hands[name]["act"][j][f] is not None:
                to[f] = 0.0
                continue
            if np.any(to[f]):
                new = to[f] * COLL_FALL
                tips[name][f] += new - to[f]
                to[f] = new
            for _ in range(2):
                tp, rf = tips[name][f], FINGER_R[f][3]
                worst, push = 0.0, None
                for a0, a1, ra in oc:
                    ab = a1 - a0
                    u = float(np.clip((tp - a0) @ ab / max(ab @ ab, 1e-9), 0, 1))
                    v = tp - (a0 + ab * u)
                    d = ra + rf + COLL_MARGIN - float(np.linalg.norm(v))
                    if d > worst:
                        worst, push = d, v
                if push is None:
                    break
                h = np.array([push[0], push[1], 0.0]); hn = np.linalg.norm(h)
                h = h / hn if hn > 1e-6 else np.array([1.0 if name == "R" else -1.0, 0.0, 0.0])
                v = h * worst * 0.9 + np.array([0.0, 0.0, 0.5 * worst])
                v = np.clip(to[f] + v, -COLL_MAX_TIP, COLL_MAX_TIP) - to[f]
                tips[name][f] += v
                to[f] += v


COLL_LIFT_MAX = 0.05


def _lift_free(fr, hands, j, solvers, raw):
    """After the IK: a free finger whose real segments sit inside the other hand rides up and over it (the
    lift is remembered and fades, so it never pops). Pressing fingers are never touched."""
    if len(fr["hands"]) < 2 or not COLL_ON:
        return
    for name, H in hands.items():
        other = "R" if name == "L" else "L"
        oc = _capsules({**fr["hands"][other], "elbow": fr["hands"][other]["wrist"]})[:-1]
        lift = H.setdefault("lift", np.zeros(5))
        for f in range(5):
            if H["act"][j][f] is not None:
                lift[f] = 0.0
                continue
            tip0, pts = raw[(name, f)]
            lift[f] *= COLL_FALL
            tip = tip0 + np.array([0, 0, lift[f]])
            if lift[f] > 1e-4:
                pts, _m = solvers[(name, f)](tip)
            for _ in range(3):
                pts = np.asarray(pts, float)
                d = -1.0
                for i in range(3):
                    rf = 0.5 * (FINGER_R[f][i] + FINGER_R[f][i + 1])
                    for a0, a1, ra in oc:
                        d = max(d, ra + rf + COLL_MARGIN - _seg_dist(pts[i], pts[i + 1], a0, a1))
                add = min(d + 0.001, COLL_LIFT_MAX - lift[f])
                if d <= 0 or add <= 0:
                    break
                lift[f] += add
                tip = tip + np.array([0, 0, add])
                pts, _m = solvers[(name, f)](tip)
            if lift[f] > 1e-4:
                fr["hands"][name]["fingers"][f] = [[round(float(c), 5) for c in q] for q in np.asarray(pts, float)]


def _cross_detail(frames):
    """Finger-order violations: adjacent fingers (index..pinky) whose tips are out of order across the hand
    (closer than 4 mm or crossed), and splay beyond what a hand can do (tip far sideways of its knuckle)."""
    cross, splay, n = 0, 0, 0
    for fr in frames:
        for name, h in fr["hands"].items():
            sgn = 1 if name == "R" else -1
            ax = np.array(h["fingers"][4][0][:2]) - np.array(h["fingers"][1][0][:2])   # across the hand, index -> pinky
            ax /= max(np.linalg.norm(ax), 1e-6)
            tx = [float(np.array(c[3][:2]) @ ax) for c in h["fingers"]]
            kx = [float(np.array(c[0][:2]) @ ax) for c in h["fingers"]]
            for f in range(1, 4):
                cross += tx[f + 1] - tx[f] < 0.004
            for f in range(1, 5):
                splay += abs(tx[f] - kx[f]) > SPLAY[f] + 0.015      # beyond a fully stretched hand
            n += 1
    return {"hand_frames": n, "crossed_pairs": int(cross), "over_splayed": int(splay)}


def _hit_detail(frames):
    """Where the fingertips actually press: share inside the hit pad, and mean leverage (1 = front edge)."""
    inside, lev, n = 0, 0.0, 0
    for fr in frames:
        for p, who in fr.get("press", {}).items():
            p = int(p)
            h = fr["hands"][who[0]]
            tip = h["fingers"][int(who[1]) - 1][3]
            lo, hi = HITPAD["black" if is_black(p) else "white"]
            inside += lo - 0.004 <= tip[1] <= hi + 0.004
            lev += leverage(p, tip[1]); n += 1
    return {"presses": n, "in_hitpad": round(inside / max(n, 1), 3), "mean_leverage": round(lev / max(n, 1), 3)}


def _pen_detail(info):
    """Where the remaining penetrations are: pressing vs free fingers, by finger, and how stretched the finger was."""
    bad = [i for i in info if i[0] > 0.001]
    reach = SEG.sum(1)
    return {"pressing": sum(1 for i in bad if i[2]), "free": sum(1 for i in bad if not i[2]),
            "by_finger": [sum(1 for i in bad if i[1] == f) for f in range(5)],
            "overstretched": sum(1 for i in bad if i[3] > 0.97 * reach[i[1]])}


# Anatomical joint ranges for fingers 2-5 (degrees of flexion; negative = bending backwards), and how far a hinge
# joint (PIP, DIP) may bend sideways. A pose outside these is a broken finger on screen.
ANAT = {"mcp": (-20.0, 90.0), "pip": (0.0, 105.0), "dip": (-5.0, 85.0), "abd": 25.0, "lat": 10.0}


def finger_angles(wrist, chains, side):
    """Per finger 2-5: (MCP flexion, PIP flexion, DIP flexion, MCP abduction, PIP lateral, DIP lateral), degrees.
    Flexion is measured about the hand's across axis carried onto the finger, positive toward the palm."""
    wrist = np.asarray(wrist, float)
    ch = [np.asarray(c, float) for c in chains]
    _, M = hand_frame(wrist, [ch[f][0] for f in range(1, 5)], side)
    ac, fwd, up = M
    ax = np.cross(fwd, -up); ax /= np.linalg.norm(ax)
    n = lambda v: v / np.linalg.norm(v)
    out = []
    for f in range(1, 5):
        q = ch[f]
        mc = n(q[0] - wrist)
        s = [n(q[k + 1] - q[k]) for k in range(3)]
        axf = n(ax - s[0] * (ax @ s[0]))
        flex = lambda u, v, a: math.degrees(math.atan2(float(np.cross(u, v) @ a), float(u @ v)))
        lat = lambda u, v: math.degrees(math.asin(float(np.clip(v @ axf, -1, 1)))) - \
            math.degrees(math.asin(float(np.clip(u @ axf, -1, 1))))
        mcp_flex = flex(mc - ax * (mc @ ax), s[0] - ax * (s[0] @ ax), ax)
        abd = math.degrees(math.asin(float(np.clip(s[0] @ n(np.cross(up, mc)), -1, 1))))
        out.append((mcp_flex, flex(s[0], s[1], axf), flex(s[1], s[2], axf), abd, lat(s[0], s[1]), lat(s[1], s[2])))
    return out


def _anat_detail(frames):
    """Finger-frames (fingers 2-5) with a joint outside ANAT, by kind. The same check for every hand model."""
    bad = {"mcp": 0, "pip": 0, "dip": 0, "abd": 0, "lat": 0}
    worst = {k: 0.0 for k in bad}
    n = 0
    for fr in frames:
        for h, hd in fr["hands"].items():
            for m, pi, di, ab, l1, l2 in finger_angles(hd["wrist"], hd["fingers"], h):
                n += 1
                for k, v, (lo, hi) in (("mcp", m, ANAT["mcp"]), ("pip", pi, ANAT["pip"]), ("dip", di, ANAT["dip"])):
                    e = max(lo - v, v - hi, 0.0)
                    bad[k] += e > 0; worst[k] = max(worst[k], e)
                for k, v, lim in (("abd", ab, ANAT["abd"]), ("lat", max(abs(l1), abs(l2)), ANAT["lat"])):
                    e = max(abs(v) - lim, 0.0)
                    bad[k] += e > 0; worst[k] = max(worst[k], e)
    return {"finger_frames": n, "out_of_range": bad, "worst_excess_deg": {k: round(v, 1) for k, v in worst.items()}}


def _reach_detail(hands):
    """Pressing fingers whose solved chain doesn't get the fingertip to its key (joint limits / segment
    lengths): a key going down with no finger on it. Same measure for every hand model."""
    tm = [i for H in hands.values() for i in H.get("tip_miss", [])]
    bad = [(m, f) for m, f in tm if m > 0.003]
    return {"pressing_frames": len(tm), "miss_over_3mm": len(bad),
            "by_finger": [sum(1 for _, f in bad if f == q) for q in range(5)],
            "max_miss_mm": round(max((m for m, _ in tm), default=0.0) * 1000, 1)}


def _penetration(pts, radii, depth, skip_tip=False):
    """Deepest point (m) of a finger's capsules inside the keyboard surface; 0 if clear."""
    worst = 0.0
    for k in range(3):
        a, b = np.asarray(pts[k]), np.asarray(pts[k + 1])
        for u in (0.33, 0.66, 1.0):
            if skip_tip and k == 2 and u > 0.5:
                continue
            q = a + (b - a) * u
            r = radii[k] + (radii[k + 1] - radii[k]) * u
            worst = max(worst, surface_under(q[0], q[1], r, depth) + r - q[2])
    return worst


def _ik(m, target, seg, fwd_hint, tip_pitch=None):
    """Finger IK in the vertical plane through knuckle m and the target. Returns 4 points (MCP, PIP, DIP, tip).
    With tip_pitch (radians below horizontal) the last segment comes down at that angle, so the fingertip
    (not the pad or the middle phalanx) meets the key, and MCP/PIP are solved exactly with the knuckle arched
    up; if that can't reach, the finger flattens step by step, like a stretched finger. Without it: the
    coupled 3-link solve (DIP = 0.75 x PIP)."""
    if tip_pitch is not None:
        v = target - m
        hd = np.array([v[0], v[1], 0.0])
        if np.linalg.norm(hd) < 1e-4:
            hd = np.array([fwd_hint[0], fwd_hint[1], 0.0])
        hd /= np.linalg.norm(hd)
        L1, L2, L3 = seg
        a3 = tip_pitch
        while a3 > 0.15:
            dip = target - L3 * (math.cos(a3) * hd + np.array([0, 0, -math.sin(a3)]))
            w = dip - m
            d2, h2 = float(w @ hd), float(w[2])
            D = math.hypot(d2, h2)
            if abs(L1 - L2) < D < L1 + L2 - 1e-4:
                phi = math.atan2(h2, d2)
                beta = math.acos(np.clip((L1 * L1 + D * D - L2 * L2) / (2 * L1 * D), -1, 1))
                a1 = phi + beta                                   # knuckle arch up
                pip = m + L1 * (math.cos(a1) * hd + np.array([0, 0, math.sin(a1)]))
                if pip[2] >= dip[2] - 0.002:                       # no hyperextended PIP
                    return [m, pip, dip, target]
            a3 -= 0.15
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
    yoff = 0.018 if nb else 0.0
    c = np.array([x, -0.006 + yoff, KNUCKLE_Z])
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


YAW_MAX = 0.45             # rad: wrist turn (radial/ulnar deviation plus forearm angle)
SPLAY = np.array([0.060, 0.036, 0.024, 0.032, 0.048])   # max sideways tip offset from its knuckle, per finger
PULL = 0.93               # pressing fingers pull the hand once their tip is further than this share of reach
KN_FWD = 0.065            # how far the knuckle line may go past the white key fronts
TETHER = 0.80             # free fingertips stay within this share of the finger's length from the knuckle
FINGER_GAP = 0.017         # free fingertips keep at least a finger's width apart, in order across the hand


def _yaw(v, psi):
    """Rotate hand-local (x across, y forward) offsets by the wrist turn psi (+ points the fingers to +x)."""
    c, s_ = math.cos(psi), math.sin(psi)
    o = np.array(v, float)
    x, y = o[..., 0].copy(), o[..., 1].copy()
    o[..., 0], o[..., 1] = x * c + y * s_, -x * s_ + y * c
    return o


def _order_free(tips, act, centre, kn_off, psi, sgn, tether=None):
    """Keep free fingertips in hand order (index < middle < ring < pinky across the hand, thumb inside the
    index) and within each finger's splay; pressing fingers stay where their keys are."""
    free = [act[f] is None for f in range(5)]
    loc = _yaw(tips - centre, -psi); loc[:, 0] *= sgn
    kl = _yaw(kn_off, -psi); kl[:, 0] *= sgn
    for f in range(1, 5):
        if free[f]:
            loc[f, 0] = np.clip(loc[f, 0], kl[f, 0] - SPLAY[f], kl[f, 0] + SPLAY[f])
    for f in range(2, 5):
        if free[f] and loc[f, 0] < loc[f - 1, 0] + FINGER_GAP:
            loc[f, 0] = loc[f - 1, 0] + FINGER_GAP
    for f in range(3, 0, -1):
        if free[f] and loc[f, 0] > loc[f + 1, 0] - FINGER_GAP:
            loc[f, 0] = loc[f + 1, 0] - FINGER_GAP
    if free[0]:
        loc[0, 0] = min(loc[0, 0], loc[1, 0] - 0.012)
    loc[:, 0] *= sgn
    out = centre + _yaw(loc, psi)
    out[:, 2] = tips[:, 2]
    if tether is not None:                  # a free finger can't be further from its knuckle than it is long:
        for f in range(5):                  # shorten it along the hand (forward, down), not sideways
            if free[f]:
                v = out[f] - (centre + kn_off[f])
                lat = float(v @ np.array([math.cos(psi), -math.sin(psi), 0.0]))
                w = v - lat * np.array([math.cos(psi), -math.sin(psi), 0.0])
                lim = math.sqrt(max(tether[f] ** 2 - lat * lat, (0.3 * tether[f]) ** 2))
                d = np.linalg.norm(w)
                if d > lim:
                    out[f] = centre + kn_off[f] + (v - w) + w * (lim / d)
    return out


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
        ns = [type(n)(n.start, n.end, n.pitch, n.velocity, n.track) for n in ns]   # motion copies (repair may shorten holds)
        for n, o in zip(ns, [m for m in (rh if name == "R" else lh)]):
            n.touch, n.lift = getattr(o, "touch", 0.3), getattr(o, "lift", 1.0)
        fing = fingering(slices(ns, hand=name), name) if ns else []
        ev = release_far(fing, finger_events(fing), name)
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
        TCH = np.zeros(N); YAW = np.zeros(N)
        shoulder = np.array([cx + sgn * 0.19, -0.45])
        for j in range(N):
            C[j], _, _ = _centre(H, H["xh"][j], H["nbh"][j], times[j] + LEAD_HAND)
            t = times[j] + LEAD_TIP
            ct, tch, yoff = _centre(H, H["xt"][j], H["nbt"][j], t)
            TCH[j] = tch
            hover = 0.003 + 0.012 * tch
            playing = []
            psi_arm = math.atan2(ct[0] - shoulder[0], ct[1] - shoulder[1])   # the forearm's own angle
            psi0 = YAW[j - 1] if j else 0.6 * psi_arm
            for f in range(5):
                loc = np.array([sgn * HOME[f], 0.020 + yoff - ct[1], 0.0])
                if f == 0:
                    loc += np.array([0.0, 0.006, 0.0])         # the thumb lies long and low along the keys
                home = ct + _yaw(loc, psi0)
                home[2] = surface_under(home[0], home[1], TIP_R[f], {}) + TIP_R[f] + hover
                tip = home
                eo = H["ev_on"][f]
                lo, hi = max(0, int(np.searchsorted(eo, t - 8.0))), int(np.searchsorted(eo, t + 0.3))
                for e in H["ev"][f][lo:hi][::-1]:
                    tp, dep = finger_tip(t, e, home, TIP_R[f], f == 0)
                    if tp is not None:
                        tip = tp
                        c = contact(e[2], e[4], f == 0) + np.array([0, 0, TIP_R[f]])
                        act[j][f] = (e[2], float(c[0]), float(c[2]))
                        if e[0] - 0.15 <= t <= e[1]:
                            playing.append(f)
                        break
                TIP[j, f] = tip
            if playing:
                ROLL[j] = sgn * ROLL_MAX * (2 - np.mean(playing)) / 2
                d = np.mean([TIP[j, f] for f in playing], axis=0) - C[j]
                psi_reach = 0.5 * math.atan2(d[0] - sgn * np.mean([HOME[f] for f in playing]) * sgn, max(d[1] + 0.06, 0.03))
            else:
                psi_reach = 0.0
            YAW[j] = float(np.clip(0.6 * psi_arm + 0.4 * psi_reach, -YAW_MAX, YAW_MAX))
        H["C"], H["ROLL"], H["TIP"], H["act"], H["TCH"], H["YAW"] = C, ROLL, TIP, act, TCH, YAW

    # ---- pass 2: springs + organic drift
    for k, (name, H) in enumerate(hands.items()):
        H["Cf"] = dynamics.filter_track(H["C"], fps, *SPRING_HAND)
        H["Cf"] += np.stack([dynamics.drift(times, a, seed * 10 + 3 * k + i) for i, a in enumerate((0.0015, 0.0015, 0.0012))], 1)
        H["Rf"] = dynamics.filter_track(H["ROLL"], fps, *SPRING_ROLL) + dynamics.drift(times, 0.02, seed * 10 + 7 + k)
        H["Yf"] = dynamics.filter_track(H["YAW"], fps, *SPRING_ROLL) + dynamics.drift(times, 0.015, seed * 10 + 9 + k)
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
        fr = {"t": round(float(t - start), 4), "keys": {}, "hands": {}, "energy": round(float(energy[j]), 3)}
        keydepth = {}
        tips, press = {}, {}
        for name, H in hands.items():                       # pressing fingers move their keys
            tips[name] = H["Tf"][j].copy()
            for f in range(5):
                a = H["act"][j][f]
                if a is None:
                    continue
                p, kx, cz = a
                tips[name][f, 2] = max(tips[name][f, 2], cz - KEY_TRAVEL)       # a key can't go past its bed
                if abs(tips[name][f, 0] - kx) < WHITE_W:
                    d = float(np.clip((cz - tips[name][f, 2]) / KEY_TRAVEL, 0, 1))
                    if d > 0.01:
                        keydepth[p] = max(keydepth.get(p, 0.0), d)
                        press[str(p)] = f"{name}{f + 1}"            # which finger holds the key (R1 = right thumb)
        st = {}; solvers = {}; tipsol = {}; raw = {}
        for name, H in hands.items():                       # then nothing may sink into a key
            sgn = 1 if name == "R" else -1
            centre, roll, psi = H["Cf"][j].copy(), float(H["Rf"][j]), float(H["Yf"][j])
            kn_off = _yaw(_roll_offsets(KNUCKLE * np.array([sgn, 1, 1]), roll), psi)
            reach = SEG.sum(1)
            side = np.array([math.cos(psi), -math.sin(psi), 0.0])         # across the hand (local +x)
            for _ in range(4):                               # pressing fingers pull the hand within reach
                pull, n = np.zeros(3), 0
                for f in range(5):
                    if H["act"][j][f] is None:
                        continue
                    if f == 0 and THUMB is not None:          # the thumb: move the hand by what its joints can't reach
                        wr = centre + _yaw(_roll_offsets(WRIST_OFF * np.array([sgn, 1, 1]), roll * 0.5), psi)
                        pts, miss = thumb_ik(centre + kn_off[0], tips[name][0], wr,
                                             [centre + kn_off[q] for q in range(1, 5)], name)
                        if miss > 0.001:
                            pull += (tips[name][0] - pts[3]) * 1.1; n += 1
                        continue
                    v = tips[name][f] - (centre + kn_off[f])
                    ex = np.linalg.norm(v) - PULL * reach[f]
                    if ex > 0:
                        vh = np.array([v[0], v[1], 0.0])
                        pull += vh / max(np.linalg.norm(vh), 1e-6) * ex; n += 1
                    lat = float(v @ side)                               # too much splay: the hand moves over
                    if abs(lat) > SPLAY[f]:
                        pull += side * (lat - math.copysign(SPLAY[f], lat)); n += 1
                if not n:
                    break
                step = pull / n
                step[1] = np.clip(centre[1] + step[1], -0.03, KN_FWD) - centre[1]  # knuckles stay behind the key fronts
                centre += step
                for f in range(5):                           # free fingers travel with the hand, short of the fallboard
                    if H["act"][j][f] is None:
                        tips[name][f] += step
                        tips[name][f][1] = min(tips[name][f][1], FALLBOARD_Y - 0.015)
            st[name] = (centre, roll, psi, kn_off, sgn)
        _separate_hands(hands, st, tips, j, times[j], fps)
        for name, H in hands.items():
            centre, roll, psi, kn_off, sgn = st[name]
            tips[name] = _order_free(tips[name], H["act"][j], centre, kn_off, psi, sgn, TETHER * SEG.sum(1))
            w_off = _yaw(_roll_offsets(WRIST_OFF * np.array([sgn, 1, 1]) - np.array([0, 0, 0.004 * H["TCH"][j]]), roll * 0.5), psi)
            wrist = centre + w_off
            chains = []
            for f in range(5):
                tip = tips[name][f]
                a = H["act"][j][f]
                own = a[0] if a is not None else None
                kn = centre + kn_off[f]
                floor = surface_under(tip[0], tip[1], TIP_R[f], keydepth) + TIP_R[f]
                if a is None or not (abs(tip[0] - a[1]) < WHITE_W / 2):
                    tip[2] = max(tip[2], floor)
                tp = 0.40 + 0.25 * float(H["TCH"][j]) if f == 0 else 0.72 + 0.50 * float(H["TCH"][j])

                def mk(f, kn, tp, own, name, centre, kn_off, wrist):
                    def solve(tip):
                        if f == 0 and THUMB is not None:          # the thumb has its own joints, not a finger's
                            return thumb_ik(kn, tip, wrist, [centre + kn_off[q] for q in range(1, 5)], name,
                                            keydepth, own is not None)
                        return _ik(kn, tip, SEG[f], np.array([0, 1.0, 0]), tp), 0.0
                    return solve
                solve = mk(f, kn, tp, own, name, centre, kn_off, wrist)
                solvers[(name, f)] = solve
                tipsol[(name, f)] = tip
                pts, miss = solve(tip)
                for _ in range(3):                           # lift the finger until no segment is inside a key
                    pen = _penetration(pts, FINGER_R[f], keydepth, skip_tip=own is not None)
                    if pen <= 0.0005:
                        break
                    tip = tip + np.array([0, 0, pen + 0.0005])
                    pts, miss = solve(tip)
                if own is not None:                          # does the fingertip actually get to its key?
                    H.setdefault("tip_miss", []).append((float(np.linalg.norm(np.asarray(pts[3]) - tip)), f))
                    H.setdefault("dbg", []).append((round(float(t), 2), name, f, own, round(float(tip[0] - kn[0]), 3), round(float(tip[1] - kn[1]), 3), round(float(tip[2]-kn[2]),3), round(float(np.linalg.norm(np.asarray(pts[3]) - tip)), 4), [None if q is None else q[0] for q in H['act'][j]], round(float(centre[0]),3), round(float(centre[1]),3)))
                H.setdefault("pen", []).append(pen)
                H.setdefault("pen_info", []).append((pen, f, own is not None, float(np.linalg.norm(tip - kn)), j, name))
                chains.append([[round(float(c), 5) for c in q] for q in pts])
                raw[(name, f)] = (tip, pts)
            fr["hands"][name] = {"wrist": wrist.round(5).tolist(), "fingers": chains}
        _lift_free(fr, hands, j, solvers, raw)
        pitch, yaw, roll_h, lean = HEAD[j]
        fr["body"] = body.pose(pitch, yaw, roll_h, lean, {h: fr["hands"][h]["wrist"] for h in fr["hands"]})
        for h in fr["hands"]:
            fr["hands"][h]["elbow"] = fr["body"]["elbows"][h]
        fr["keys"] = {str(p): round(d, 3) for p, d in keydepth.items()}
        fr["press"] = press
        frames.append(fr)
    global LAST
    LAST = hands
    fing_out = {name: [[[n.start, n.pitch, f] for n, f in s] for s in H["fing"]] for name, H in hands.items()}
    if os.environ.get("REACH_DBG"):
        json.dump([d for H in hands.values() for d in H.get("dbg", [])], open(os.environ["REACH_DBG"], "w"))
    pens = np.concatenate([H.get("pen", [0.0]) for H in hands.values()])
    return {"fps": fps, "start": start, "speed": speed, "centre_x": cx, "keyboard": keyboard(),
            "finger_radii": FINGER_R.tolist(), "hand_model": HAND_MODEL,
            "qa": {"finger_frames": int(len(pens)), "penetrating_over_1mm": int((pens > 0.001).sum()),
                   "max_penetration_mm": round(float(pens.max()) * 1000, 2),
                   "detail": _pen_detail([i for H in hands.values() for i in H.get("pen_info", [])]),
                   "hits": _hit_detail(frames), "order": _cross_detail(frames), "reach": _reach_detail(hands), "anatomy": _anat_detail(frames), "hands": _hands_detail(frames)},
            "schedule": sched, "beats": [b - start for b in beats.tolist() if start <= b <= end],
            "downbeats": [b - start for b in downbeats.tolist() if start <= b <= end],
            "fingering": fing_out, "frames": frames}


if os.environ.get("PIANO_OVR"):      # experiment hook: {"KNUCKLE_Z":0.07,"HOME_S":1.1,...}
    _o = json.loads(os.environ["PIANO_OVR"])
    for _k, _v in _o.items():
        if _k.endswith("_S"):
            globals()[_k[:-2]] = globals()[_k[:-2]] * _v
        else:
            globals()[_k] = _v


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
    ap.add_argument("--hand", default="mpfb", choices=["mpfb", "mannequin"],
                    help="hand proportions: the MPFB2 rig's (default) or the old mannequin's")
    a = ap.parse_args()
    set_hand_model(a.hand)
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


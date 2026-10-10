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


# Hit pads per finger (y range, metres from the white-key fronts; black keys from their own front). The thumb and
# pinky strike near the front edge; the long fingers 2-4 land deeper, in among the black keys (Chopin's natural
# position): the hand is a curve and the key front a line, and with a real hand's proportions a curled middle
# finger reaches several cm past the thumb. (User call 2026-10-09, replacing one front pad for every finger.)
_BY0 = WHITE_L - BLACK_L
HITPAD_W = [(0.010, 0.025), (0.025, 0.040), (0.030, 0.050), (0.025, 0.040), (0.010, 0.025)]
HITPAD_B = [(_BY0 + 0.004, _BY0 + 0.014), (_BY0 + 0.010, _BY0 + 0.025), (_BY0 + 0.010, _BY0 + 0.025),
            (_BY0 + 0.010, _BY0 + 0.025), (_BY0 + 0.004, _BY0 + 0.014)]
# white keys when the hand is up among the black keys (e.g. D-flat major): the whole hand plays deeper
HITPAD_WN = [(0.015, 0.030), (0.030, 0.045), (0.030, 0.045), (0.030, 0.045), (0.015, 0.030)]
HITPAD = {"white": (0.010, 0.055), "black": (_BY0 + 0.004, _BY0 + 0.025)}     # union, for anything finger-blind


def hitpad(p, f, near_black=False):
    lo, hi = (HITPAD_B if is_black(p) else HITPAD_WN if near_black else HITPAD_W)[f]
    if HITPAD_SHIFT:                          # strike nearer the key fronts (negative = toward the player)
        floor = (WHITE_L - BLACK_L + 0.006) if is_black(p) else 0.006
        lo, hi = max(floor, lo + HITPAD_SHIFT), max(floor + 0.008, hi + HITPAD_SHIFT)
    return lo, hi


def contact(p, near_black=False, f=2):
    """Fingertip strike point on key p for finger f (0 = thumb), inside that finger's pad; on a white key, at the
    back of the pad when the hand is up among the black keys."""
    lo, hi = hitpad(p, f, near_black)
    return np.array([key_x(p), 0.5 * (lo + hi), BLACK_H if is_black(p) else 0.0])


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


def key_lever(p, y):
    """How much of KEY_TRAVEL key p drops at depth y (keys pivot at the back: the front edge drops it all)."""
    return float(np.clip((WHITE_L - y) / (BLACK_L if is_black(p) else WHITE_L), 0.05, 1.0))


SOFT_EDGE = 0.0010         # m: key edges eased over this width in the solver's collision surface


def _sig(u):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, u))))


def surface_soft(x, y, depth, e=SOFT_EDGE):
    """surface() with its edges eased over e, so a solver can feel a key's side or front before it is inside it
    (a step has no slope to follow: the solve stalls against it, then snaps past)."""
    xl = x + _MID_C
    i0 = int(np.clip(xl // WHITE_W, 0, len(_WHITES) - 1))
    f = xl / WHITE_W - i0
    lever = max(WHITE_L - min(max(y, 0.0), WHITE_L), 0.0) / WHITE_L
    zw = lambda i: -depth.get(_WHITES[int(np.clip(i, 0, len(_WHITES) - 1))], 0.0) * KEY_TRAVEL * lever
    z0 = zw(i0)
    z = z0 + (zw(i0 + 1) - z0) * _sig((f - 1) * WHITE_W / e) + (zw(i0 - 1) - z0) * _sig(-f * WHITE_W / e)
    k = int(np.searchsorted(_BLACK_X, x))
    for kk in (k - 1, k):
        if 0 <= kk < len(_BLACK_X):
            s_ = _sig((BLACK_W / 2 - abs(_BLACK_X[kk] - x)) / e) * _sig((y - (WHITE_L - BLACK_L)) / e)
            if s_ > 1e-6:
                zb = BLACK_H - depth.get(int(_BLACKS[kk]), 0.0) * KEY_TRAVEL * max(WHITE_L - y, 0.0) / BLACK_L
                z += s_ * max(0.0, zb - z)
    z = RAIL_Z + (z - RAIL_Z) * _sig(y / e)
    return z + (FALLBOARD_Z - z) * _sig((y - FALLBOARD_Y) / e)


def surface_under_soft(x, y, r, depth):
    return max(surface_soft(x + dx, y + dy, depth) for dx, dy in ((0, 0), (0.8 * r, 0), (-0.8 * r, 0), (0, 0.8 * r), (0, -0.8 * r)))


_WHITE_IDX = {p: i for i, p in enumerate(_WHITES)}
_BLACK_IDX = {int(p): i for i, p in enumerate(_BLACKS)}


def _depth_arrays(depth):
    DW, DB = np.zeros(len(_WHITES)), np.zeros(len(_BLACKS))
    for p, d in depth.items():
        if p in _WHITE_IDX:
            DW[_WHITE_IDX[p]] = d
        elif p in _BLACK_IDX:
            DB[_BLACK_IDX[p]] = d
    return DW, DB


def _sigv(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -40.0, 40.0)))


def surface_soft_v(X, Y, DW, DB, e=SOFT_EDGE):
    """surface_soft for arrays of points (DW, DB: white / black key depths by index, from _depth_arrays)."""
    xl = X + _MID_C
    n = len(_WHITES)
    i0 = np.clip(np.floor(xl / WHITE_W), 0, n - 1).astype(int)
    f = xl / WHITE_W - i0
    lever = np.maximum(WHITE_L - np.clip(Y, 0.0, WHITE_L), 0.0) / WHITE_L
    z0 = -DW[i0] * KEY_TRAVEL * lever
    zn = -DW[np.minimum(i0 + 1, n - 1)] * KEY_TRAVEL * lever
    zp = -DW[np.maximum(i0 - 1, 0)] * KEY_TRAVEL * lever
    z = z0 + (zn - z0) * _sigv((f - 1) * WHITE_W / e) + (zp - z0) * _sigv(-f * WHITE_W / e)
    k = np.searchsorted(_BLACK_X, X)
    yb = _sigv((Y - (WHITE_L - BLACK_L)) / e)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(_BLACK_X))
        kc = np.clip(kk, 0, len(_BLACK_X) - 1)
        s_ = np.where(ok, _sigv((BLACK_W / 2 - np.abs(_BLACK_X[kc] - X)) / e) * yb, 0.0)
        zb = BLACK_H - DB[kc] * KEY_TRAVEL * np.maximum(WHITE_L - Y, 0.0) / BLACK_L
        z = z + s_ * np.maximum(0.0, zb - z)
    z = RAIL_Z + (z - RAIL_Z) * _sigv(Y / e)
    return z + (FALLBOARD_Z - z) * _sigv((Y - FALLBOARD_Y) / e)


_DISC = np.array([(0, 0), (0.8, 0), (-0.8, 0), (0, 0.8), (0, -0.8)])


def surface_under_soft_v(X, Y, R, DW, DB):
    """Highest soft surface under discs of radius R at (X, Y)."""
    Xs = X[:, None] + R[:, None] * _DISC[None, :, 0]
    Ys = Y[:, None] + R[:, None] * _DISC[None, :, 1]
    return surface_soft_v(Xs.ravel(), Ys.ravel(), DW, DB).reshape(Xs.shape).max(1)


def surface_under(x, y, r, depth):
    """Highest surface under a disc of radius r (a finger pad seen from above)."""
    return max(surface(x + dx, y + dy, depth) for dx, dy in ((0, 0), (0.8 * r, 0), (-0.8 * r, 0), (0, 0.8 * r), (0, -0.8 * r)))


# ---------------------------------------------------------------- hand model
# finger 0 = thumb ... 4 = pinky. Home tip offsets: a five-finger position over adjacent white keys.
# Home tip offsets across the hand, from reference footage of pianists: the fingers fan out from the wrist
# and the thumb sits well out to the side, so a relaxed playing hand spans about a seventh to an octave.
HOME = np.array([-0.062, -0.036, 0.000, 0.030, 0.058])   # thumb rests ~6 cm out, not splayed wide
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
HITPAD_SHIFT = -0.012       # m: every strike band moved toward the key fronts (the user: fingers further back on the keys)
PITCH0 = 0.0               # rad: the palm's resting pitch (+ = wrist above the knuckles); calibrated from FürElise
KNUCKLE_Z = 0.064          # knuckle line height over white key tops (mannequin; set_hand_model sets the real hand's)
Y_KN = -0.006              # knuckle line's resting distance from the white key fronts
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


# Fingers 2-5: the MCP is a 2-axis joint (flexion, abduction), PIP and DIP are hinges and the DIP follows the PIP
# (shared tendon: DIP = k * PIP, k in FINGER_K). Degrees. Inside ANAT, the anatomical ranges the QA checks.
FINGER_MCP = (-19.5, 88.0)
FINGER_PIP = (0.0, 103.0)
FINGER_ABD = 24.0
FINGER_K = np.arange(0.45, 0.901, 0.05)
_FGRID = {}


def _finger_grid(f):
    """In-plane chain for MCP flexion 0 over a grid of PIP angles x DIP couplings: distance and bearing of the tip."""
    if (f, tuple(SEG[f])) not in _FGRID:
        L1, L2, L3 = SEG[f]
        tp = np.radians(np.arange(FINGER_PIP[0], FINGER_PIP[1] + 0.01, 0.5))[:, None]
        k = FINGER_K[None, :]
        td = np.minimum(k * tp, np.radians(80.0))
        z = L1 + L2 * np.exp(1j * tp) + L3 * np.exp(1j * (tp + td))
        b = lambda a: np.broadcast_to(a, z.shape)
        _FGRID[(f, tuple(SEG[f]))] = (b(tp), b(k), b(td), np.abs(z), np.angle(z))
    return _FGRID[(f, tuple(SEG[f]))]


def finger_ik(f, mcp, target, wrist, mcps, side, depth=None, skip_tip=False):
    """Finger f (1-4) chain [MCP, PIP, DIP, tip] reaching for target with every joint inside its range, and the
    miss (m) when it can't: a target too far OR too close is not folded or stretched into, it is reported, and
    the hand placement moves the hand instead."""
    _, Mh = hand_frame(wrist, mcps, side)
    ac, fwd, up = Mh
    mcp = np.asarray(mcp, float)
    d0 = mcp - np.asarray(wrist, float); d0 /= np.linalg.norm(d0)   # the metacarpal: zero flexion, zero abduction
    up = up - d0 * (up @ d0); up /= np.linalg.norm(up)              # back of the hand, square to the metacarpal
    side_ = np.cross(up, d0)
    t = np.asarray(target, float) - mcp
    ab = math.atan2(float(t @ side_), float(t @ d0))
    ab = float(np.clip(ab, -math.radians(FINGER_ABD), math.radians(FINGER_ABD)))
    e1 = math.cos(ab) * d0 + math.sin(ab) * side_                 # finger plane: e1 (straight) and -up (palm)
    x, y = float(t @ e1), float(t @ -up)
    r, phi = math.hypot(x, y), math.atan2(y, x)
    tp, k, td, D, beta = _finger_grid(f)
    tm = phi - beta
    lo, hi = math.radians(FINGER_MCP[0]), math.radians(FINGER_MCP[1])
    tmc = np.clip(tm, lo, hi)
    # tip error for each candidate (clamped MCP), then prefer the tendon's usual coupling and a mid-range PIP
    ex = D * np.cos(tmc + beta) - x
    ey = D * np.sin(tmc + beta) - y
    cost = ex * ex + ey * ey + 1e-6 * ((k - 0.65) / 0.2) ** 2 + 1e-6 * ((tp - math.radians(40)) / 1.0) ** 2
    L1, L2, L3 = SEG[f]

    def chain(i):
        a1 = float(tmc[i]); a2 = a1 + float(tp[i]); a3 = a2 + float(td[i])
        pip = mcp + L1 * (math.cos(a1) * e1 - math.sin(a1) * up)
        dip = pip + L2 * (math.cos(a2) * e1 - math.sin(a2) * up)
        tip = dip + L3 * (math.cos(a3) * e1 - math.sin(a3) * up)
        return [mcp, pip, dip, tip]
    flat = np.argsort(cost, axis=None)
    best = np.unravel_index(int(flat[0]), cost.shape)
    pts = chain(best)
    if depth is not None and _penetration(pts, FINGER_R[f], depth, skip_tip) > 0.0005:
        # the PIP/DIP coupling leaves one spare degree of freedom: spend it keeping the finger out of the keys
        e0 = math.sqrt(float(ex[best] ** 2 + ey[best] ** 2))
        cand = [np.unravel_index(int(q), cost.shape) for q in flat[:4000]]
        slack = 0.003 if skip_tip else 0.030       # a free finger only hovers: clearing the keys beats its exact spot
        cand = [c for c in cand if math.sqrt(float(ex[c] ** 2 + ey[c] ** 2)) < e0 + slack][:160]
        scored = []
        for c in cand:
            q = chain(c)
            scored.append((_penetration(q, FINGER_R[f], depth, skip_tip), math.sqrt(float(ex[c] ** 2 + ey[c] ** 2)), q))
        pen0 = min(x[0] for x in scored)
        pts = min((x for x in scored if x[0] <= pen0 + 0.0003), key=lambda x: x[1])[2]
    return pts, float(np.linalg.norm(pts[3] - np.asarray(target, float)))


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


def _rig_home():
    """Home pad offsets across the hand (right hand, m from the knuckle-line centre) in the rig's relaxed,
    human-calibrated pose: where the fingertips sit when nothing pulls them."""
    from .rig.hand import HandRig
    rig = HandRig("R")
    rr, rp = rig.root_from_hand_frame(np.eye(3), np.zeros(3))
    _, _, pts = rig.points(np.concatenate([rp, np.zeros(3), rig.q_rest]), rr)
    return np.array([float(e[0]) for e in rig.effectors(pts, rig._rots)])


def _rig_rest_height():
    """Knuckle-line height over the keys and setback from the key fronts at which the rig's relaxed curl puts the
    index..ring pads on their keys (hand level): derived from the hand, not tuned per piece."""
    from .rig.hand import HandRig
    rig = HandRig("R")
    rr, rp = rig.root_from_hand_frame(np.eye(3), np.zeros(3))
    x = np.concatenate([rp, np.zeros(3), rig.q_rest])
    _, _, pts = rig.points(x, rr)
    eff = rig.effectors(pts, rig._rots)
    dz = float(np.mean([eff[f][2] for f in (1, 2, 3)]))
    dy = float(np.mean([eff[f][1] for f in (1, 2, 3)]))
    pad_y = float(np.mean([0.5 * sum(HITPAD_W[f]) for f in (1, 2, 3)]))
    return -dz, pad_y - dy


def set_hand_model(name):
    """'mpfb': segment lengths, knuckle layout, palm length and finger radii measured off the CC0 MPFB2 hand
    (blockout/hand_model/), so the IK chains and the rendered skinned hand agree bone for bone.
    'mannequin': the hand-tuned proportions the capsule/skin hands were built with."""
    global SEG, KNUCKLE, FINGER_R, TIP_R, PALM_LEN, WRIST_OFF, HAND_MODEL, THUMB, KNUCKLE_Z, Y_KN, HOME
    HAND_MODEL = name
    if name == "mannequin":
        SEG, KNUCKLE, FINGER_R = _SEG0, _KNUCKLE0, _FINGER_R0
        KNUCKLE_Z, Y_KN = 0.064, -0.006
        HOME = _HOME0
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
        from .rig.skeleton import hand_scale
        hs_ = hand_scale()                       # a pianist-sized hand (FürElise), as the rig and the renderer use
        SEG, KNUCKLE, WRIST_OFF, FINGER_R = SEG * hs_, KNUCKLE * hs_, WRIST_OFF * hs_, FINGER_R * hs_
        PALM_LEN = float(-WRIST_OFF[1])
        # a real hand's natural curl (MCP ~15, PIP ~40 deg) puts the long fingertips ~7 cm below and ~6-7 cm in
        # front of the knuckles, so the knuckle line rides higher and further back than the mannequin's
        KNUCKLE_Z, Y_KN = _rig_rest_height()
        from .rig.hand import _furelise_cal, _cal_parts
        cal = _furelise_cal()
        if cal:                                  # measured on concert pianists (blockout.furelise.calibrate)
            global HITPAD_W, HITPAD_WN, HITPAD_B, PITCH0
            parts = _cal_parts()
            if "place" in parts:
                KNUCKLE_Z, Y_KN = cal["knuckle_z"], cal["knuckle_y"]
                PITCH0 = math.radians(cal["pitch_deg"])
            if "hitpad" in parts:
                HITPAD_W = [tuple(b) for b in cal["hitpad"]["white"]]
                HITPAD_WN = [tuple(b) for b in cal["hitpad"]["white_near_black"]]
                HITPAD_B = [tuple(b) for b in cal["hitpad"]["black"]]
        # HOME = _rig_home()   # rig-derived home spread: not yet validated by the fleet, so off for the render lock
        loc = lambda n, k: M @ (np.array(B[n][k]) - c)
        d = [loc(f"finger1-{s}", "tail") - loc(f"finger1-{s}", "head") for s in (1, 2)]
        n0, d2 = d[0] / np.linalg.norm(d[0]), d[1] / np.linalg.norm(d[1])
        axis = np.cross(n0, d2); axis /= np.linalg.norm(axis)        # the rest MCP bend fixes the hinge axis
        THUMB = {"seg": SEG[0], "n0": n0, "axis": axis, "dirs": _cone_dirs(n0, THUMB_CONE),
                 "swings": lambda m, n0=n0: _swings(n0, m)}
    TIP_R = FINGER_R[:, 3]


_SEG0, _KNUCKLE0, _FINGER_R0, _HOME0 = SEG, KNUCKLE, FINGER_R, HOME
set_hand_model("mpfb")

# Parncutt-style finger-pair spans in semitones (right hand, finger i < j, signed pitch_j - pitch_i):
# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac)
SPAN = {(0, 1): (-5, -3, 1, 4, 5, 6), (0, 2): (-4, -2, 3, 6, 7, 8), (0, 3): (-3, -1, 5, 8, 9, 10),
        (0, 4): (3, 4, 7, 10, 13, 15), (1, 2): (1, 1, 1, 2, 3, 5), (1, 3): (1, 1, 3, 4, 5, 7),
        (1, 4): (2, 2, 5, 6, 8, 10), (2, 3): (1, 1, 1, 2, 2, 4), (2, 4): (1, 1, 3, 4, 5, 7),
        (3, 4): (1, 1, 1, 2, 3, 5)}
_CAL = __import__("blockout.rig.hand", fromlist=["_furelise_cal"])._furelise_cal() if HAND_MODEL == "mpfb" else None
if _CAL and "span" in __import__("blockout.rig.hand", fromlist=["_cal_parts"])._cal_parts():   # FürElise spans
    SPAN.update({(int(k[0]), int(k[1])): tuple(v) for k, v in _CAL["span"].items()})


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


SPLIT_WIDE = True           # chords wider than a hand / > 5 keys: rolled sub-chords (False: drop to playable())


def slices(notes, tol=SLICE_TOL, hand="R"):
    out = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if out and n.start - out[-1][0].start < tol:
            out[-1].append(n)
        else:
            out.append([n])
    if not SPLIT_WIDE:
        return [playable(s, hand) for s in out]
    res = []                                       # every note gets a finger: a chord wider than a hand or with more
    for s in out:                                  # than five keys is played as a roll, in onset order (outer voice
        ns, seen = [], set()                       # first), split into sub-chords that one hand can take
        for n in sorted(s, key=lambda n: (n.start, n.pitch if hand == "L" else -n.pitch)):
            if n.pitch not in seen:
                seen.add(n.pitch); ns.append(n)
        cur = []
        for n in ns:
            ps = [m.pitch for m in cur] + [n.pitch]
            if cur and (len(ps) > 5 or abs(key_x(max(ps)) - key_x(min(ps))) > HAND_SPAN):
                res.append(sorted(cur, key=lambda m: m.pitch)); cur = []
            cur.append(n)
        res.append(sorted(cur, key=lambda m: m.pitch))
    return res


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
        ps = sorted({n.pitch for n in s})          # a key doubled in the score (both staves on it) is struck once
        groups = [[p] for p in ps]                 # more than five notes: one finger takes two neighbouring keys
        while len(groups) > 5:                     # (the closest pair, a tone or less apart), as pianists do
            k = min(range(len(groups) - 1), key=lambda i: groups[i + 1][0] - groups[i][-1])
            if groups[k + 1][0] - groups[k][-1] > 2:
                break
            groups[k:k + 2] = [groups[k] + groups[k + 1]]
        cs = []
        relax = False
        for fs in itertools.combinations(range(5), min(len(groups), 5)):
            order = fs if hand == "R" else fs[::-1]
            a = [(p, f) for g, f in zip(groups, order) for p in g]
            c = sum(_span_cost(f1, p1, f2, p2, hand) for (p1, f1), (p2, f2) in itertools.combinations(a, 2))
            c += sum(3.0 for p, f in a if f == 0 and is_black(p)) + sum(1.0 for p, f in a if f == 4 and is_black(p))
            if c < 50 or relax:
                nb = any(is_black(p) for p in ps)
                cs.append((a, c, _place(a, nb, hand)))
        if not cs and len(groups) <= 5:            # wider than a hand: a rolled chord (the onsets are a few ms apart);
            for fs in itertools.combinations(range(5), len(groups)):   # still finger every note, the least-bad
                order = fs if hand == "R" else fs[::-1]                 # stretch, and let the wrist travel the roll
                a = [(p, f) for g, f in zip(groups, order) for p in g]
                c = 50.0 + sum(min(_span_cost(f1, p1, f2, p2, hand), 50.0) for (p1, f1), (p2, f2) in itertools.combinations(a, 2))
                cs.append((a, c, _place(a, any(is_black(p) for p in ps), hand)))
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
    t_end = np.array([min(n.end for n, _ in s) for s in fing])   # the slice's first release frees the hand
    x = np.empty(len(times))
    for j, t in enumerate(times):
        i = np.searchsorted(t_on, t, side="right") - 1
        if i < 0:
            u = _smooth((t - (t_on[0] - 0.3)) / 0.3)
            x[j] = rest_x + (hx[0] - rest_x) * u
            continue
        x[j] = hx[i]
        if i + 1 < len(t_on):  # once its notes let go, the hand drifts toward where it plays next (it doesn't wait,
            gap = t_on[i + 1] - t_on[i]          # or go home): legato leaves only the short glide before the onset
            t1 = t_on[i + 1] - 0.025
            g0 = max(t_end[i], t_on[i] + 0.05)
            dur = max(min(0.12, 0.7 * gap), t1 - g0)
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
            kx = contact(e[2], e[4], f)[0]
            for i in np.nonzero((t_on > e[0] + 0.02) & (t_on < e[1]))[0]:
                if abs(kx - (hx[i] + sgn * HOME[f])) > D:
                    e[1] = max(e[0] + 0.05, min(e[1], t_on[i] - REACH_LEAD))
                    break
    return ev


def strike_time(touch):
    """Caressed notes are pressed slowly; struck notes are fast."""
    return 0.080 - 0.050 * touch


def finger_tip(t, e, home_tip, r, f=2):
    """Fingertip (pad centre) target for one note event at time t, and the key depth it causes (or None if
    idle). The pad centre sits one pad radius r above the key surface it touches."""
    t_on, t_off, p, vel, nbk, touch, lift = e
    S = strike_time(touch)
    rel = 0.10 - 0.05 * touch
    approach = 0.10 + 0.08 * touch
    if t < t_on - S - approach or t > t_off + rel:
        return None, 0.0
    c = contact(p, nbk, f) + np.array([0, 0, r])
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
            tip = (h["pads"] if "pads" in h else [c[3] for c in h["fingers"]])[int(who[1]) - 1]
            f = int(who[1]) - 1
            inside += any(lo - 0.004 <= tip[1] <= hi + 0.004 for lo, hi in (hitpad(p, f), hitpad(p, f, True)))
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
    Measured from the metacarpal (wrist -> MCP) with the back of the hand as up: MCP flexion and abduction are the
    proximal phalanx's angles off the metacarpal; PIP/DIP flexion are about the finger plane's normal (positive
    toward the palm); lateral is how far a hinge segment leaves the finger plane (the plane of the proximal
    phalanx and the hand's up), which a hinge can't do."""
    wrist = np.asarray(wrist, float)
    ch = [np.asarray(c, float) for c in chains]
    _, M = hand_frame(wrist, [ch[f][0] for f in range(1, 5)], side)
    n = lambda v: v / np.linalg.norm(v)
    deg = math.degrees
    out = []
    for f in range(1, 5):
        q = ch[f]
        mc = n(q[0] - wrist)
        hup = M[2]                                       # the hand's own palm normal (back of the hand)
        up = n(hup - mc * (hup @ mc))
        sd = np.cross(up, mc)
        s = [n(q[k + 1] - q[k]) for k in range(3)]
        mcp_p = n(mc - hup * (mc @ hup))                 # metacarpal and phalanx seen in the palm plane
        e1 = n(s[0] - hup * (s[0] @ hup))
        abd = deg(math.atan2(float(np.cross(mcp_p, e1) @ hup), float(e1 @ mcp_p)))
        mcp_flex = deg(math.atan2(float(-(s[0] @ up)), float(s[0] @ n(s[0] - up * (s[0] @ up)))))
        a = n(np.cross(s[0], -hup))                      # finger plane: the proximal phalanx and the palm normal
        flex = lambda u, v: deg(math.atan2(float(np.cross(u, v) @ a), float(u @ v)))
        lat = lambda v: deg(math.asin(float(np.clip(v @ a, -1, 1))))
        out.append((mcp_flex, flex(s[0], s[1]), flex(s[1], s[2]), abd, lat(s[1]), lat(s[2])))
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
SPRING_FREE = (4.5, 0.45, 0.6)     # free fingers: soft and under-damped, the hand's subconscious ragdoll
EXT_UP, EXT_FWD = 0.010, 0.008    # m: how far an idle finger lifts and lengthens
EXT_OUT = (-0.006, -0.003, 0.0, 0.002, 0.005)   # m: and fans away from the middle finger (right hand; mirrored)
# ---- two hands close, overlapping, crossing (from footage study; notes in the commit message). Two cases:
#  * Overlap / shared register while both play: the hand that is TRAVELLING along the keys while it plays (the
#    passage hand) stays UNDER: low, flat wrist, at the key fronts, its thumb tucked under its palm. The settled
#    hand goes OVER: knuckles ~2.5 cm higher and ~2 cm deeper, wrist lifted so the hand pitches down and the
#    fingers drop steeply past the under hand's knuckles onto the deep end of their keys; its free fingers and
#    thumb ride up and curl in, clear of the under hand.
#  * Leap across: a hand that moves sideways with nothing held (airborne) goes OVER in an arc, 6-9 cm above the
#    keys; the planted hand stays under. The lift begins ~0.3 s before the hands meet and eases out after.
CLOSE_GAP = (0.075, 0.140)          # m between hand centres (xh): fully layered .. not layered
CLOSE_LEAD, CLOSE_HOLD = 0.30, 0.35  # s: layering starts this far ahead of the meeting, and lingers this long after
ROLE_SWITCH = 0.035                 # m/s of travel-score difference needed to swap over/under (hysteresis)
ROLE_EASE = 0.45                    # s: over/under roles cross-fade this slowly when they do swap
OVER_DY, OVER_DZ, OVER_WZ = 0.028, 0.036, 0.014     # m: over hand deeper, higher; extra wrist lift (pitch down)
UNDER_DY, UNDER_DZ, UNDER_WZ = -0.016, -0.012, -0.016
UNDER_YAW = 0.10                    # rad: the under hand turns its fingers away from the other hand
OVER_YAW = 0.35                     # rad: the over hand turns its fingers in, across the other hand (its thumb swings clear)
OVER_ROLL, UNDER_ROLL = 0.30, 0.15  # rad: the over hand rolls its side facing the other hand up; the under hand, down
LEAP_BUNCH = 0.4                    # share by which a leaping hand's free fingertips close toward the middle finger
LEAP_DZ, LEAP_V = 0.060, 0.80       # m: extra arc of a hand leaping over the other; sideways speed (m/s) that is a leap
OVER_FREE_UP = 0.015                # m: the over hand's free fingers lift clear of the under hand ...
OVER_FREE_BACK = 0.0                # m: ... (curling them in as well runs the PIPs into their stops)
TUCK = (-0.050, -0.014, -0.038)     # under hand's free thumb tip, hand-local (x: + toward the little finger): under the index/middle metacarpals, below the knuckles
OVER_THUMB = (-0.045, 0.000, -0.008)    # over hand's free thumb tip: laid along the index, up at knuckle height
MIN_GAP, MAX_PUSH = 0.105, 0.03     # m: working gap the hand centres keep where their notes allow; most they give way
THUMB_GAP = (0.15, 0.20)            # m between hand centres: free thumbs close in to the index .. stay out
THUMB_IN = 0.0                       # how far an idle thumb draws in to lie along the hand
THUMB_TUCK = (0.0, 0.0, 0.0, 0.0)  # rad added to an idle thumb's relaxed pose (cmc abd, cmc flex, mcp, ip): the
                                     # top joint tucks to clear space
THUMB_CLOSE = (-0.054, -0.008, -0.030)  # a free thumb drawn in against the index side when the other hand is near
ROLE_FORCE = None                   # experiment hook: 1.0 = right hand always under, 0.0 = left
CLOSE = {}
FREE_GAP = (0.25, 0.35)           # s after a release / before a strike in which a finger is still working
GLIDE_MAX = 0.8                    # s: a free finger starts drifting toward its next key at most this early
ENSLAVE = (0.45, 0.20)             # share of a pressing neighbour's dip a free finger follows (next, next-but-one)
SPRING_ROLL = (3.0, 0.75, 1.0)
SPRING_HEAD = (1.7, 0.50, 0.9)
LEAD_HAND, LEAD_TIP, LEAD_HEAD = 0.07, 0.018, 0.06
YOFF_NB = 0.018            # m: the hand moves in when black keys are in play
PRONATE = 0.15             # rad: base roll toward the thumb (forearm pronation past flat)
ROLL_MAX = 0.12            # rad: the hand rolls toward the fingers that are playing


def _centre(H, x, nb, t):
    """Raw hand-centre target at time t: placement, attack give, arm weight, breathing."""
    yoff = YOFF_NB if nb else 0.0
    c = np.array([x, Y_KN + yoff, KNUCKLE_Z])
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
PULL_IT = 12               # iterations of the reach pull
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


# ---------------------------------------------------------------- the hand rig (blockout/rig): one solve per hand
THUMB_REST = 4.0                    # an idle thumb follows its resting spot by the keys this much harder
PRESS_AXES = (3.0, 1.0, 1.5)
PRESS_AXES_BLACK = (3.0, 3.0, 1.5)  # a black key starts 55 mm in: a pad short of it misses it, so depth counts too
HAND_CLEAR, HAND_W = 0.003, 1.0    # m kept between the two hands' skin; weight of that against the targets
RIG_W = dict(press=1.0, free=0.15, root_pos=0.02, root_rot=1e-4, env=20.0, comfort=9e-6, comfort_sigma=True, iters=12, smooth=2e-5)   # target weights and the plan prior
_RIGS = {}


def _rig(side):
    if side not in _RIGS:
        from .rig.hand import HandRig
        _RIGS[side] = HandRig(side)
    return _RIGS[side]


def hand_frame_from_plan(sgn, roll, psi, pitch=0.0):
    """Rows across (thumb -> pinky), forward, up (back of hand) for a planned hand: pitch (+ = wrist above the
    knuckles, fingers pointing down), roll toward the thumb about the forward axis, then yaw psi."""
    from .rig.skeleton import _rot
    c, s_ = math.cos(pitch), math.sin(pitch)
    ac0, fw0, up0 = np.array([sgn, 0.0, 0.0]), np.array([0.0, c, -s_]), np.array([0.0, s_, c])
    Rr = _rot(fw0, -roll)
    cy, sy = math.cos(psi), math.sin(psi)
    Ry = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return np.stack([Ry @ Rr @ ac0, Ry @ Rr @ fw0, Ry @ Rr @ up0])


def _hand_blobs(samples, root_rot, root_pos, rig):
    """Spheres covering a posed hand's skin: every finger cross-section (radius ~ its half-width) and the palm
    (spheres from the wrist to each knuckle)."""
    C, R = [], []
    for f in range(5):
        for i, u, c, low, w in samples[f]:
            C.append(c); R.append(w * 0.9)
    for (bR, bH), ch in zip(rig.bases(root_rot, root_pos)[1:], rig.chains[1:]):
        mcp = bH + bR[:, 1] * rig.sk.L[ch.base_bone]
        for u in (0.35, 0.7, 1.0):
            C.append(root_pos + (mcp - root_pos) * u); R.append(0.017)
    return np.array(C), np.array(R)


def _key_env(keydepth, pressing, other=None, rig=None):
    """Collision penalty (m^2) of a posed hand against the keyboard heightfield, from the skin's real cross-
    sections (the lowest point of each bone's off-centre ellipse; a pressing finger's pad may rest on its key),
    and against the other hand's skin (other = its spheres)."""
    DW, DB = _depth_arrays(keydepth)

    def env(samples, root_rot, root_pos):
        if not SOFT_ENV:
            return _env_hard(samples, root_rot, root_pos)
        out = []                                       # one residual per skin sample (and per hand blob), so the
        if other is not None:                          # solver sees which part collides and which way is out
            C, R = _hand_blobs(samples, root_rot, root_pos, rig)
            d = np.linalg.norm(C[:, None, :] - other[0][None, :, :], axis=2)
            ov = np.clip(R[:, None] + other[1][None, :] + HAND_CLEAR - d, 0, None)
            out.extend(math.sqrt(HAND_W) * ov.sum(1))
        S_ = [(f, i, u, c, low, w) for f in range(5) for i, u, c, low, w in samples[f]]
        if S_:
            C = np.array([s_[3] for s_ in S_]); low = np.array([s_[4] for s_ in S_])
            W_ = np.array([s_[5] for s_ in S_])
            d = surface_under_soft_v(C[:, 0], C[:, 1], 0.8 * W_, DW, DB) - low
            d -= np.array([(THUMB_PRESS_GIVE if f == 0 and i >= 1 else OWN_KEY_GIVE) if pressing[f] and (i == 2 and u >= 0.5 or f == 0 and i >= 1)
                           else 0.0 for f, i, u, *_r in S_])   # a pressing thumb lies over the neighbour's edge
            d[low > 0.02] = 0.0                        # well above every key top
            out.extend(np.maximum(d, 0.0))             # (a pressing pad rests on its own lowered key, not a neighbour)
        return np.array(out)

    def _env_hard(samples, root_rot, root_pos):
        pen = 0.0
        if other is not None:
            C, R = _hand_blobs(samples, root_rot, root_pos, rig)
            d = np.linalg.norm(C[:, None, :] - other[0][None, :, :], axis=2)
            ov = np.clip(R[:, None] + other[1][None, :] + HAND_CLEAR - d, 0, None)
            pen += float((ov * ov).sum()) * HAND_W
        for f in range(5):
            for i, u, c, low, w in samples[f]:
                if low > 0.016:
                    continue                           # well above every key top
                d = surface_under(c[0], c[1], 0.8 * w, keydepth) - low
                if pressing[f] and i == 2 and u >= 0.5:
                    d -= 0.001                         # its pad rests on its own (lowered) key, not in a neighbour
                if d > 0:
                    pen += d * d
        return pen
    return env


WRIST_BAND = (0.30, 0.22)   # rad: wrist flexion/extension and sideways deviation (against the forearm) that cost nothing
WRIST_W = 0.05              # weight of the wrist joint past that band (a 10 deg excess ~ a 1 cm pad miss)
WRIST_LAMBDA = 12.0         # smoothing of the wrist's path (Whittaker, 2nd differences; ~2 Hz at 24 fps)
WRIST_IDLE_W = 0.15         # how much a frame with no pressing finger pulls the smoothed wrist to its solve
WRIST_STRIKE_W = 8.0        # ... and how hard the frames where a finger is striking pin it (the arm lands on the chord)
WRIST_HOLD_W = 0.6          # a held note pulls the wrist path less: between strikes the hand leans to the next one
WRIST_STRIKE_T = 0.12       # s: a strike lasts this long from the finger taking its key
WRIST_LAMBDA_Z = 0.25       # the vertical path is this much less smoothed (an arm drop is quick)
WRIST_PASSES = 2            # wrist path refinements: solve the whole hand around the path, smooth, repeat
WRIST_PRIOR = 0.3           # root_pos weight around the refined path (the first pass uses RIG_W's, around the plan)
EASE_ITERS = 2              # ragdoll ease passes (0: off)
EASE_PRIOR = 2.0            # how firmly the wrist's spring holds it to its path during the ease solve (root_pos weight)
EASE_MAX = 0.025            # m: the most the wrist gives per pass
EASE_LAMBDA = 3.0           # how softly the give eases in and out (Whittaker, 2nd differences)
EASE_IDLE_W = 0.05          # frames with no pressing finger barely hold the give at zero
FINGER_REFINE = True        # solve the fingers, smooth their joint paths, solve again held near them
FINGER_LAMBDA = 6.0         # finger joint path smoothing (Whittaker, 2nd differences)
FINGER_PRESS_W = 8.0        # ... pinned this hard on the frames the finger is on its key
FINGER_PRIOR = 5e-3         # how hard a finger holds its smoothed path in the second solve
FINGER_SPLIT = True         # with the wrist set, each finger solves on its own
FREE_FOLLOW = True          # free fingers' float targets move with the solved wrist (across/along the keys)
PRESS_OVERSHOOT = 0.0       # m the pressing pad aims below the key bed (0: at the bed)
REPAIR_ITERS = 0            # verify-and-repair passes: re-solve the fingers with missed notes' presses boosted
REPAIR_W = 4.0              # how much harder a missed note's finger pulls on each repair pass
OWN_KEY_GIVE = 0.001        # m a pressing pad may sink past its own (lowered) key's surface in the collision model
THUMB_PRESS_GIVE = 0.004    # m a pressing thumb's distal and proximal phalanx may sink past key tops (its side
                            # rests over the neighbouring key's edge as it presses its own)
SOFT_ENV = True             # collisions as eased per-sample residuals (False: v16's single hard-edged penalty)
FINGER_ITERS = 20           # finger solve iterations once the wrist is set
FINGER_SMOOTH = 2e-5        # finger joints' frame-to-frame smoothness once the wrist is set


def _whittaker(Y, w, lam):
    """Penalised least squares: sum w (z - y)^2 + sum lam_k (D2 z)_k^2, every column of Y. lam: a number or one
    value per second difference (n - 2), e.g. relaxed where the path has to move fast."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n = len(Y)
    if n < 4 or np.all(np.asarray(lam) <= 0):
        return Y.copy()
    D = sp.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    L = sp.diags(np.broadcast_to(np.asarray(lam, float), (n - 2,)).copy())
    lu = splu((sp.diags(w) + (D.T @ L @ D)).tocsc())
    return np.stack([lu.solve(w * Y[:, k]) for k in range(Y.shape[1])], 1)


WRIST_MODEL = os.environ.get("RIG_WRIST_MODEL", "1") == "1"   # the FürElise wrist model drives the wrist reference
WRIST_MODEL_PRIOR = 0.3     # its pull in the first whole-hand pass (our own plan only got 0.02)
_WM = None


def _wrist_model():
    global _WM
    if _WM is None:
        p = os.path.join(os.path.dirname(__file__), "rig", "furelise_wrist.json")
        _WM = json.load(open(p)) if os.path.exists(p) else {}
    return _WM


def _learned_wrist(H, name, ts):
    """Wrist position per frame time from the FürElise model: placement at each strike group from the struck keys
    and the fingers striking them, and the pianists' travel curve between groups (ballistic for big moves, landing
    early). Long gaps hold, then travel over the last second."""
    M = _wrist_model()
    if not M or not H["fing"]:
        return None
    hs_ = __import__("blockout.rig.skeleton", fromlist=["hand_scale"]).hand_scale()
    k = hs_ / (17.7 / 24.3) if hs_ != 1.0 else 24.3 / 17.7   # model offsets are for a 17.7 cm hand
    on, P = [], []
    for sl in H["fing"]:
        v = []
        for n, f in sl:
            key = f"{name}{f}.{'black' if is_black(n.pitch) else 'white'}"
            o = M["offset"].get(key) or M["offset"].get(f"{name}{f}.white")
            v.append(np.array([key_x(n.pitch), 0.0, 0.0]) + np.array(o["median"]) * np.array([k, k, 1.0]))  # offsets
            # scale with the hand (x and depth), so the knuckles land where a pianist's do whatever the hand's size
        on.append(sl[0][0].start); P.append(np.mean(v, 0))
    on, P = np.array(on), np.array(P)
    tau = np.array(M["travel"]["small"]["tau"])
    cs, cl = np.array(M["travel"]["small"]["median"]), np.array(M["travel"]["large"]["median"])
    out = np.empty((len(ts), 3))
    for i, t in enumerate(ts):
        g = int(np.searchsorted(on, t, side="right")) - 1
        if g < 0:
            out[i] = P[0]; continue
        if g >= len(on) - 1:
            out[i] = P[-1]; continue
        t0, t1 = on[g], on[g + 1]
        t0 = max(t0, t1 - 1.0)
        u = float(np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1))
        d = P[g + 1] - P[g]
        c = cl if np.linalg.norm(d[:2]) > 0.04 else cs
        out[i] = P[g] + d * float(np.interp(u, tau, c))
    return out


LEAP_V = 1e9                # (off: brought the shake back) m/s of the wrist's wanted path past which the smoothing eases off (an arm leaping)


def _leap_lambda(P, fps, lam):
    """Per-step smoothing for a wrist path: full where the hand stays put, down to ~5% where its wanted path
    moves faster than LEAP_V (a leap), eased over a few frames so the relief starts before the move."""
    v = np.linalg.norm(np.gradient(P, axis=0), axis=1) * fps
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    v = uniform_filter1d(maximum_filter1d(v, size=int(0.25 * fps) | 1), size=5)
    k = lam / (1.0 + (v / LEAP_V) ** 2)
    return np.maximum(k, 0.05 * lam)[1:-1]


def _frame_targets(hands, j):
    """Per hand: the pad targets (pressing on their keys, free ones floating) and each struck key's planned depth."""
    targets, est_depth = {}, {}
    for name, H in hands.items():
        tg = H["Tf"][j].copy() - np.array([[0.0, 0.0, r] for r in TIP_R])
        for f in range(5):
            a = H["act"][j][f]
            if a is None:
                continue
            p, kx, cz = a
            top, lev = cz - TIP_R[f], key_lever(p, tg[f, 1])
            tg[f, 2] = max(tg[f, 2] - PRESS_OVERSHOOT, top - KEY_TRAVEL * lev - PRESS_OVERSHOOT)   # a key stops on its
            # bed; with PRESS_OVERSHOOT the pad aims a little below it, so contact never stops short of the bottom
            if abs(tg[f, 0] - kx) < WHITE_W:
                est_depth[p] = max(est_depth.get(p, 0.0), float(np.clip((top - tg[f, 2]) / (KEY_TRAVEL * lev), 0, 1)))
        targets[name] = tg
    return targets, est_depth


def _plan_root(H, name, j):
    """The motion plan's wrist (root) rotation and position at frame j."""
    sgn = 1 if name == "R" else -1
    wz = float(H["WZf"][j]) - 0.004 * float(H["TCH"][j]) if "WZf" in H else 0.0
    M = hand_frame_from_plan(sgn, float(H["Rf"][j]), float(H["Yf"][j]), math.atan2(wz, PALM_LEN) + PITCH0)
    return _rig(name).root_from_hand_frame(M, H["Cf"][j])


def _hand_fwd(name, root_rot, root_pos=None):
    """The hand's forward (wrist -> knuckles) direction for a root rotation."""
    rig = _rig(name)
    return root_rot @ (rig.sk.R0["wrist"].T @ rig.M_rest[1])


def _solve_frame(hands, j, targets, est_depth, prev, roots=None, forearms=None, refs=None, prior=None, qprior=None,
                 boost=None):
    """Both hands at frame j (two passes: each hand sees the other's latest skin). roots: a set wrist per hand
    (x[:6]); then only the fingers move. forearms: (elbow, band) per hand for the wrist joint term. refs: the
    wrist reference (rot, pos) per hand instead of the plan's; prior: its root_pos weight."""
    solved, blobs = {}, {}
    order = list(hands.items())
    for name, H in order + (order if len(order) == 2 else []):
        rig = _rig(name)
        ref_rot, ref_pos = refs[name] if refs else _plan_root(H, name, j)
        pressing = [H["act"][j][f] is not None for f in range(5)]
        if refs and FREE_FOLLOW:                        # free fingers float where the hand is, not where it was
            pr, pp = _plan_root(H, name, j)             # planned: carried across and along with the wrist
            tg = targets[name].copy()
            dR = ref_rot @ pr.T
            for f in range(5):
                if not pressing[f]:
                    v = dR @ (tg[f] - pp) + ref_pos
                    tg[f, :2] = v[:2]
            targets = {**targets, name: tg}
        # a key is narrow across the keyboard and long along it: aim pressing pads hardest in x
        w = [np.array(PRESS_AXES_BLACK if is_black(H["act"][j][f][0]) else PRESS_AXES) * RIG_W["press"] if pressing[f]
             else RIG_W["free"] * (THUMB_REST if f == 0 else 1.0) for f in range(5)]
        if boost:                                       # repair pass: a finger that missed its key here pulls harder
            w = [w[f] * boost.get((name, j, f), 1.0) if pressing[f] else w[f] for f in range(5)]
        x_init = solved[name][0] if name in solved else prev.get(name)
        W = {k: v for k, v in RIG_W.items() if k not in ("press", "free")}
        tk = float(H["WF"][j][0]) if "WF" in H else 0.0
        if not pressing[0] and tk > 0.05:               # off duty: the thumb's relaxed pose tucks, easing in and out
            W["rest_override"] = {0: np.clip(rig.q_rest[:rig.sl[1]] + tk * np.array(THUMB_TUCK)[:rig.sl[1]],
                                             rig.lo[:rig.sl[1]], rig.hi[:rig.sl[1]])}
            W["rest_override_w"] = 1.0 + 29.0 * tk
        if roots is not None:
            q0 = x_init[6:] if x_init is not None else rig.q_rest
            if qprior is not None:                      # start on the smoothed finger path, and stay near it
                q0 = qprior[name][6:]
            x_init = np.concatenate([roots[name], q0])
            W["iters"] = FINGER_ITERS
            W["smooth"] = FINGER_SMOOTH if qprior is None else FINGER_PRIOR
        else:
            W["wrist"] = WRIST_W
            if prior is not None:
                W["root_pos"] = prior
        env = _key_env(est_depth, pressing, blobs.get("L" if name == "R" else "R"), rig)
        if roots is not None and FINGER_SPLIT:          # the wrist is set: five independent finger solves
            x = rig.solve_fingers(ref_rot, x_init, list(targets[name]), w, env=env, W=W,
                                  last=prev.get(name) if qprior is None else qprior[name],
                                  iters=FINGER_ITERS)
            rr, rp, pts = rig.points(x, ref_rot)
            pads = rig.effectors(pts, rig._rots)
            miss = [None if targets[name][f] is None else float(np.linalg.norm(pads[f] - targets[name][f])) for f in range(5)]
        else:
            x, (rr, rp, pts, pads), miss = rig.solve(ref_rot, ref_pos, list(targets[name]), w, prev=x_init,
                                               last=prev.get(name), env=env,
                                               forearm=None if forearms is None else forearms[name],
                                               W=W, fix_root=roots is not None)
        rig.points(x, ref_rot)                          # rotations of the accepted pose (not a trial step)
        solved[name] = (x, rr, rp, pts, pressing, pads, ref_pos, miss)
        blobs[name] = _hand_blobs([ch.samples(P, Rs) for ch, P, Rs in zip(rig.chains, pts, rig._rots)], rr, rp, rig)
    return solved


REACH_COMFORT = 0.80        # share of the arm's length a hand reaches before the torso helps (arms stretch first)
LEAN_K, LEAN_MAX = 2.2, 0.22   # rad of lean toward the far hand per m past that, and the most
TURN_START, TURN_K, TURN_MAX = 0.05, 2.5, 0.35   # m past comfort before the shoulders turn to open toward that end
SPRING_TORSO = (0.9, 1.0, 0.0)  # the torso moves slowly


def _torso_drive(hands, body, N, fps):
    """(pitch, yaw, roll) the torso adds per frame from the hands' reach: nothing while both hands are within
    comfortable reach of their shoulders; past it the torso leans toward the far hand, and further out the
    shoulders turn to open toward that end of the keyboard."""
    sh = body.pose(0.0, 0.0, 0.0, 0.0, {})["shoulders"]
    arm = performer.UPPER_ARM + performer.FOREARM
    T = np.zeros((N, 3))
    for j in range(N):
        far, side = 0.0, 0.0
        for name, H in hands.items():
            c = H["Cf"][j]
            d = float(np.linalg.norm(c - np.asarray(sh[name])))
            ex = d - REACH_COMFORT * arm
            if ex > far:
                far, side = ex, (1.0 if c[0] > body.hips[0] else -1.0)
        if far > 0:
            T[j, 2] = side * min(LEAN_MAX, LEAN_K * far)
            T[j, 1] = -side * min(TURN_MAX, TURN_K * max(0.0, far - TURN_START))
    return dynamics.filter_track(T, fps, *SPRING_TORSO)


def _pass3_rig(hands, times, j0, N, start, energy, HEAD, body, fps, TORSO=None):
    """The hand hangs off its wrist. Pass A solves the whole hand (wrist free, held near its plan and against the
    forearm) to find where the wrist needs to be; that path is smoothed like an arm moves it; pass B sets the wrist
    there and solves only the fingers."""
    from .rig.hand import rotvec
    T = {j: _frame_targets(hands, j) for j in range(j0, N)}
    _bp = {j: body.pose(*HEAD[j], {}, torso=TORSO[j]) for j in range(j0, N)}
    shoulders = {j: _bp[j]["shoulders"] for j in range(j0, N)}
    TORSO_BOX = {j: performer.torso_box(_bp[j]) for j in range(j0, N)}
    REF = {name: [_plan_root(H, name, j) for j in range(j0, N)] for name, H in hands.items()}
    first_prior = None
    if WRIST_MODEL:                                     # where a pianist's wrist goes (FürElise), not our plan
        for name, H in hands.items():
            P = _learned_wrist(H, name, times[j0:N])
            if P is not None:
                REF[name] = [(r[0], p_) for r, p_ in zip(REF[name], P)]
        first_prior = WRIST_MODEL_PRIOR
    PW = {}
    ns = max(1, int(round(WRIST_STRIKE_T * fps)))
    for name, H in hands.items():
        busy = np.array([[a is not None for a in H["act"][j]] for j in range(N)])
        new = busy & ~np.vstack([np.zeros((1, 5), bool), busy[:-1]])        # a finger takes its key
        strike = np.convolve(new.any(1).astype(float), np.ones(ns))[:N] > 0
        PW[name] = np.where(strike, WRIST_STRIKE_W, np.where(busy.any(1), WRIST_HOLD_W, WRIST_IDLE_W))[j0:N]
    for k in range(max(1, WRIST_PASSES)):               # pass A: where the wrist needs to be, then smoothed
        XA = {name: [] for name in hands}
        prev = {}
        for j in range(j0, N):
            refs = {name: REF[name][j - j0] for name in hands}
            forearms = {name: (performer.arm_ik(np.asarray(shoulders[j][name]), refs[name][1], 1 if name == "R" else -1,
                                                _hand_fwd(name, *refs[name]), torso=TORSO_BOX[j]), WRIST_BAND) for name in hands}
            solved = _solve_frame(hands, j, *T[j], prev, forearms=forearms, refs=refs, prior=WRIST_PRIOR if k else first_prior)
            for name in hands:
                prev[name] = solved[name][0]
                XA[name].append(solved[name][0][:6])
        for name in hands:                              # the smoothed path becomes the new reference (offset 0)
            X_ = np.array(XA[name])
            lam = _leap_lambda(X_[:, :3], fps, WRIST_LAMBDA)       # an arm leaps fast: the smoothing lets it
            S_ = _whittaker(X_, PW[name], lam)
            S_[:, 2:3] = _whittaker(X_[:, 2:3], PW[name], lam * WRIST_LAMBDA_Z)
            REF[name] = [(rotvec(S_[i, 3:6]) @ REF[name][i][0], S_[i, :3]) for i in range(len(S_))]
    roots = {name: [np.concatenate([r[1], np.zeros(3)]) for r in REF[name]] for name in hands}

    for _e in range(EASE_ITERS):                        # ragdoll ease: the wrist hangs on a firm spring off its
        prev, DV = {}, {n: np.zeros((N - j0, 3)) for n in hands}    # path; the whole-hand solve lets the pressing
        WE = {n: np.full(N - j0, EASE_IDLE_W) for n in hands}       # fingers pull it as far as they need, and that
        for j in range(j0, N):                                      # give is eased in and out over time
            refs = {name: REF[name][j - j0] for name in hands}
            forearms = {name: (performer.arm_ik(np.asarray(shoulders[j][name]), refs[name][1], 1 if name == "R" else -1,
                                                _hand_fwd(name, *refs[name]), torso=TORSO_BOX[j]), WRIST_BAND) for name in hands}
            solved = _solve_frame(hands, j, *T[j], prev, forearms=forearms, refs=refs, prior=EASE_PRIOR)
            for name in hands:
                x = solved[name][0]
                prev[name] = x
                if any(solved[name][4]):
                    WE[name][j - j0] = 1.0
                    DV[name][j - j0] = np.clip(x[:3] - refs[name][1], -EASE_MAX, EASE_MAX)
        for name in hands:
            C = _whittaker(DV[name], WE[name], EASE_LAMBDA)
            for i in range(N - j0):
                REF[name][i] = (REF[name][i][0], REF[name][i][1] + C[i])
        roots = {name: [np.concatenate([r[1], np.zeros(3)]) for r in REF[name]] for name in hands}
    QP = None
    if FINGER_REFINE:                                   # the fingers once, then their joint paths smoothed: in pass B
        XB = {name: [] for name in hands}               # each finger stays near its smoothed path (no shape it has
        prev = {}                                       # spare wanders frame to frame), the notes still pull it on
        for j in range(j0, N):
            solved = _solve_frame(hands, j, *T[j], prev, roots={n: roots[n][j - j0] for n in hands},
                                  refs={n: REF[n][j - j0] for n in hands})
            for name in hands:
                prev[name] = solved[name][0]
                XB[name].append(solved[name][0])
        QP = {}
        for name, H in hands.items():                   # each finger's path is pinned where it is on its key, so only
            X_ = np.array(XB[name])                     # its travel and its spare wandering are smoothed
            rig = _rig(name)
            for f in range(5):
                a, b = 6 + rig.sl[f], 6 + rig.sl[f + 1]
                w = np.array([FINGER_PRESS_W if H["act"][j][f] is not None else 1.0 for j in range(j0, N)])
                X_[:, a:b] = np.clip(_whittaker(X_[:, a:b], w, FINGER_LAMBDA), rig.lo[a - 6:b - 6], rig.hi[a - 6:b - 6])
            QP[name] = X_

    BOOST = {}
    for rep in range(REPAIR_ITERS + 1):
        for H in hands.values():
            for k in ("tip_miss", "pen", "pen_info"):
                H.pop(k, None)
        frames = _pass_b(hands, times, j0, N, start, energy, HEAD, body, T, roots, REF, QP, TORSO, BOOST)
        if rep == REPAIR_ITERS:
            break
        nmiss = 0                                       # verify: every note's key at least half down within 0.25 s
        for name, H in hands.items():
            for f in range(5):
                for e in H["ev"][f]:
                    a, b = int(round((e[0] - times[j0]) * fps)), int(round((e[0] + 0.25 - times[j0]) * fps))
                    if a < 0 or a >= len(frames):
                        continue
                    d = max(frames[k]["keys"].get(str(e[2]), 0.0) for k in range(a, min(b + 1, len(frames))))
                    if d >= 0.5:
                        continue
                    nmiss += 1
                    for k in range(max(0, a - int(0.15 * fps)), min(len(frames), b + 1)):
                        BOOST[(name, j0 + k, f)] = BOOST.get((name, j0 + k, f), 1.0) * REPAIR_W
        if not nmiss:
            break
    return frames


def _pass_b(hands, times, j0, N, start, energy, HEAD, body, T, roots, REF, QP, TORSO, BOOST):
    frames = []
    prev = {}
    for j in range(j0, N):                              # pass B: the fingers, from the set wrist
        t = times[j]
        fr = {"t": round(float(t - start), 4), "keys": {}, "hands": {}, "energy": round(float(energy[j]), 3)}
        targets, est_depth = T[j]
        solved = _solve_frame(hands, j, targets, est_depth, prev, roots={n: roots[n][j - j0] for n in hands},
                              refs={n: REF[n][j - j0] for n in hands},
                              qprior=None if QP is None else {n: QP[n][j - j0] for n in hands}, boost=BOOST)
        keydepth, press = {}, {}
        for name, H in hands.items():
            x, rr, rp, pts, pressing, pads, ref_pos, miss = solved[name]
            prev[name] = x
            for f in range(5):                           # a key goes down only as far as the real finger pushes it
                a = H["act"][j][f]
                if a is None:
                    continue
                p, kx, cz = a
                pad = pads[f]
                y0 = WHITE_L - BLACK_L if is_black(p) else 0.0
                if abs(pad[0] - kx) < (BLACK_W if is_black(p) else WHITE_W) / 2 + 0.002 and y0 - 0.002 <= pad[1] <= WHITE_L:
                    d = float(np.clip((cz - TIP_R[f] - pad[2]) / (KEY_TRAVEL * key_lever(p, pad[1])), 0, 1))
                    if d > 0.01:
                        keydepth[p] = max(keydepth.get(p, 0.0), d)
                        press[str(p)] = f"{name}{f + 1}"
                H.setdefault("tip_miss", []).append((float(miss[f]), f))
        for name, (x, rr, rp, pts, pressing, pads, ref_pos, miss) in solved.items():
            H = hands[name]
            for f in range(5):
                pen = _penetration(pts[f], FINGER_R[f], keydepth, skip_tip=pressing[f])
                H.setdefault("pen", []).append(pen)
                H.setdefault("pen_info", []).append((pen, f, pressing[f], 0.0, j, name))
            fr["hands"][name] = {"wrist": [round(float(c), 5) for c in rp],
                                 "fingers": [[[round(float(c), 5) for c in q] for q in P] for P in pts],
                                 "pads": [[round(float(c), 5) for c in q] for q in pads],
                                 **({"targets": np.round(targets[name], 5).tolist(),
                                     "plan0": np.round(_plan_root(hands[name], name, j)[1], 5).tolist(),
                                     "xh": round(float(hands[name]["xh"][j]), 5), "C": np.round(hands[name]["C"][j], 5).tolist(),
                                     "Cf": np.round(hands[name]["Cf"][j], 5).tolist()} if os.environ.get("RIG_DBG") else {}),
                                 "root": {"rot": np.round(rr, 6).tolist(), "pos": [round(float(c), 6) for c in rp],
                                          "plan": [round(float(c), 6) for c in ref_pos]},
                                 "angles": {b: {k: round(v, 5) for k, v in d.items()} for b, d in _rig(name).angles(x).items()}}
        pitch, yaw, roll_h, lean = HEAD[j]
        fr["body"] = body.pose(pitch, yaw, roll_h, lean, {h: fr["hands"][h]["wrist"] for h in fr["hands"]},
                               {h: _hand_fwd(h, solved[h][1]) for h in fr["hands"]}, torso=TORSO[j])
        for h in fr["hands"]:
            fr["hands"][h]["elbow"] = fr["body"]["elbows"][h]
        fr["keys"] = {str(p): round(d, 3) for p, d in keydepth.items()}
        fr["press"] = press
        frames.append(fr)
    return frames


def _box(x, n):
    """Centred moving average over 2n+1 frames (edges held)."""
    if n < 1:
        return np.asarray(x, float)
    return np.convolve(np.pad(x, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), "valid")


def _close_roles(hands, times, fps):
    """Two hands close: per frame and hand, how much it is the OVER hand and the UNDER hand (0..1, proximity
    included), and how much of its lift is a leap. The under hand is the one travelling along the keys while it
    plays; a hand that moves with nothing held (a leap) or that sits still goes over. Roles have hysteresis and
    only swap while the hands are apart (or on a clear leap), so the hands never pass through each other."""
    N = len(times)
    gap = hands["R"]["xh"] - hands["L"]["xh"]
    raw = np.clip((CLOSE_GAP[1] - gap) / (CLOSE_GAP[1] - CLOSE_GAP[0]), 0, 1)
    a, b = int(round(CLOSE_HOLD * fps)), int(round(CLOSE_LEAD * fps))
    pad = np.pad(raw, (a, b), mode="edge")
    near = np.array([pad[j:j + a + b + 1].max() for j in range(N)])     # look ahead (anticipate) and linger
    near = _smooth(_box(near, int(0.15 * fps)))
    score, leap = {}, {}
    for n, H in hands.items():
        v = np.abs(np.gradient(H["xh"])) * fps
        th = times + LEAD_HAND                                     # the clock xh runs on
        held = np.zeros(N, bool)
        if len(H["on"]):
            i = np.searchsorted(H["on"], th, side="right") - 1
            held = (i >= 0) & (H["end"][np.clip(i, 0, None)] >= th - 0.05)
        held &= v < LEAP_V                                        # this fast, the hand is in the air whatever rings on
        rate = (np.searchsorted(H["on"], times + 0.5) - np.searchsorted(H["on"], times - 0.5)).astype(float)
        score[n] = _box(np.where(held, v, -1.5 * v), int(0.3 * fps)) + 0.008 * rate
        leap[n] = np.clip(_box(np.where(held, 0.0, v), int(0.2 * fps)) / LEAP_V, 0, 1)
    d = score["R"] - score["L"]
    r = 1.0 if float((d * near).sum()) >= 0 else 0.0            # 1: the right hand is under
    role = np.empty(N)
    for j in range(N):
        if near[j] < 0.5 or abs(d[j]) > 3 * ROLE_SWITCH:
            if d[j] > ROLE_SWITCH:
                r = 1.0
            elif d[j] < -ROLE_SWITCH:
                r = 0.0
        role[j] = r
    if ROLE_FORCE is not None:
        role[:] = ROLE_FORCE
    role = _smooth(_box(role, int(ROLE_EASE * fps / 2)))
    tc = np.clip((THUMB_GAP[1] - gap) / (THUMB_GAP[1] - THUMB_GAP[0]), 0, 1)
    pad = np.pad(tc, (a, b), mode="edge")
    tc = _smooth(_box(np.array([pad[j:j + a + b + 1].max() for j in range(N)]), int(0.15 * fps)))
    out = {"R": {"under": near * role, "over": near * (1 - role), "tclose": tc},
           "L": {"under": near * (1 - role), "over": near * role, "tclose": tc}}
    for n in out:
        out[n]["leap"] = out[n]["over"] * leap[n]
        out[n]["away"] = np.sign(hands[n]["xh"] - hands["L" if n == "R" else "R"]["xh"] + 1e-9)
    return out, near, role


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

    if "R" in hands and "L" in hands:
        roles, near, role = _close_roles(hands, times, fps)
        CLOSE.update(near=near, role=role, times=times)
    else:
        roles = {n: {k: np.zeros(N) for k in ("under", "over", "leap", "tclose")} | {"away": np.ones(N)} for n in hands}

    # ---- pass 1: raw targets (read slightly ahead to cancel spring lag)
    for name, H in hands.items():
        sgn = 1 if name == "R" else -1
        CL = roles[name]
        C = np.zeros((N, 3)); ROLL = np.zeros(N); TIP = np.zeros((N, 5, 3)); act = [[None] * 5 for _ in range(N)]
        TCH = np.zeros(N); YAW = np.zeros(N); FREE = np.zeros((N, 5))
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
                loc = np.array([sgn * HOME[f], 0.5 * sum(HITPAD_W[f]) + yoff - ct[1], 0.0])
                home = ct + _yaw(loc, psi0)
                home[2] = surface_under(home[0], home[1], TIP_R[f], {}) + TIP_R[f] + hover
                tip = home
                eo = H["ev_on"][f]
                lo, hi = max(0, int(np.searchsorted(eo, t - 8.0))), int(np.searchsorted(eo, t + 0.3))
                cand = H["ev"][f][lo:hi][::-1]
                # a note that is sounding (or being struck) now outranks the next note's approach: with fast repeated
                # notes the next approach window opens before the current note is released
                cand = sorted(cand, key=lambda e: 0 if e[0] - strike_time(e[5]) <= t <= e[1] else 1)
                for e in cand:
                    tp, dep = finger_tip(t, e, home, TIP_R[f], f)
                    if tp is not None:
                        tip = tp
                        c = contact(e[2], e[4], f) + np.array([0, 0, TIP_R[f]])
                        if CL["over"][j] > 0 and not is_black(e[2]):     # over hand: in deep, past the under hand
                            tip = tip.copy()
                            tip[1] += CL["over"][j] * max(0.0, hitpad(e[2], f)[1] - 0.002 - c[1])
                        act[j][f] = (e[2], float(c[0]), float(c[2]))
                        if e[0] - 0.15 <= t <= e[1]:
                            playing.append(f)
                        break
                if act[j][f] is None:                         # free: float just over the keys, gliding to the next note
                    i_n = int(np.searchsorted(eo, t))
                    w = 0.0
                    if i_n < len(eo):
                        en = H["ev"][f][i_n]
                        prev_end = H["ev"][f][i_n - 1][1] if i_n else t - 10.0
                        g0 = max(prev_end, en[0] - GLIDE_MAX)
                        g1 = en[0] - strike_time(en[5]) - 0.10
                        if g1 > g0 and t > g0:
                            w = _smooth(min(1.0, (t - g0) / (g1 - g0)))
                            goal = contact(en[2], en[4], f)
                            tip = tip.copy()
                            tip[:2] += w * (goal[:2] - tip[:2])
                    idle = (1.0 - w) * (0.0 if f == 0 else 1.0)  # off duty, a finger eases out and up, very softly;
                    # the thumb doesn't: an idle thumb rests low along the key edge
                    tip = tip.copy() + _yaw(np.array([sgn * EXT_OUT[f], EXT_FWD, 0.0]), psi0) * idle
                    tip[2] = surface_under(tip[0], tip[1], TIP_R[f], {}) + TIP_R[f] + hover * (0.4 if f == 0 else 1.0) + EXT_UP * idle
                    uw, ow = float(CL["under"][j]), float(CL["over"][j])
                    idle_t = 1.0 - w                          # (the thumb's own off-duty share; `idle` is 0 for it)
                    if f == 0:                                # an idle thumb lies in along the hand, low and parallel
                        tk = ct + _yaw(np.array([sgn * THUMB_CLOSE[0], THUMB_CLOSE[1], THUMB_CLOSE[2]]), psi0)
                        tk[2] = max(tk[2], surface_under(tk[0], tk[1], TIP_R[f], {}) + TIP_R[f] + 0.004)
                        tip = tip + max(float(CL["tclose"][j]), THUMB_IN) * idle_t * (tk - tip)
                    if f == 0 and uw > 0:                     # under hand: the thumb tucks in under the palm
                        tk = ct + _yaw(np.array([sgn * TUCK[0], TUCK[1], TUCK[2]]), psi0)
                        tk[2] = max(tk[2], surface_under(tk[0], tk[1], TIP_R[f], {}) + TIP_R[f] + 0.006)
                        tip = tip + uw * idle_t * (tk - tip)
                    if ow > 0:                                # over hand: fingers ride up and curl in, thumb closes
                        if f == 0:
                            tk = ct + _yaw(np.array([sgn * OVER_THUMB[0], OVER_THUMB[1], OVER_THUMB[2]]), psi0)
                            tip = tip + ow * idle * (tk - tip)
                        else:
                            tip = tip + ow * (_yaw(np.array([0.0, -OVER_FREE_BACK, 0.0]), psi0) + np.array([0, 0, OVER_FREE_UP]))
                TIP[j, f] = tip
            for f in range(1, 5):                              # shared tendons: free fingers sink with a pressing neighbour
                if act[j][f] is not None:
                    continue
                dz = 0.0
                for g, share in ((f - 1, ENSLAVE[0]), (f + 1, ENSLAVE[0]), (f - 2, ENSLAVE[1]), (f + 2, ENSLAVE[1])):
                    if 1 <= g <= 4 and act[j][g] is not None:
                        base = surface_under(TIP[j, g, 0], TIP[j, g, 1], TIP_R[g], {}) + TIP_R[g] + hover
                        dz = max(dz, share * max(0.0, base - TIP[j, g, 2]))
                floor = surface_under(TIP[j, f, 0], TIP[j, f, 1], TIP_R[f], {}) + TIP_R[f] + 0.001
                TIP[j, f, 2] = max(floor, TIP[j, f, 2] - dz)
            # off duty = no note now AND none just before or coming up: between fast repeated notes a finger stays
            # on its firm spring (it is working, not idling)
            fr_ = []
            for f in range(5):
                eo = H["ev_on"][f]
                i_n = int(np.searchsorted(eo, t))
                near = (i_n < len(eo) and eo[i_n] - t < FREE_GAP[1]) or \
                       (i_n > 0 and t - H["ev"][f][i_n - 1][1] < FREE_GAP[0])
                fr_.append(act[j][f] is None and not near)
            FREE[j] = fr_
            ROLL[j] = sgn * PRONATE                           # the resting hand leans toward the thumb
            if playing:
                ROLL[j] += sgn * ROLL_MAX * (2 - np.mean(playing)) / 2
                d = np.mean([TIP[j, f] for f in playing], axis=0) - C[j]
                psi_reach = 0.5 * math.atan2(d[0] - sgn * np.mean([HOME[f] for f in playing]) * sgn, max(d[1] + 0.06, 0.03))
            else:
                psi_reach = 0.0
            YAW[j] = float(np.clip(0.6 * psi_arm + 0.4 * psi_reach, -YAW_MAX, YAW_MAX))
        H["C"], H["ROLL"], H["TIP"], H["act"], H["TCH"], H["YAW"], H["FREE"] = C, ROLL, TIP, act, TCH, YAW, FREE

    # ---- two hands close: the over hand rises, goes deeper and pitches down (wrist up); the under hand drops a
    # little, flattens, comes to the key fronts and turns its fingers away; a leaping over hand arcs higher.
    # Free fingertips travel with their hand. All of it is eased by the hand springs below.
    if "R" in hands and "L" in hands:              # first they make room sideways where their notes allow
        push = np.clip(MIN_GAP - (hands["R"]["C"][:, 0] - hands["L"]["C"][:, 0]), 0, MAX_PUSH) / 2
        for name, sg in (("R", 1), ("L", -1)):
            H = hands[name]
            H["C"][:, 0] += sg * push
            H["TIP"][:, :, 0] += np.where(H["FREE"] > 0.5, sg * push[:, None], 0.0)
    for name, H in hands.items():
        o, u, lp = roles[name]["over"], roles[name]["under"], roles[name]["leap"]
        dy = o * OVER_DY + u * UNDER_DY
        dz = o * OVER_DZ + u * UNDER_DZ + lp * LEAP_DZ
        H["C"][:, 1] += dy
        H["C"][:, 2] += dz
        free = H["FREE"] > 0.5
        H["TIP"][:, :, 1] += np.where(free, dy[:, None], 0.0)
        H["TIP"][:, :, 2] += np.where(free, dz[:, None], 0.0)
        mid = H["TIP"][:, 2:3, :2].copy()                          # in the air the fingers close up, bunched
        H["TIP"][:, :, :2] += np.where(free[:, :, None], LEAP_BUNCH * lp[:, None, None] * (mid - H["TIP"][:, :, :2]), 0.0)
        H["WZ"] = o * OVER_WZ + u * UNDER_WZ + lp * 0.5 * OVER_WZ
        H["YAW"] = np.clip(H["YAW"] + roles[name]["away"] * (u * UNDER_YAW - o * OVER_YAW), -YAW_MAX, YAW_MAX)
        H["ROLL"] = H["ROLL"] + roles[name]["away"] * (u * UNDER_ROLL - o * OVER_ROLL)

    # ---- pass 2: springs + organic drift
    for k, (name, H) in enumerate(hands.items()):
        H["Cf"] = dynamics.filter_track(H["C"], fps, *SPRING_HAND)
        H["Cf"] += np.stack([dynamics.drift(times, a, seed * 10 + 3 * k + i) for i, a in enumerate((0.0015, 0.0015, 0.0012))], 1)
        H["Rf"] = dynamics.filter_track(H["ROLL"], fps, *SPRING_ROLL) + dynamics.drift(times, 0.02, seed * 10 + 7 + k)
        H["Yf"] = dynamics.filter_track(H["YAW"], fps, *SPRING_ROLL) + dynamics.drift(times, 0.015, seed * 10 + 9 + k)
        H["WZf"] = dynamics.filter_track(H.get("WZ", np.zeros(N)), fps, *SPRING_HAND)
        firm = dynamics.filter_track(H["TIP"].reshape(N, 15), fps, *SPRING_TIP).reshape(N, 5, 3)
        soft = dynamics.filter_track(H["TIP"].reshape(N, 15), fps, *SPRING_FREE).reshape(N, 5, 3)
        wf = np.clip(dynamics.filter_track(H["FREE"], fps, 6.0, 1.0, 1.0), 0, 1)[:, :, None]
        H["Tf"] = wf * soft + (1 - wf) * firm          # a finger off duty goes limp: it trails, floats and settles
        H["WF"] = wf[:, :, 0]                          # (the eased off-duty share, per finger)
    TORSO = _torso_drive(hands, body, N, fps)
    HEAD = np.array([performer.head_angles(t, beats, downbeats, float(energy[j]), sched) for j, t in enumerate(th)])
    HEAD = dynamics.filter_track(HEAD, fps, *SPRING_HEAD)
    HEAD[:, 1] += dynamics.drift(times, 0.03, seed * 10 + 11)
    HEAD[:, 2] += dynamics.drift(times, 0.02, seed * 10 + 12)

    # ---- pass 3: pose, IK, keys from the sprung fingertips
    frames = []
    j0 = int(round(pre * fps))
    if HAND_MODEL == "mpfb":
        frames = _pass3_rig(hands, times, j0, N, start, energy, HEAD, body, fps, TORSO)
    for j in (range(j0, N) if HAND_MODEL != "mpfb" else ()):
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
            for _ in range(PULL_IT):                         # pressing fingers pull the hand within reach
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
                    if THUMB is not None:
                        wr = centre + _yaw(_roll_offsets(WRIST_OFF * np.array([sgn, 1, 1]), roll * 0.5), psi)
                        pts, miss = finger_ik(f, centre + kn_off[f], tips[name][f], wr,
                                              [centre + kn_off[q] for q in range(1, 5)], name)
                        if miss > 0.001:
                            pull += tips[name][f] - pts[3]; n += 1
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
                step[1] = np.clip(centre[1] + step[1], -0.07, KN_FWD) - centre[1]  # knuckles stay behind the key fronts
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
            w_off = _yaw(_roll_offsets(WRIST_OFF * np.array([sgn, 1, 1]) - np.array([0, 0, 0.004 * H["TCH"][j] - H["WZf"][j]]), roll * 0.5), psi)
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
                        if THUMB is not None:                      # real joints for the fingers too
                            return finger_ik(f, kn, tip, wrist, [centre + kn_off[q] for q in range(1, 5)], name,
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


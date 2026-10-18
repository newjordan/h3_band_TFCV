"""Working piano: the animated hands play a simulated grand action, and what the action plays is the sound.

piano.py moves each key only with the finger the plan gives it, and the audio has been the written MIDI. Here
the keyboard listens to the hands instead:

- Contact. The hands' skin is rebuilt from each frame's joint angles with the rig (rig/ik.py cross-sections, the
  same ones the solver collides with; the pads match the take's to 0.01 mm): every bone's lowest skin at
  u = 0.2/0.5/0.8/1.0 plus the fingertip pad, each a sphere as wide as the section. Whatever skin is in a key's
  top presses it, at the depth along the key it touches, so the edge of a finger can catch the next key. Between
  the 24 fps frames positions are PCHIP-interpolated at SIM_HZ (monotone: no made-up overshoot through the key bed).
- Key. A lever on a grand's balance rail (BALANCE behind the key fronts; a press at depth y moves the front
  1 / lever(y) times as far; the renderer tilts keys about the back of the visible key instead, which would let a
  touch near the fallboard throw the front all the way down). It has touch weight: it goes down only under the
  down-weight (DW, ~50 g at the front), stays down while the finger holds it with more than the up-weight (UW, ~25 g),
  and between the two friction holds it where it is. The fingertip pulp is a spring (K_PAD), so "pressure" is the
  pulp compression times K_PAD. The key bed stops a finger: skin deeper than PULP_MAX below a bed lifts that whole
  finger (the take aims pressing pads PRESS_OVERSHOOT below the bed), and it falls back no faster than V_FALL. Let go,
  the key rises on its own weight (at most A_RETURN) and the felt key bed stops it at KEY_TRAVEL.
- Contact rule. A finger pushes a key only after coming down onto it from above; skin that slides onto a key while
  already below its top went in through the key's side (an animation fault), and is counted ("through key sides"),
  not played.
- Hammer. It rides the key (RATIO) and lets off LETOFF before the string. After let-off, or when the key slows
  faster than gravity would slow it (a thrown stroke), it flies; it sounds the string only if it gets there.
  A slow press therefore sounds nothing. Its speed at the string sets the MIDI velocity. After a blow the jack is
  out from under it until the key rises above U_RESET (repetition), so one stroke sounds once.
- Damper. Off the string above U_DAMP_UP, back on below U_DAMP_DN: the note sounds from the blow until its damper
  drops (then the sustain pedal, which is the same score-driven pedal the song uses, blockout/sampler.py).

The velocity curve (hammer m/s -> MIDI velocity) is one global log line, fit on the matched notes so the model is
heard on a piano voiced for its touch; the fit's r is reported (it is the only thing fit to the score).

    python -m blockout.action SUITE.json NOTES.json OUT_DIR [--sfz SFZ] [--t0 S --t1 S] [--no-audio]

OUT_DIR gets: played.json (every blow: t_on, t_off, pitch, vel, hammer m/s, who), keys_phys.json (per frame key
depth 0..1 and finger force N, for a renderer that should show the keys the physics moved), report.json /
report.txt (per piece: sounded, missed, wrong, timing, velocity, holds), and with --sfz the song and the model
through the same sampler and pedal: song.wav, model.wav, song_dry.wav, model_dry.wav (no pedal).
"""
import json, os, sys

import numpy as np
from scipy.interpolate import PchipInterpolator

from .piano import WHITE_W, WHITE_L, BLACK_W, BLACK_L, BLACK_H, KEY_TRAVEL, LOW, HIGH, is_black, key_x, _MID_C

SIM_HZ = 1000
K_PAD = 1000.0          # N/m: fingertip pulp stiffness (~1 N/mm)
DW, UW = 0.49, 0.245    # N at the key front: down-weight 50 g, up-weight 25 g (a regulated grand)
A_RETURN = 6.0          # m/s^2: fastest free rise of a let-go key (~58 ms from the key bed to the top)
RATIO = 5.0             # hammer travel per key-front travel
BLOW = 0.047            # m: hammer rest to string
LETOFF = 0.0025         # m: the jack escapes this far before the string
G_HAMMER = 9.81         # m/s^2: a free hammer falls back like a weight
REBOUND = 0.5
U_RESET = 0.6 * KEY_TRAVEL      # the jack resets under the hammer when the key rises above this (repetition)
U_DAMP_UP, U_DAMP_DN = 0.45 * KEY_TRAVEL, 0.35 * KEY_TRAVEL
SECTIONS = {"Debussy": (39.9, 79.9), "Rach": (79.9, 127.0), "Chopin": (127.0, 390.5), "suite": (39.9, 390.5)}

PITCHES = np.arange(LOW, HIGH + 1)
NK = len(PITCHES)
BLK = np.array([is_black(p) for p in PITCHES])
KX = np.array([key_x(p) for p in PITCHES])
KW = np.where(BLK, BLACK_W, WHITE_W)
KY0 = np.where(BLK, WHITE_L - BLACK_L, 0.0)
KTOP = np.where(BLK, BLACK_H, 0.0)
KL = np.where(BLK, BLACK_L, WHITE_L)
BALANCE = 0.25          # m: balance-rail pivot behind the white key fronts (a grand's key is a long lever; the renderer
                        # tilts keys about the back of the visible key, which would make a touch near the fallboard
                        # throw the key front down 10x as far)


def lever(k, y):
    """How far key k's top drops at depth y per unit of key-front travel (1 at its front edge)."""
    return np.clip((BALANCE - y) / (BALANCE - KY0[k]), 0.05, 1.0)
WHITE_IDX = np.nonzero(~BLK)[0]          # key index of the i-th white key
BLACK_IDX = np.nonzero(BLK)[0]
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def name(p):
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def spheres(anim):
    """The hands' skin as contact spheres per frame, rebuilt from each frame's joint angles with the rig (the same
    skin cross-sections the solver collides with): for every bone of every finger, the lowest skin point of its
    cross-section at u = 0.2, 0.5, 0.8, 1.0, plus the fingertip pad (the effector). A sphere's radius is the
    section's half-width and it sits on that lowest point. Returns (F, S, 3) centres, (S,) radii, S ids."""
    os.environ.setdefault("HAND_SCALE", "1")
    os.environ["RIG_THUMB_AXIS"] = anim.get("thumb_axis", "rest")
    from .rig.hand import HandRig
    fr = anim["frames"]
    cols, rad, ids = [], [], []
    for h in "RL":
        if h not in fr[0]["hands"]:
            continue
        rig = HandRig(h)
        rig.sk.thumb_axis = anim.get("thumb_axis", "rest")
        pos = {}                                         # (f, part) -> list of points over frames
        rr = {}
        for j, f_ in enumerate(fr):
            H = f_["hands"][h]
            R0, P0 = np.array(H["root"]["rot"]), np.array(H["root"]["pos"])
            A = H["angles"]
            for f, (ch, (bR, bH)) in enumerate(zip(rig.chains, rig.bases(R0, P0))):
                q = np.array([A[b][n] for b, n, *_r in ch.dofs])
                P, Rs = ch.fk(bR, bH, q)
                for i in range(3):
                    for u in (0.2, 0.5, 0.8, 1.0):
                        w = float(ch._sect_at(i, min(u, 0.95))[0])
                        pos.setdefault((f, f"b{i}u{u}"), []).append(ch.support(P, Rs, i, u) + np.array([0, 0, w]))
                        rr[(f, f"b{i}u{u}")] = w
                w = float(ch._sect_at(2, 0.8)[0])
                pos.setdefault((f, "pad"), []).append(ch.effector(P, Rs) + np.array([0, 0, w]))
                rr[(f, "pad")] = w
        for (f, part), pts in pos.items():
            cols.append(np.array(pts)); rad.append(rr[(f, part)]); ids.append((h, f, part))
    return np.stack(cols, 1), np.array(rad), ids


SIDE_EPS = 0.0005       # m: a sphere that arrives over a key already this far below its top came in from the side
SIDE_RULE = os.environ.get("ACTION_SIDE", "1") == "1"
V_FALL = 0.3            # m/s: the fastest a finger the key bed held up comes back down
PULP_MAX = 0.003        # m: the most a fingertip squashes against a bottomed key; past that the key bed stops the finger
                        # (the take aims pressing pads PRESS_OVERSHOOT below the bed; a real finger can't go there)


def _contacts(C, r, carry, fid=None):
    """For sphere centres C (n, S, 3) and radii r (S,): per (step, key) the contact that pushes the key front
    deepest: a = how far below the key's rest top the sphere reaches (m), lev = key_lever there, sid = sphere.
    A sphere only pushes a key it came down onto from above: one that slides onto a key while already below its
    top went in through the key's side (an animation fault, not a key press); it is counted, not played, until it
    has been above the key again. carry = {(s, k): came-from-above} for pairs in reach at the previous step;
    updated in place. Returns dense (n, NK) a, lev, sid and a list of side entries (step, s, k, depth m)."""
    n, S, _ = C.shape
    A = np.zeros((n, NK)); LV = np.ones((n, NK)); SID = -np.ones((n, NK), int)
    near = (C[..., 2] - r[None]) < BLACK_H + 1e-3
    ti, si = np.nonzero(near)
    if not len(ti):
        carry.clear()
        carry["lift"] = np.zeros(int(fid.max()) + 1) if fid is not None else None
        return A, LV, SID, []
    x, y, z, rr = C[ti, si, 0], C[ti, si, 1], C[ti, si, 2], r[si]
    wi = np.floor((x + _MID_C) / WHITE_W).astype(int)
    bk = np.searchsorted(KX[BLACK_IDX], x)
    over_black = np.zeros(len(x), bool)
    kb = np.full(len(x), -1)
    for d in (-1, 0):
        k = BLACK_IDX[np.clip(bk + d, 0, len(BLACK_IDX) - 1)]
        ob = (np.abs(x - KX[k]) < BLACK_W / 2) & (y >= KY0[k]) & (y <= WHITE_L)
        over_black |= ob
        kb[ob] = k[ob]
    if fid is not None and PULP_MAX is not None:      # the key bed stops a finger: lift each finger by how far its
        ku = np.where(over_black, kb, WHITE_IDX[np.clip(wi, 0, len(WHITE_IDX) - 1)])     # skin is below a bed
        on = (y >= 0) & (y <= WHITE_L)                  # past what the pulp can squash
        bed = KTOP[ku] - KEY_TRAVEL * lever(ku, np.clip(y, KY0[ku], WHITE_L))
        pen = np.where(on, bed - (z - rr) - PULP_MAX, 0.0)
        nf = int(fid.max()) + 1
        lift = np.zeros((n, nf))
        np.maximum.at(lift, (ti, fid[si]), pen)
        prev = carry.pop("lift", np.zeros(nf))       # a bed pushes a finger up at once; it falls back only as fast
        fall = V_FALL / SIM_HZ                         # as a hand does (no teleport off a black key's edge)
        for i in range(n):
            prev = np.maximum(lift[i], prev - fall)
            lift[i] = prev
        carry["lift"] = prev
        z = z + lift[ti, fid[si]]
    T, Sx, K, Aa, L = [], [], [], [], []
    for k in [WHITE_IDX[np.clip(wi + d, 0, len(WHITE_IDX) - 1)] for d in (-1, 0, 1)] + \
             [BLACK_IDX[np.clip(bk + d, 0, len(BLACK_IDX) - 1)] for d in (-1, 0)]:
        dx = np.maximum(np.abs(x - KX[k]) - KW[k] / 2, 0.0)
        dy = np.maximum(np.maximum(KY0[k] - y, y - WHITE_L), 0.0)
        d2 = dx * dx + dy * dy
        ok = (d2 < rr * rr) & ~(over_black & ~BLK[k])          # in reach of the key top
        yc = np.clip(y, KY0[k], WHITE_L)
        lev = lever(k, yc)
        a = KTOP[k] - (z - np.sqrt(np.maximum(rr * rr - d2, 0.0)))
        T.append(ti[ok]); Sx.append(si[ok]); K.append(k[ok]); Aa.append(a[ok]); L.append(lev[ok])
    T, Sx, K, Aa, L = (np.concatenate(v) for v in (T, Sx, K, Aa, L))
    # the same (step, sphere, key) can come from two candidate lists: keep one
    key = (T.astype(np.int64) * (S + 1) + Sx) * NK + K
    key, first = np.unique(key, return_index=True)
    T, Sx, K, Aa, L = T[first], Sx[first], K[first], Aa[first], L[first]
    # runs of consecutive steps per (sphere, key) pair; within a run the pair pushes only once it has been above
    pair = Sx.astype(np.int64) * NK + K
    o = np.lexsort((T, pair))
    T, Sx, K, Aa, L, pair = T[o], Sx[o], K[o], Aa[o], L[o], pair[o]
    new_run = np.ones(len(T), bool)
    new_run[1:] = (pair[1:] != pair[:-1]) | (T[1:] != T[:-1] + 1)
    above = Aa <= SIDE_EPS if SIDE_RULE else np.ones(len(Aa), bool)
    start = carry.pop("start", False)                         # the take's first step: whatever touches, holds
    init = np.array([(start or carry.get((int(s_), int(k_)), False)) if t_ == 0 else False
                     for t_, s_, k_ in zip(T[new_run], Sx[new_run], K[new_run])], bool)
    run = np.cumsum(new_run) - 1
    flag = above.copy()
    flag[np.nonzero(new_run)[0]] |= init
    ok = np.maximum.accumulate(run * 2 + flag) == run * 2 + 1
    # side entries: the first step of a run that starts below the top (and was not carried in from above)
    side = []
    st_idx = np.nonzero(new_run & ~ok & (Aa > SIDE_EPS) & ~((T == 0) & np.isin(pair, [s_ * NK + k_ for (s_, k_) in
                                                                                      [c for c in carry if isinstance(c, tuple)]])))[0]
    for i in st_idx:
        side.append((int(T[i]), int(Sx[i]), int(K[i]), float(Aa[i])))
    lift_keep = carry.get("lift")
    carry.clear()
    if lift_keep is not None:
        carry["lift"] = lift_keep
    last = T == n - 1
    for s_, k_, f_ in zip(Sx[last], K[last], ok[last]):
        carry[(int(s_), int(k_))] = bool(f_)
    use = ok & (Aa > 0)
    T, Sx, K, Aa, L = T[use], Sx[use], K[use], Aa[use], L[use]
    push = Aa / L
    o = np.argsort(push)                                       # later writes win: the deepest push per (step, key)
    A[T[o], K[o]] = Aa[o]; LV[T[o], K[o]] = L[o]; SID[T[o], K[o]] = Sx[o]
    return A, LV, SID, side


def simulate(anim, t0=None, t1=None, chunk=8.0):
    """Run the action over the take. Returns (blows, frames) where blows = list of dicts and frames = per anim
    frame {pitch: (depth 0..1, force N)} for keys that are down or touched."""
    fps, st = anim["fps"], anim["start"]
    C, r, ids = spheres(anim)
    fid = np.array([("RL".index(h)) * 5 + f for h, f, _p in ids])
    tf = st + np.array([f["t"] for f in anim["frames"]])
    t0 = tf[0] if t0 is None else max(t0, tf[0])
    t1 = tf[-1] if t1 is None else min(t1, tf[-1])
    ip = PchipInterpolator(tf, C, axis=0)
    dt = 1.0 / SIM_HZ
    u = np.zeros(NK); v = np.zeros(NK)
    h = np.zeros(NK); hv = np.zeros(NK)
    state = np.zeros(NK, int)            # 0 on the jack, 1 in flight, 2 parked on the repetition lever (jack out)
    armed = np.ones(NK, bool)            # jack under the hammer
    damper_up = np.zeros(NK, bool)
    opened = {}                          # key -> index into blows of the note now sounding
    launch = -np.ones(NK, int)           # the sphere that launched each key's hammer
    blows = []
    frames = {}
    carry, sides = {"start": True}, []
    first = True
    t = t0
    nxt_frame = int(np.searchsorted(tf, t0 - 1e-9))
    while t < t1 - 1e-9:
        n = int(np.ceil(min(chunk, t1 - t) * SIM_HZ - 1e-6))
        if n <= 0:
            break
        ts = t + np.arange(n) * dt
        A, LV, SID, side = _contacts(ip(ts), r, carry, fid)
        sides += [(float(ts[i_]), ids[s_], int(PITCHES[k_]), d_) for i_, s_, k_, d_ in side]
        for i in range(n):
            a, lv = A[i], LV[i]
            ud = np.where(a > 0, (a - DW / (K_PAD * lv)) / lv, 0.0)     # where the finger can push the key down to
            uu = np.where(a > 0, (a - UW / (K_PAD * lv)) / lv, -1.0)    # where it still holds the key up to
            if first:                                    # the take starts mid-music: keys start where the hands hold
                u = np.clip(ud, 0.0, KEY_TRAVEL); v = np.zeros(NK)       # them, hammers parked, no blows
                damper_up = u > U_DAMP_UP; armed = u < U_RESET; state = np.where(armed, 0, 2)
                h = np.minimum(RATIO * u, BLOW - LETOFF - 0.002); hv = np.zeros(NK)
                first = False
            un = u.copy()
            dn = ud > u
            un[dn] = np.minimum(ud[dn], KEY_TRAVEL)
            up = ~dn & (uu < u)
            vf = np.minimum(v, 0.0) - A_RETURN * dt
            un[up] = np.maximum(np.maximum(uu[up], u[up] + vf[up] * dt), 0.0)
            vn = (un - u) / dt
            u, v = un, vn
            # hammer
            on = state == 0
            he, hve = RATIO * u, RATIO * v
            esc = on & (he >= BLOW - LETOFF)
            thrown = on & ~esc & (hv > 0) & (hv - G_HAMMER * dt > hve)
            h[on] = he[on]; hv[on & ~thrown] = hve[on & ~thrown]
            hv[thrown] -= G_HAMMER * dt
            armed[esc] = False
            state[esc | thrown] = 1
            launch[esc | thrown] = SID[i, esc | thrown]
            fl = state == 1
            if fl.any():
                fl_now = fl & ~(esc | thrown)
                hv[fl_now] -= G_HAMMER * dt
                h[fl] += hv[fl] * dt
                hit = fl & (h >= BLOW)
                for k in np.nonzero(hit)[0]:
                    tk = ts[i]
                    if k in opened:
                        blows[opened[k]]["t_off"] = tk
                    s = int(launch[k])
                    blows.append({"t_on": float(tk), "t_off": None, "pitch": int(PITCHES[k]), "hammer": float(hv[k]),
                                  "who": ids[s][0] + str(ids[s][1] + 1) + ("" if ids[s][2] == "pad" else "/" + ids[s][2])
                                  if s >= 0 else "?", "damped_at_blow": bool(not damper_up[k])})
                    opened[k] = len(blows) - 1
                    if not damper_up[k]:
                        blows[-1]["t_off"] = float(tk) + 0.03
                        del opened[k]
                    h[k] = BLOW; hv[k] = -REBOUND * hv[k]; armed[k] = False
                # an armed hammer (thrown, jack still under it) lands back on the jack, or the jack catches up with
                # it; an escaped one falls onto the repetition lever / back check a little under let-off
                back = fl & ((armed & (h <= RATIO * u)) |
                             (~armed & (hv <= 0) & (h <= np.minimum(RATIO * u, BLOW - LETOFF - 0.002))))
                state[back & armed] = 0
                state[back & ~armed] = 2
                h[back] = np.minimum(RATIO * u[back], BLOW - LETOFF - 0.002)
            pk = state == 2
            h[pk] = np.minimum(RATIO * u[pk], BLOW - LETOFF - 0.002); hv[pk] = 0.0
            rs = pk & (u < U_RESET)
            armed[rs] = True; state[rs] = 0
            # dampers
            du = ~damper_up & (u > U_DAMP_UP)
            dd = damper_up & (u < U_DAMP_DN)
            damper_up[du] = True; damper_up[dd] = False
            for k in np.nonzero(dd)[0]:
                if k in opened:
                    blows[opened.pop(k)]["t_off"] = float(ts[i])
            # one record per animation frame
            while nxt_frame < len(tf) and tf[nxt_frame] <= ts[i] + 1e-9:
                frc = np.where(a > 0, K_PAD * np.maximum(a - u * lv, 0.0) * lv, 0.0)
                rec = {}
                for k in np.nonzero((u > 1e-4) | (frc > 0))[0]:
                    rec[str(int(PITCHES[k]))] = [round(float(u[k] / KEY_TRAVEL), 3), round(float(frc[k]), 3)]
                frames[nxt_frame] = rec
                nxt_frame += 1
        t += n * dt
    for k, j in opened.items():
        blows[j]["t_off"] = float(t1)
    return blows, frames, sides


def velocity_curve(blows, matches, notes):
    """MIDI velocity = a + b ln(hammer m/s), least squares over the matched notes (b >= 0). Returns a, b, r."""
    hv = np.array([blows[j]["hammer"] for i, j in matches.items()])
    vw = np.array([notes[i][3] for i in matches])
    if len(hv) < 10:
        return 20.0, 30.0, float("nan")
    x = np.log(np.maximum(hv, 1e-3))
    r = float(np.corrcoef(x, vw)[0, 1])
    b, a = np.polyfit(x, vw, 1)
    if b <= 0:
        b, a = 1e-3, float(np.mean(vw))
    return float(a), float(b), r


def match(blows, notes, t0, t1, tol=0.12):
    """Written note index -> blow index (same pitch, nearest blow within tol of the onset, each blow once)."""
    by = {}
    for j, b in enumerate(blows):
        by.setdefault(b["pitch"], []).append(j)
    used, out = set(), {}
    order = sorted((i for i, n in enumerate(notes) if t0 <= n[0] < t1), key=lambda i: notes[i][0])
    for i in order:
        s, p = notes[i][0], notes[i][2]
        best, bd = None, tol
        for j in by.get(p, []):
            d = abs(blows[j]["t_on"] - s)
            if j not in used and d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best); out[i] = best
    return out


def report(blows, notes, matches, pedal, a, b, r, frames, anim, window=None):
    fps, st = anim["fps"], anim["start"]
    rows = {}
    secs = dict(SECTIONS)
    if window is not None:                              # a test window: its own row, sections clipped to it
        secs = {k: (max(s0, window[0]), min(s1, window[1])) for k, (s0, s1) in SECTIONS.items()
                if min(s1, window[1]) - max(s0, window[0]) > 1.0}
    for sec, (s0, s1) in secs.items():
        W = [i for i, n in enumerate(notes) if s0 + 0.1 <= n[0] < s1 - 0.15]
        M = [i for i in W if i in matches]
        Bw = [j for j, bl in enumerate(blows) if s0 + 0.1 <= bl["t_on"] < s1 - 0.15]
        mset = set(matches.values())
        extra = [j for j in Bw if j not in mset]
        rep = [j for j in extra if any(abs(blows[j]["t_on"] - notes[i][0]) < 0.4 and notes[i][2] == blows[j]["pitch"]
                                       for i in W)]
        wrong = [j for j in extra if j not in rep]
        dt_ms = np.array([1000 * (blows[matches[i]]["t_on"] - notes[i][0]) for i in M])
        vel_m = np.array([a + b * np.log(max(blows[matches[i]]["hammer"], 1e-3)) for i in M])
        vel_w = np.array([notes[i][3] for i in M])
        # holds: the model's key-sound length (blow -> damper) over the written length; audible cut = with the
        # pedal, the model's note stops more than 0.1 s before the song's does
        hold, cut = [], 0
        for i in M:
            n, bl = notes[i], blows[matches[i]]
            L = n[1] - n[0]
            if L < 0.08:
                continue
            hold.append(min(1.5, (bl["t_off"] - bl["t_on"]) / L))
            song_end = _pedal_end(pedal, n[1])
            model_end = _pedal_end(pedal, bl["t_off"])
            cut += model_end < song_end - 0.1
        # silent presses: a written note whose key went at least 80 % down within 0.25 s but sounded nothing
        missed = [i for i in W if i not in matches]
        silent = 0
        for i in missed:
            p, s = str(notes[i][2]), notes[i][0]
            j0, j1 = int(np.searchsorted(st + np.arange(len(anim["frames"])) / fps, s - 0.12)), int((s + 0.25 - st) * fps)
            silent += any(frames.get(j, {}).get(p, [0, 0])[0] >= 0.8 for j in range(j0, j1 + 1))
        held_force = []
        for i in M:
            n = notes[i]; p = str(n[2])
            fs = [frames.get(j, {}).get(p, [0, 0])[1] for j in range(int((n[0] - st) * fps) + 2, int((n[1] - st) * fps))]
            if fs:
                held_force.append(np.median(fs))
        dur = max(1e-6, s1 - s0)
        rows[sec] = {"written": len(W), "sounded %": 100 * len(M) / max(1, len(W)), "missed": len(missed),
                     "silent presses": silent, "wrong notes": len(wrong), "wrong /min": 60 * len(wrong) / dur,
                     "double blows": len(rep),
                     "timing |ms| p50": float(np.median(np.abs(dt_ms))) if len(dt_ms) else 0,
                     "timing |ms| p90": float(np.percentile(np.abs(dt_ms), 90)) if len(dt_ms) else 0,
                     "timing ms mean": float(dt_ms.mean()) if len(dt_ms) else 0,
                     "velocity r": float(np.corrcoef(vel_m, vel_w)[0, 1]) if len(M) > 2 else 0,
                     "key held / written p50": float(np.median(hold)) if hold else 0,
                     "held >= 90 %": 100 * float(np.mean(np.array(hold) >= 0.9)) if hold else 0,
                     "audible cut % (pedal)": 100 * cut / max(1, len(hold)),
                     "hold force N p50": float(np.median(held_force)) if held_force else 0}
    wrong_list = []
    mset = set(matches.values())
    for j, bl in enumerate(blows):
        if j in mset:
            continue
        near = [i for i, n in enumerate(notes) if abs(n[0] - bl["t_on"]) < 0.4 and n[2] == bl["pitch"]]
        if not near:
            wrong_list.append({"t": round(bl["t_on"], 3), "video_t": round(bl["t_on"] - st, 3), "key": name(bl["pitch"]),
                               "who": bl["who"], "vel": round(a + b * np.log(max(bl["hammer"], 1e-3)), 1)})
    return {"velocity_curve": {"a": a, "b": b, "r_all": r}, "sections": rows, "wrong": wrong_list}


def _pedal_end(pedal, e):
    if pedal is None:
        return e
    k = np.searchsorted(pedal, e)
    return pedal[k] + 0.04 if k < len(pedal) else e + 3.0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("suite"); ap.add_argument("notes"); ap.add_argument("out")
    ap.add_argument("--sfz"); ap.add_argument("--t0", type=float); ap.add_argument("--t1", type=float)
    ap.add_argument("--no-audio", action="store_true")
    o = ap.parse_args(argv)
    os.makedirs(o.out, exist_ok=True)
    anim = json.load(open(o.suite))
    notes = json.load(open(o.notes))
    blows, frames, sides = simulate(anim, o.t0, o.t1)
    st = anim["start"]
    t0 = o.t0 or st
    t1 = o.t1 or st + anim["frames"][-1]["t"]
    matches = match(blows, notes, t0, t1)
    a, b, r = velocity_curve(blows, matches, notes)
    for bl in blows:
        bl["vel"] = int(np.clip(round(a + b * np.log(max(bl["hammer"], 1e-3))), 1, 127))
    from . import sampler
    win_notes = [n for n in notes if t0 - 0.05 <= n[0] < t1]
    pedal = sampler.auto_pedal(win_notes)
    rep = report(blows, notes, matches, pedal, a, b, r, frames, anim, [t0, t1])
    rep["window"] = [t0, t1]
    deep = [x for x in sides if x[3] > 0.002]          # skin more than 2 mm inside a key's side when it got there
    for sec in rep["sections"]:
        s0, s1 = SECTIONS[sec]
        rep["sections"][sec]["through key sides"] = sum(s0 <= x[0] < s1 for x in deep)
    rep["sides"] = [{"t": round(x[0], 3), "video_t": round(x[0] - st, 3), "who": f"{x[1][0]}{x[1][1] + 1}/{x[1][2]}",
                     "key": name(x[2]), "mm": round(1000 * x[3], 1)} for x in deep]
    json.dump(blows, open(os.path.join(o.out, "played.json"), "w"))
    json.dump({"fps": anim["fps"], "start": st, "frames": [frames.get(j, {}) for j in range(len(anim["frames"]))]},
              open(os.path.join(o.out, "keys_phys.json"), "w"))
    json.dump(rep, open(os.path.join(o.out, "report.json"), "w"), indent=1)
    with open(os.path.join(o.out, "report.txt"), "w") as f:
        vc = rep["velocity_curve"]
        f.write(f"velocity curve: vel = {vc['a']:.1f} + {vc['b']:.1f} ln(hammer m/s), r = {vc['r_all']:.2f}\n")
        cols = list(next(iter(rep["sections"].values())))
        f.write(f"{'':26s}" + "".join(f"{s:>10s}" for s in rep["sections"]) + "\n")
        for c in cols:
            f.write(f"{c:26s}" + "".join(f"{rep['sections'][s][c]:10.1f}" for s in rep["sections"]) + "\n")
        f.write(f"\nskin through a key side > 2 mm ({len(rep['sides'])}), first 40:\n")
        for w in rep["sides"][:40]:
            f.write(f"  video {w['video_t']:8.2f} s  {w['key']:4s} by {w['who']:12s} {w['mm']} mm\n")
        f.write(f"\nwrong notes ({len(rep['wrong'])}):\n")
        for w in rep["wrong"]:
            f.write(f"  video {w['video_t']:8.2f} s  {w['key']:4s} by {w['who']:10s} vel {w['vel']}\n")
    print(open(os.path.join(o.out, "report.txt")).read())
    if o.sfz and not o.no_audio:
        import soundfile as sf
        model = [[bl["t_on"] - t0, bl["t_off"] - t0, bl["pitch"], bl["vel"], 1] for bl in blows]
        song = [[n[0] - t0, n[1] - t0, n[2], n[3], n[4]] for n in win_notes]
        ped = pedal - t0
        dur = t1 - t0 + 4.0
        outs = {}
        for tag, ns, pd in (("song", song, ped), ("model", model, ped), ("song_dry", song, "off"), ("model_dry", model, "off")):
            outs[tag] = sampler.render(ns, o.sfz, dur, pedal=pd, norm=False)
        for pair in (("song", "model"), ("song_dry", "model_dry")):
            g = 0.89 / max(1e-9, max(np.abs(outs[k]).max() for k in pair))
            for k in pair:
                sf.write(os.path.join(o.out, f"{k}.wav"), np.clip(outs[k] * g, -1, 1), sampler.SR, subtype="PCM_16")
        print("wrote", ", ".join(f"{k}.wav" for k in outs))


if __name__ == "__main__":
    main()

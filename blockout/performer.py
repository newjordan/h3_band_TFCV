"""Performer body shared by every instrument: torso, neck, head, arms (2-bone IK), and head "emotion".

Head motion is the audience's window into the player, so it is driven by the music, not by noise:
- beat grid (beats + downbeats) gives WHEN the head moves,
- energy (0..1, loudness or note density) gives HOW MUCH,
- a style schedule gives the CHARACTER. Styles crossfade (STYLE_XFADE s) when the schedule changes.

Styles (all angles radians, relative to the player's base gaze at the instrument):
  focused     eyes on the instrument, small nods on beats
  groove      nod on every beat (downbeat accented), sway over the bar
  wild        sharp headbang on beats, roll and yaw thrash
  crowd       head lifts and turns out to the audience, slow nods on downbeats
  expressive  slow sway with the phrase, leans in as energy rises, barely any beat nods

Frame of reference as in piano.py: x along the instrument, y away from the player, z up.
"""
import math

import numpy as np

STYLES = {
    #              nod   sharp  down_acc  lag    sway_roll sway_yaw  pitch  yaw_out  lean  jitter
    "focused":    (0.035, 1.0,  1.3,      0.03,  0.020,    0.015,    0.10,  0.00,    0.02, 0.00),
    "groove":     (0.110, 1.5,  1.5,      0.03,  0.060,    0.040,    0.02,  0.00,    0.03, 0.00),
    "wild":       (0.330, 3.0,  1.3,      0.02,  0.120,    0.100,   -0.05,  0.00,    0.05, 0.10),
    "crowd":      (0.050, 1.0,  2.0,      0.04,  0.030,    0.030,   -0.45,  0.55,    0.00, 0.00),
    "expressive": (0.015, 1.0,  2.0,      0.05,  0.070,    0.060,    0.05,  0.00,    0.10, 0.00),
}
KEYS = ("nod", "sharp", "down_acc", "lag", "sway_roll", "sway_yaw", "pitch", "yaw_out", "lean", "jitter")
STYLE_XFADE = 0.6
UPPER_ARM, FOREARM = 0.29, 0.26
try:                                    # the rigged hand's own forearm (elbow -> wrist), so arm and mesh agree
    import json as _json, os as _os
    _B = _json.load(open(_os.path.join(_os.path.dirname(__file__), "hand_model", "skeleton.json")))["hands"]["R"]
    FOREARM = _B["lowerarm01"]["length"] + _B["lowerarm02"]["length"]
except Exception:
    pass


def style_params(t, schedule):
    """schedule: [(t_start, style), ...] sorted. Returns the crossfaded parameter dict at time t."""
    cur = dict(zip(KEYS, STYLES[schedule[0][1]]))
    for i, (t0, name) in enumerate(schedule[1:], 1):
        if t < t0:
            break
        nxt = dict(zip(KEYS, STYLES[name]))
        u = min(1.0, (t - t0) / STYLE_XFADE)
        u = u * u * (3 - 2 * u)
        cur = {k: cur[k] + (nxt[k] - cur[k]) * u for k in KEYS}
    return cur


def _beat_phase(t, beats):
    i = np.searchsorted(beats, t, side="right") - 1
    if i < 0:
        return 0.0, -1
    if i + 1 >= len(beats):
        period = beats[-1] - beats[-2] if len(beats) > 1 else 0.5
        return ((t - beats[i]) / period) % 1.0, i
    return (t - beats[i]) / (beats[i + 1] - beats[i]), i


def head_angles(t, beats, downbeats, energy, schedule, seed=0):
    """(pitch_down, yaw, roll) offsets in radians and lean (m, forward) at time t."""
    P = style_params(t, schedule)
    e = 0.4 + 0.9 * energy
    ph, i = _beat_phase(t - P["lag"], beats)
    # nod: lands on the beat; sharper styles snap down and float back up
    shape = (0.5 * (1 + math.cos(2 * math.pi * ph))) ** P["sharp"]
    acc = 1.0
    if i >= 0 and len(downbeats) and np.min(np.abs(downbeats - beats[i])) < 1e-3:
        acc = P["down_acc"]
    nod = P["nod"] * shape * acc * e
    # bar-length sway (period = 2 bars for roll, 4 bars for yaw)
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 2.0
    j = np.searchsorted(downbeats, t, side="right") - 1
    bph = ((t - downbeats[j]) / bar if j >= 0 else t / bar)
    nb = (j if j >= 0 else 0) + bph
    roll = P["sway_roll"] * math.sin(math.pi * nb) * e
    yaw = P["sway_yaw"] * math.sin(0.5 * math.pi * nb + 0.7) * e + P["yaw_out"]
    if P["jitter"]:
        rng = np.random.default_rng(seed + max(i, 0))
        yaw += P["jitter"] * float(rng.uniform(-1, 1)) * shape
        roll += 0.5 * P["jitter"] * float(rng.uniform(-1, 1)) * shape
    pitch = P["pitch"] + nod
    lean = P["lean"] * energy
    return pitch, yaw, roll, lean


def rot(pitch_down, yaw, roll):
    """Rotation matrix: yaw about z, then pitch (down = forward tilt) about x, then roll about y."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(-pitch_down), math.sin(-pitch_down)
    cy, sy = math.cos(roll), math.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


ARM_FOLLOW = 0.85                       # how far the elbow swings to put the forearm in line behind the hand


ELBOW_UP = 0.0                          # how far an arm working inside its shoulder lifts its elbow (0..1)
TORSO_R, ARM_R, TORSO_CLEAR = 0.15, 0.04, 0.012   # torso as a capsule radius, upper-arm radius, clearance (m)


SHOULDER_R = 0.065                      # the shoulder balls of the upper-torso hitbox


TORSO_HALF_W, TORSO_HALF_D = 0.13, 0.10      # the ribcage at elbow height: narrower than the shoulders (0.19 out), so a
                                             # hanging upper arm just brushes it; shallower than wide: elbows may come in front


def _clear_of_torso(p, torso, skip_shoulder=None):
    """How far a point of the arm (with the arm's radius) is inside the upper-torso hitbox: an elliptic cylinder
    (half-width x half-depth in the torso's own frame) between its two ends, plus a sphere at each shoulder
    (skip_shoulder: the arm's own). <= 0: clear."""
    a, b = (np.asarray(v, float) for v in torso[:2])
    R = torso[3] if len(torso) > 3 else np.eye(3)
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    v = R.T @ (p - (a + t * ab))                      # torso frame: x across, y forward, z up
    ax, ay = TORSO_HALF_W + ARM_R + TORSO_CLEAR, TORSO_HALF_D + ARM_R + TORSO_CLEAR
    r = math.hypot(v[0] / ax, v[1] / ay)
    d = (1.0 - r) * min(ax, ay)
    for name, sp in (torso[2] if len(torso) > 2 else {}).items():
        if name != skip_shoulder:
            d = max(d, SHOULDER_R + ARM_R + TORSO_CLEAR - float(np.linalg.norm(p - np.asarray(sp, float))))
    return d


def torso_box(pose):
    """The upper-torso hitbox of a Body.pose() result: capsule ends and shoulder centres."""
    R = np.array(pose["torso_rot"]); hips = np.array(pose["hips"]); top = np.array(pose["chest"])
    return (hips + R @ np.array([0, 0, 0.12]), top - R @ np.array([0, 0, 0.08]),
            {k: np.array(v) for k, v in pose["shoulders"].items()}, R)


def arm_clearance(shoulder, elbow, wrist, torso, own=None):
    """Worst penetration of the upper arm (its outer half) and forearm into the upper-torso hitbox."""
    s, e, w = (np.asarray(v, float) for v in (shoulder, elbow, wrist))
    pts = [s + (e - s) * u for u in (0.75, 1.0)] + [e + (w - e) * u for u in (0.25, 0.5, 0.75)]
    return max(_clear_of_torso(p, torso, own) for p in pts)


def arm_ik(shoulder, wrist, side, hand_fwd=None, torso=None, lift=True):
    """Elbow for a 2-bone arm; the elbow points down and out to the player's side. With hand_fwd (the hand's
    wrist -> knuckles direction) the elbow swings around the shoulder-wrist axis so the forearm lines up behind
    the hand: the arm carries the hand, the wrist doesn't bend to suit the arm."""
    d = wrist - shoulder
    L = float(np.linalg.norm(d))
    L = min(max(L, 1e-4), UPPER_ARM + FOREARM - 1e-3)
    u = d / max(np.linalg.norm(d), 1e-6)
    a = (UPPER_ARM ** 2 - FOREARM ** 2 + L ** 2) / (2 * L)
    h = math.sqrt(max(UPPER_ARM ** 2 - a * a, 0.0))
    pole = np.array([side * 0.35, -0.3, -1.0])          # elbows hang close to the body, slightly out
    pole -= (pole @ u) * u
    pole /= max(np.linalg.norm(pole), 1e-6)
    if hand_fwd is not None:
        hf = np.asarray(hand_fwd, float); hf = hf / max(np.linalg.norm(hf), 1e-9)
        want = wrist - FOREARM * hf - (shoulder + a * u)   # where the elbow would put the forearm in line
        want -= (want @ u) * u
        if np.linalg.norm(want) > 1e-6:
            p2 = ARM_FOLLOW * want / np.linalg.norm(want) + (1 - ARM_FOLLOW) * pole
            if p2[2] > -0.1:                             # the elbow never rises above the shoulder-wrist line
                p2[2] = -0.1
                p2 -= (p2 @ u) * u
            pole = p2 / max(np.linalg.norm(p2), 1e-6)
    c = shoulder + a * u
    el = c + h * pole
    own = "R" if side > 0 else "L"
    hf = None if hand_fwd is None else np.asarray(hand_fwd, float) / max(np.linalg.norm(hand_fwd), 1e-9)
    if torso is not None and (hf is not None or arm_clearance(shoulder, el, wrist, torso, own) > 0):
        # the elbow swings around the shoulder-wrist axis to where the forearm lines up best behind the hand,
        # among the spots where the whole arm clears the upper-torso / shoulder hitbox (it rises only if it must)
        e1 = pole
        e2 = np.cross(u, e1)
        best = None
        for ang in np.linspace(-math.pi, math.pi, 145):
            e = c + h * (math.cos(ang) * e1 + math.sin(ang) * e2)
            if e[2] > shoulder[2] + 0.05:                  # never above the shoulder
                continue
            pen = max(0.0, arm_clearance(shoulder, e, wrist, torso, own))
            if hf is not None:
                fa = (wrist - e) / max(np.linalg.norm(wrist - e), 1e-9)
                bend = math.acos(float(np.clip(fa @ hf, -1, 1)))
            else:
                bend = abs(ang)
            key = 1000.0 * pen + bend
            if best is None or key < best[0]:
                best = (key, e)
        if best is not None:
            el = best[1]
    return el


class Body:
    """Seated or standing body. hips is the pivot the torso leans and sways around."""

    def __init__(self, hips, shoulder_half=0.19, torso_h=0.55, base_gaze_pitch=0.8, eye_ahead=0.09):
        self.hips = np.asarray(hips, float)
        self.sh, self.th = shoulder_half, torso_h
        self.gaze = base_gaze_pitch      # where the eyes look at rest (radians below horizontal)
        self.eye_ahead = eye_ahead

    def pose(self, pitch, yaw, roll, lean, wrists, hand_fwd=None, torso=(0.0, 0.0, 0.0)):
        # torso follows the head a little; the head carries the rest. torso = (pitch, yaw, roll) the reach adds:
        # it leans toward a far hand, then turns to open toward that end of the keyboard
        Rt = rot(0.25 * pitch + 0.06 + 0.6 * lean + torso[0], 0.3 * yaw + torso[1], 0.35 * roll + torso[2])
        top = self.hips + Rt @ np.array([0, 0, self.th])
        shoulders = {"L": self.hips + Rt @ np.array([-self.sh, 0, self.th - 0.03]),
                     "R": self.hips + Rt @ np.array([self.sh, 0, self.th - 0.03])}
        neck_top = top + Rt @ np.array([0, 0.01, 0.09])
        head_pitch = 0.55 * self.gaze + pitch          # head tilts part of the way; the eyes do the rest
        Rh = rot(head_pitch, yaw, roll)
        head = neck_top + Rh @ np.array([0, 0.02, 0.11])
        eye = head + Rh @ np.array([0, self.eye_ahead, 0.02])
        Rg = rot(self.gaze + 0.6 * pitch, 0.8 * yaw, 0.6 * roll)   # first-person view: gaze, with head motion
        out = {"hips": self.hips.tolist(), "chest": top.tolist(), "neck": neck_top.tolist(), "head": head.tolist(),
               "head_rot": Rh.tolist(), "torso_rot": Rt.tolist(), "eye": eye.tolist(),
               "cam_fwd": (Rg @ np.array([0, 1.0, 0])).tolist(), "cam_up": (Rg @ np.array([0, 0, 1.0])).tolist(),
               "shoulders": {k: v.tolist() for k, v in shoulders.items()}, "elbows": {}}
        for side, w in wrists.items():
            out["elbows"][side] = arm_ik(shoulders[side], np.asarray(w), 1 if side == "R" else -1,
                                         None if hand_fwd is None else hand_fwd.get(side),
                                         torso=(self.hips + Rt @ np.array([0, 0, 0.12]), top - Rt @ np.array([0, 0, 0.08]),
                                                {k: v for k, v in shoulders.items()}, Rt)).tolist()
        return out


def energy_from_onsets(onsets, times, win=1.5):
    """Note density as energy, 0..1 (for MIDI). Gaussian-smoothed onsets per second, normalised to the 90th pct."""
    on = np.asarray(onsets)
    e = np.array([np.exp(-0.5 * ((on - t) / (win / 2)) ** 2).sum() for t in times])
    return np.clip(e / max(np.percentile(e, 90), 1e-6), 0, 1)

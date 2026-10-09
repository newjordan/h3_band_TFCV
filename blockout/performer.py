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


def arm_ik(shoulder, wrist, side):
    """Elbow for a 2-bone arm; the elbow points down and out to the player's side."""
    d = wrist - shoulder
    L = float(np.linalg.norm(d))
    L = min(max(L, 1e-4), UPPER_ARM + FOREARM - 1e-3)
    u = d / max(np.linalg.norm(d), 1e-6)
    a = (UPPER_ARM ** 2 - FOREARM ** 2 + L ** 2) / (2 * L)
    h = math.sqrt(max(UPPER_ARM ** 2 - a * a, 0.0))
    pole = np.array([side * 0.35, -0.3, -1.0])          # elbows hang close to the body, slightly out
    pole -= (pole @ u) * u
    pole /= max(np.linalg.norm(pole), 1e-6)
    return shoulder + a * u + h * pole


class Body:
    """Seated or standing body. hips is the pivot the torso leans and sways around."""

    def __init__(self, hips, shoulder_half=0.19, torso_h=0.55, base_gaze_pitch=0.8, eye_ahead=0.09):
        self.hips = np.asarray(hips, float)
        self.sh, self.th = shoulder_half, torso_h
        self.gaze = base_gaze_pitch      # where the eyes look at rest (radians below horizontal)
        self.eye_ahead = eye_ahead

    def pose(self, pitch, yaw, roll, lean, wrists):
        # torso follows the head a little; the head carries the rest
        Rt = rot(0.25 * pitch + 0.06 + 0.6 * lean, 0.3 * yaw, 0.35 * roll)
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
            out["elbows"][side] = arm_ik(shoulders[side], np.asarray(w), 1 if side == "R" else -1).tolist()
        return out


def energy_from_onsets(onsets, times, win=1.5):
    """Note density as energy, 0..1 (for MIDI). Gaussian-smoothed onsets per second, normalised to the 90th pct."""
    on = np.asarray(onsets)
    e = np.array([np.exp(-0.5 * ((on - t) / (win / 2)) ** 2).sum() for t in times])
    return np.clip(e / max(np.percentile(e, 90), 1e-6), 0, 1)

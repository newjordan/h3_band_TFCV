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


def arm_pair_clearance(shoulder, elbow, wrist, other_arm):
    """Worst capsule penetration between two arms, in metres (zero when clear)."""
    def distance(a, b, c, d):
        u, v, w = b - a, d - c, a - c
        aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
        if aa < 1e-16:
            s, t = 0.0, float(np.clip(ee / max(cc, 1e-16), 0, 1))
        elif cc < 1e-16:
            s, t = float(np.clip(-dd / aa, 0, 1)), 0.0
        else:
            den = aa * cc - bb * bb
            s = float(np.clip((bb * ee - cc * dd) / den, 0, 1)) if den > 1e-16 else 0.0
            t = (bb * s + ee) / cc
            if t < 0:
                t, s = 0.0, float(np.clip(-dd / aa, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((bb - dd) / aa, 0, 1))
        return float(np.linalg.norm(w + s * u - t * v))
    s, e, w = (np.asarray(v, float) for v in (shoulder, elbow, wrist))
    os, oe, ow = (np.asarray(v, float) for v in other_arm)
    return max(0.0, *(r + ro + 0.004 - distance(a, b, c, d)
                      for a, b, r in ((s, e, ARM_R), (e, w, 0.030))
                      for c, d, ro in ((os, oe, ARM_R), (oe, ow, 0.030))))


def arm_ik(shoulder, wrist, side, hand_fwd=None, torso=None, lift=True,
           previous=None, other_arm=None, dt=None, max_angular_speed=4.0):
    """Elbow for a 2-bone arm; the elbow points down and out to the player's side. With hand_fwd (the hand's
    wrist -> knuckles direction) the elbow swings around the shoulder-wrist axis so the forearm lines up behind
    the hand: the arm carries the hand, the wrist doesn't bend to suit the arm.

    A previous physical elbow selects a continuous bend plane. Its projection
    onto the NEW shoulder/wrist circle preserves both bone lengths; averaging
    elbow coordinates would shorten the bones. dt bounds motion around that
    circle (rad/s), not motion forced by moving wrist/shoulder endpoints.
    Torso/other-arm clearance is prioritized inside this rate limit. Conflicting
    endpoint/contact constraints can remain infeasible and need wrist replanning.
    """
    shoulder, wrist = np.asarray(shoulder, float), np.asarray(wrist, float)
    if not np.all(np.isfinite(np.r_[shoulder, wrist])):
        raise ValueError("arm endpoints must be finite")
    d = wrist - shoulder
    actual_length = float(np.linalg.norm(d))
    # Degenerate/unreachable endpoints have no exact two-bone solution. Keep
    # the upper arm rigid and point toward the target; the reach error remains
    # visible to callers instead of manufacturing a huge/NaN elbow circle.
    u = d / actual_length if actual_length > 1e-9 else np.array([0.0, 1.0, 0.0])
    if actual_length >= UPPER_ARM + FOREARM:
        return shoulder + UPPER_ARM * u
    if actual_length <= abs(UPPER_ARM - FOREARM):
        return shoulder + math.copysign(UPPER_ARM, UPPER_ARM - FOREARM) * u
    L = actual_length
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
    if previous is not None or other_arm is not None or (torso is not None and (hf is not None or arm_clearance(shoulder, el, wrist, torso, own) > 0)):
        # the elbow swings around the shoulder-wrist axis to where the forearm lines up best behind the hand,
        # among the spots where the whole arm clears the upper-torso / shoulder hitbox (it rises only if it must)
        e1 = pole
        if previous is not None:
            previous = np.asarray(previous, float)
            if previous.shape != (3,) or not np.all(np.isfinite(previous)):
                raise ValueError("previous elbow must be a finite 3-vector")
            radial = previous - c
            radial -= (radial @ u) * u
            if np.linalg.norm(radial) > 1e-8:
                e1 = radial / np.linalg.norm(radial)
        e2 = np.cross(u, e1)
        span = math.pi
        if previous is not None and dt is not None:
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(max_angular_speed) or max_angular_speed <= 0:
                raise ValueError("arm dt and max_angular_speed must be finite and positive")
            span = min(span, dt * max_angular_speed)
        # Dense local samples avoid the old 2.5-degree quantization during slow
        # holds. The entire range remains available for the first frame.
        angles = np.linspace(-span, span, 49 if span < math.pi else 145)
        best = None
        for ang in angles:
            e = c + h * (math.cos(ang) * e1 + math.sin(ang) * e2)
            pen = max(0.0, e[2] - shoulder[2] - 0.05)
            if torso is not None:
                pen = max(pen, arm_clearance(shoulder, e, wrist, torso, own))
            other_pen = 0.0 if other_arm is None else arm_pair_clearance(shoulder, e, wrist, other_arm)
            if hf is not None:
                fa = (wrist - e) / max(np.linalg.norm(wrist - e), 1e-9)
                bend = math.acos(float(np.clip(fa @ hf, -1, 1)))
            else:
                bend = abs(ang)
            continuity = 0.35 * ang * ang if previous is not None else 0.0
            # Never resolve crossed arms by burying an elbow in the torso.
            # If the rate-limited circle contains no torso-clear solution, make
            # that remaining infeasibility explicit through the returned geometry.
            key = (max(0.0, pen - 0.0005), 1000.0 * other_pen + bend + continuity)
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

    def pose(self, pitch, yaw, roll, lean, wrists, hand_fwd=None, torso=(0.0, 0.0, 0.0),
             previous_elbows=None, dt=None, max_angular_speed=4.0):
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
        hitbox = (self.hips + Rt @ np.array([0, 0, 0.12]), top - Rt @ np.array([0, 0, 0.08]), shoulders, Rt)
        for side, w in wrists.items():
            out["elbows"][side] = arm_ik(shoulders[side], np.asarray(w), 1 if side == "R" else -1,
                                         None if hand_fwd is None else hand_fwd.get(side),
                                         torso=hitbox, previous=(previous_elbows or {}).get(side),
                                         dt=dt, max_angular_speed=max_angular_speed).tolist()
        # Coupled clearance: the forearms can meet even while the hands' skin
        # blobs are far apart. Repeated passes use the same physical prior, so
        # they cannot spend the speed budget twice within one frame.
        if len(wrists) == 2 and arm_pair_clearance(shoulders["L"], out["elbows"]["L"], wrists["L"],
                                                  (shoulders["R"], out["elbows"]["R"], wrists["R"])) > 0:
            for _ in range(2):
                for side, w in wrists.items():
                    other = "L" if side == "R" else "R"
                    out["elbows"][side] = arm_ik(shoulders[side], np.asarray(w), 1 if side == "R" else -1,
                                                 None if hand_fwd is None else hand_fwd.get(side), torso=hitbox,
                                                 previous=(previous_elbows or {}).get(side),
                                                 other_arm=(shoulders[other], out["elbows"][other], wrists[other]),
                                                 dt=dt, max_angular_speed=max_angular_speed).tolist()
        return out


def _segment_distances(a, b, c, d):
    """Vectorized finite-segment distances, including parallel/point segments."""
    u, v, w = b - a, d - c, a - c
    dot = lambda x, y: np.sum(x * y, axis=-1)
    aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    den = aa * cc - bb * bb
    safe = den > 1e-16
    s = np.divide(bb * ee - cc * dd, den, out=np.zeros_like(den), where=safe)
    t = np.divide(aa * ee - bb * dd, den, out=np.zeros_like(den), where=safe)
    aa, cc = np.maximum(aa, 1e-16), np.maximum(cc, 1e-16)
    ss = np.stack([s, np.zeros_like(s), np.ones_like(s), np.clip(-dd / aa, 0, 1),
                   np.clip((bb - dd) / aa, 0, 1)], -1)
    tt = np.stack([t, np.clip(ee / cc, 0, 1), np.clip((ee + bb) / cc, 0, 1),
                   np.zeros_like(t), np.ones_like(t)], -1)
    delta = w[..., None, :] + ss[..., None] * u[..., None, :] - tt[..., None] * v[..., None, :]
    dist = np.sum(delta * delta, -1)
    dist[..., 0] = np.where(safe & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), dist[..., 0], np.inf)
    return np.sqrt(np.maximum(np.min(dist, axis=-1), 0.0))


def _track_obstacles(frame, side, finger_radii):
    other = 'L' if side == 'R' else 'R'
    if other not in frame['hands']:
        return []
    hand = frame['hands'][other]
    shoulder = frame['body']['shoulders'][other]
    obstacles = [(shoulder, hand['elbow'], ARM_R), (hand['elbow'], hand['wrist'], .030)]
    for f, points in enumerate(hand.get('fingers', [])):
        for k in range(len(points) - 1):
            obstacles.append((points[k], points[k + 1], float(finger_radii[other][f][k])))
        if f > 0:
            obstacles.append((hand['wrist'], points[0], .015))
    return obstacles


def _track_overlaps(shoulder, elbows, wrist, obstacles):
    if not obstacles:
        return np.zeros((len(elbows), 0))
    first = np.stack([np.broadcast_to(shoulder, elbows.shape), elbows], 1)
    last = np.stack([elbows, np.broadcast_to(wrist, elbows.shape)], 1)
    oa = np.asarray([v[0] for v in obstacles]); ob = np.asarray([v[1] for v in obstacles])
    radius = np.asarray([v[2] for v in obstacles])
    distances = _segment_distances(first[:, :, None, :], last[:, :, None, :],
                                   oa[None, None, :, :], ob[None, None, :, :])
    penetration = np.maximum(0., np.array([ARM_R, .030])[None, :, None] + radius[None, None, :] + .004 - distances)
    return penetration.reshape(len(elbows), -1)


def relax_arm_track(frames, fps, max_angular_speed=4.0, passes=2,
                    previous_elbows=None, previous_t=None):
    """Anticipate arm crossings over a whole saved frame sequence using DP.

    Only elbow coordinates change. Each frame has145 exact two-bone solutions:
    144 samples around its elbow circle plus the incoming elbow projected onto
    that circle. Torso clearance and projected bend-plane angular speed are hard
    constraints; collision with the opposite arm, fingers and palm is a cost.
    Alternate arm tracks so each sees the latest opposite-arm solution. If no
    complete safe, rate-bounded path exists, retain that side's incoming track
    and report the failure. Remaining collisions are reported, never concealed.
    """
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_angular_speed) or max_angular_speed <= 0:
        raise ValueError('arm track fps and speed must be finite and positive')
    if not frames:
        return {'frames': 0, 'passes': [], 'max_overlap_mm': 0.0}
    from .rig.skeleton import Skeleton
    sides = [s for s in ('R', 'L') if all(s in f['hands'] for f in frames)]
    unreachable = {}
    for j, frame in enumerate(frames):
        for side in sides:
            distance = float(np.linalg.norm(np.asarray(frame['hands'][side]['wrist'])
                                             - frame['body']['shoulders'][side]))
            if not abs(UPPER_ARM - FOREARM) < distance < UPPER_ARM + FOREARM:
                unreachable.setdefault(j, []).append({'side': side, 'reach_error_mm': 1000. * max(
                    0., distance - UPPER_ARM - FOREARM, abs(UPPER_ARM - FOREARM) - distance)})
    if unreachable:
        # A malformed warmup frame must not disable a reachable scored passage.
        # Keep it verbatim, disclose it, and seed the next valid run from its
        # physical incoming elbow so the first valid transition is still bounded.
        report = {'frames': len(frames), 'passes': [], 'unreachable_frames': [
            {'frame': j, 'time': frames[j].get('t', j / fps), 'details': detail}
            for j, detail in unreachable.items()], 'reachable_run_metrics_only': True,
            'max_overlap_mm': 0., 'max_torso_penetration_mm': 0., 'overlap_frames': 0}
        j = 0
        while j < len(frames):
            if j in unreachable:
                j += 1
                continue
            first = j
            while j < len(frames) and j not in unreachable:
                j += 1
            before = frames[first - 1] if first else None
            result = relax_arm_track(frames[first:j], fps, max_angular_speed, passes,
                                      previous_elbows=({s: before['hands'][s]['elbow'] for s in sides}
                                                        if before else previous_elbows),
                                      previous_t=(before.get('t', (first - 1) / fps) if before else previous_t))
            report['passes'].extend({**p, 'run_start': first, 'run_end': j} for p in result['passes'])
            for key in ('max_overlap_mm', 'max_torso_penetration_mm'):
                report[key] = max(report[key], result[key])
            report['overlap_frames'] += result['overlap_frames']
        return report
    obstacle_sides = [s for s in ('R', 'L') if any(s in f['hands'] for f in frames)]
    skeletons = {s: Skeleton(s) for s in obstacle_sides}
    radii = {s: [[max(float(v or .008) for v in skeletons[s].radii[f'finger{f}-{k}'])
                   for k in (1, 2, 3)] for f in range(1, 6)] for s in obstacle_sides}
    audit = {'frames': len(frames), 'max_angular_speed': float(max_angular_speed),
             'torso_tolerance_mm': .5, 'passes': []}
    circles = {}
    for side in sides:
        candidates, centres, axes, unary, safe = [], [], [], [], []
        invalid = None
        for j, frame in enumerate(frames):
            hand = frame['hands'][side]
            shoulder = np.asarray(frame['body']['shoulders'][side], float)
            wrist = np.asarray(hand['wrist'], float)
            d = wrist - shoulder; length = float(np.linalg.norm(d))
            if not abs(UPPER_ARM - FOREARM) < length < UPPER_ARM + FOREARM:
                invalid = {'reason': 'unreachable endpoints', 'frame': j}
                break
            axis = d / length
            along = (UPPER_ARM ** 2 - FOREARM ** 2 + length ** 2) / (2 * length)
            centre = shoulder + along * axis
            height = math.sqrt(max(UPPER_ARM ** 2 - along ** 2, 0.))
            radial = np.asarray(hand['elbow']) - centre
            radial -= axis * (radial @ axis)
            if np.linalg.norm(radial) < 1e-9:
                radial = np.cross(axis, np.eye(3)[int(np.argmin(abs(axis)))])
            radial /= np.linalg.norm(radial)
            normal = np.cross(axis, radial)
            # Offset the regular samples half a bin so the exact incoming pose
            # is a distinct candidate, even during very slow passages.
            angle = np.r_[0., (np.arange(144) + .5) * (2 * math.pi / 144)]
            elbows = centre + height * (np.cos(angle)[:, None] * radial + np.sin(angle)[:, None] * normal)
            torso = torso_box(frame['body'])
            clearance = np.array([max(0., arm_clearance(shoulder, e, wrist, torso, side),
                                      e[2] - shoulder[2] - .05) for e in elbows])
            allowed = clearance <= .0005 + 1e-10
            if not np.any(allowed):
                invalid = {'reason': 'no torso-clear elbow candidate', 'frame': j}
                break
            sk = skeletons[side]
            forward = (np.asarray(hand['root']['rot']) @ sk.R0['wrist'].T @ sk.rest_frame[1]
                       if 'root' in hand else wrist - np.asarray(hand['elbow']))
            forward /= max(np.linalg.norm(forward), 1e-9)
            forearm = (wrist - elbows) / FOREARM
            bend = np.arccos(np.clip(forearm @ forward, -1, 1))
            prior = np.arccos(np.clip(np.cos(angle), -1, 1))
            candidates.append(elbows); centres.append(centre); axes.append(axis)
            unary.append(.01 * bend ** 2 + .001 * prior ** 2); safe.append(allowed)
        if invalid:
            circles[side] = invalid
        else:
            circles[side] = tuple(np.asarray(v) for v in (candidates, centres, axes, unary, safe))
    for iteration in range(int(passes)):
        for side in sides:
            data = circles[side]
            entry = {'pass': iteration + 1, 'side': side, 'applied': False}
            if isinstance(data, dict):
                audit['passes'].append({**entry, **data})
                continue
            candidates, centres, axes, base_cost, safe = data
            costs = base_cost.copy()
            for j, frame in enumerate(frames):
                overlap = _track_overlaps(np.asarray(frame['body']['shoulders'][side]), candidates[j],
                                          np.asarray(frame['hands'][side]['wrist']),
                                          _track_obstacles(frame, side, radii))
                costs[j] += 1e6 * np.sum(overlap ** 2, axis=1)
            costs[~safe] = np.inf
            if previous_elbows is not None and side in previous_elbows:
                radial = np.asarray(previous_elbows[side]) - centres[0]
                radial -= (radial @ axes[0]) * axes[0]
                norm = np.linalg.norm(radial)
                current = candidates[0] - centres[0]
                current /= np.linalg.norm(current, axis=1)[:, None]
                angle = np.arccos(np.clip(current @ (radial / max(norm, 1e-12)), -1, 1))
                dt = 1. / fps if previous_t is None else float(frames[0].get('t', 0.) - previous_t)
                costs[0, (angle > max_angular_speed * dt + 1e-9) | (norm < 1e-9) | (dt <= 0)] = np.inf
                if not np.any(np.isfinite(costs[0])):
                    audit['passes'].append({**entry, 'reason': 'no safe speed-bounded entry to reachable run', 'frame': 0})
                    continue
            accumulated = costs[0].copy(); parents = []
            failed = None
            for j in range(1, len(frames)):
                # Transport each previous elbow to the current circle before
                # measuring bend-plane change; endpoint motion is independent.
                radial = candidates[j - 1] - centres[j]
                radial -= (radial @ axes[j])[:, None] * axes[j]
                norm = np.linalg.norm(radial, axis=1)
                radial /= np.maximum(norm[:, None], 1e-12)
                current = candidates[j] - centres[j]
                current /= np.linalg.norm(current, axis=1)[:, None]
                angles = np.arccos(np.clip(radial @ current.T, -1, 1))
                dt = float(frames[j].get('t', j / fps) - frames[j - 1].get('t', (j - 1) / fps))
                if dt <= 0:
                    failed = {'reason': 'non-increasing frame time', 'frame': j}
                    break
                transition = accumulated[:, None] + .05 * angles ** 2
                transition[(angles > max_angular_speed * dt + 1e-9) | (norm[:, None] < 1e-9)] = np.inf
                best = transition.argmin(axis=0)
                accumulated = transition[best, np.arange(len(best))] + costs[j]
                parents.append(best)
                if not np.any(np.isfinite(accumulated)):
                    failed = {'reason': 'no torso-safe path within angular speed bound', 'frame': j}
                    break
            if failed:
                audit['passes'].append({**entry, **failed})
                continue
            path = [int(accumulated.argmin())]
            for parent in reversed(parents):
                path.append(int(parent[path[-1]]))
            path.reverse()
            for frame, elbows, k in zip(frames, candidates, path):
                value = elbows[k].tolist()
                frame['hands'][side]['elbow'] = value
                frame['body']['elbows'][side] = value
            audit['passes'].append({**entry, 'applied': True, 'path_cost': float(np.min(accumulated))})
    worst, torso_worst, overlap_frames = 0., 0., 0
    for frame in frames:
        overlap_frame = False
        for side in sides:
            shoulder = np.asarray(frame['body']['shoulders'][side]); hand = frame['hands'][side]
            elbow = np.asarray(hand['elbow']); wrist = np.asarray(hand['wrist'])
            ov = _track_overlaps(shoulder, elbow[None], wrist, _track_obstacles(frame, side, radii))
            peak = float(np.max(ov)) if ov.size else 0.
            worst = max(worst, peak); overlap_frame |= peak > .003
            torso_worst = max(torso_worst, arm_clearance(shoulder, elbow, wrist, torso_box(frame['body']), side))
        overlap_frames += int(overlap_frame)
    audit.update(max_overlap_mm=worst * 1000., overlap_frames=overlap_frames,
                 max_torso_penetration_mm=torso_worst * 1000.)
    return audit


def energy_from_onsets(onsets, times, win=1.5):
    """Note density as energy, 0..1 (for MIDI). Gaussian-smoothed onsets per second, normalised to the 90th pct."""
    on = np.asarray(onsets)
    e = np.array([np.exp(-0.5 * ((on - t) / (win / 2)) ** 2).sum() for t in times])
    return np.clip(e / max(np.percentile(e, 90), 1e-6), 0, 1)

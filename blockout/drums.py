"""Bounded, event-driven drummer blockout (metres, x right / y forward / z up).

This is a kinematic contact controller, not a rigid-body simulation or an MPFB
hand retargeter. Drum planes, rigid sticks, arm lengths, a retained finger grip,
and contact/rebound timing are explicit and independently measurable. Impossible
reach is projected to the arm's workspace and reported as a miss, never hidden
by stretching a bone. Run ``python -m blockout.drums --demo --out drums.json``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math

import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length < 1e-10:
        raise ValueError("A direction cannot be zero")
    return v / length


@dataclass(frozen=True)
class Drum:
    name: str
    center: tuple[float, float, float]
    radius: float
    normal: tuple[float, float, float] = (0., 0., 1.)

    def __post_init__(self):
        if np.shape(self.center) != (3,) or not np.all(np.isfinite(self.center)) or not math.isfinite(self.radius) or self.radius <= .025:
            raise ValueError("Invalid drum center/radius")
        if np.shape(self.normal) != (3,) or not np.all(np.isfinite(self.normal)):
            raise ValueError("Invalid drum normal")
        object.__setattr__(self, "normal", tuple(unit(self.normal)))
        if self.normal[2] <= .2:
            raise ValueError("This controller needs an upward-facing playing surface")


@dataclass(frozen=True)
class Rig:
    upper_arm: float = .30
    forearm: float = .27
    elbow_min_deg: float = 15.
    elbow_max_deg: float = 145.
    stick_length: float = .406
    grip_to_tip: float = .29
    stick_radius: float = .006
    tip_radius: float = .009
    finger_radius: float = .0055
    wrist_to_grip: float = .055
    strike_pitch: float = .24
    min_hand_interval: float = .075
    restitution: float = .55
    max_wrist_speed: float = 3.
    max_tip_speed: float = 9.

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("All rig parameters must be finite")
        if not (0 < self.elbow_min_deg < self.elbow_max_deg < 180):
            raise ValueError("Elbow bounds must be strictly inside (0, 180)")
        if not (0 < self.grip_to_tip < self.stick_length):
            raise ValueError("The grip must lie inside the rigid stick")
        if not (0 < self.restitution <= 1):
            raise ValueError("Restitution must be in (0, 1]")
        for name in ("upper_arm", "forearm", "stick_radius", "tip_radius",
                     "finger_radius", "wrist_to_grip", "min_hand_interval", "max_wrist_speed", "max_tip_speed"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tip_radius < self.stick_radius:
            raise ValueError("This tip-contact model needs a tip at least as wide as the shaft")
        if not 0 < self.strike_pitch < math.pi/2:
            raise ValueError("Strike pitch must be in (0, pi/2)")


DEFAULT_KIT = (
    Drum("snare", (0., .18, .10), .16),
    Drum("hihat", (-.32, .22, .23), .18),
    Drum("tom", (.12, .43, .22), .15, (0., -.12, 1.)),
    Drum("ride", (.40, .32, .26), .23, (-.10, -.05, 1.)),
)
SHOULDERS = {"L": np.array([-.18, -.20, .56]), "R": np.array([.18, -.20, .56])}


def _smooth(u):
    u = np.clip(u, 0., 1.)
    return u ** 3 * (10. + u * (-15. + 6. * u))


def _hermite(a, b, da, db, u, duration):
    """Endpoint values and physical derivatives; duration is in seconds."""
    return ((2*u**3-3*u**2+1)*a + (u**3-2*u**2+u)*duration*da
            + (-2*u**3+3*u**2)*b + (u**3-u**2)*duration*db)


def solve_arm(wrist, side, rig):
    shoulder = SHOULDERS[side]
    delta = np.asarray(wrist) - shoulder
    requested = float(np.linalg.norm(delta))
    direction = unit(delta) if requested > 1e-9 else np.array([0., 1., 0.])
    a, b = rig.upper_arm, rig.forearm
    distance = lambda degrees: math.sqrt(a*a+b*b+2*a*b*math.cos(math.radians(degrees)))
    reach = np.clip(requested, distance(rig.elbow_max_deg), distance(rig.elbow_min_deg))
    wrist = shoulder + direction * reach
    along = (a*a-b*b+reach*reach)/(2*reach)
    height = math.sqrt(max(0., a*a-along*along))
    pole = np.array([-1. if side == "L" else 1., -.25, -.55])
    bend = pole - np.dot(pole, direction)*direction
    if np.linalg.norm(bend) < 1e-6:
        pole = np.eye(3)[np.argmin(np.abs(direction))]
        bend = pole - np.dot(pole, direction)*direction
    elbow = shoulder + along*direction + height*unit(bend)
    flex = math.degrees(math.acos(np.clip((reach*reach-a*a-b*b)/(2*a*b), -1, 1)))
    return shoulder, elbow, wrist, flex, abs(requested-reach)


def segment_distance(a, b, c, d):
    """Distance between finite segments, including parallel/degenerate cases."""
    a, b, c, d = map(np.asarray, (a, b, c, d))
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    dd, ee = np.dot(u, w), np.dot(v, w)
    candidates = []
    for s in (0., 1.):
        t = np.clip((ee+bb*s)/cc, 0, 1) if cc > 1e-15 else 0.
        candidates.append(np.linalg.norm(w+s*u-t*v))
    for t in (0., 1.):
        s = np.clip((bb*t-dd)/aa, 0, 1) if aa > 1e-15 else 0.
        candidates.append(np.linalg.norm(w+s*u-t*v))
    determinant = aa*cc-bb*bb
    if determinant > 1e-15:
        s, t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w+s*u-t*v))
    return float(min(candidates))


class Drummer:
    def __init__(self, events, kit=DEFAULT_KIT, rig=Rig()):
        self.kit = {d.name: d for d in kit}
        if not self.kit or len(self.kit) != len(kit):
            raise ValueError("The kit needs unique instrument names")
        self.rig = rig
        self.events = []
        self.by_hand = {"L": [], "R": []}
        previous = {"L": -math.inf, "R": -math.inf}
        for raw in sorted(events, key=lambda e: e["time"]):
            event = dict(raw)
            time, velocity = float(event["time"]), float(event.get("velocity", .7))
            if not math.isfinite(time) or time < 0 or not math.isfinite(velocity) or not 0 <= velocity <= 1:
                raise ValueError("Events need finite nonnegative times and velocity in [0, 1]")
            if event["drum"] not in self.kit:
                raise ValueError(f"Unknown drum {event['drum']!r}; feet/pedals need their own controller")
            hand = event.get("hand", "auto")
            if hand == "auto":
                candidates = [h for h in ("L", "R") if time-previous[h] >= rig.min_hand_interval-1e-9]
                if not candidates:
                    raise ValueError(f"No hand available for hit at {time:.6f}s")
                preferred = "L" if event["drum"] == "hihat" else "R"
                hand = min(candidates, key=lambda h: (previous[h], h != preferred))
            if hand not in self.by_hand:
                raise ValueError(f"Unknown hand {hand!r}")
            if time-previous[hand] < rig.min_hand_interval-1e-9:
                raise ValueError(f"Hand {hand} cannot make both hits near {time:.6f}s")
            event.update(time=time, velocity=velocity, hand=hand)
            previous[hand] = time
            self.events.append(event)
            self.by_hand[hand].append(event)

    def _anchor(self, event, side):
        drum = self.kit[event["drum"]]
        normal = np.array(drum.normal)
        # Separate contact zones permit two hands on one drum without coincident sticks.
        across = unit(np.cross(np.array([0., 1., 0.]), normal))
        contact = np.array(drum.center) + across * (-.035 if side == "L" else .035)
        approach = contact - SHOULDERS[side]
        tangent = unit(approach-normal*np.dot(approach, normal))
        direction = tangent*math.cos(self.rig.strike_pitch)-normal*math.sin(self.rig.strike_pitch)
        tip = contact + normal*self.rig.tip_radius
        return tip-direction*self.rig.grip_to_tip, normal, tangent

    def _lift(self, t, before, after):
        """Bounded Hermite rebound/approach with a quiet plateau between strokes."""
        rig = self.rig
        v0, v1 = before["velocity"], after["velocity"]
        interval = after["time"]-before["time"]
        rise, fall = min(.13, interval*.40), min(.20, interval*.60)
        # Fast rolls use a smaller arc; hard hits use a larger preparatory stroke.
        amplitude = (.18+.52*max(v0, v1))*min(1., interval/.20)
        age, remaining = t-before["time"], after["time"]-t
        omega0 = min((2.+5.*v0)*rig.restitution, 3.*amplitude/rise)
        omega1 = min(2.+5.*v1, 3.*amplitude/fall)
        if age < rise:
            return _hermite(0., amplitude, omega0, 0., age/rise, rise)
        if remaining < fall:
            return _hermite(amplitude, 0., 0., -omega1, 1.-remaining/fall, fall)
        return amplitude

    def pose(self, side, t):
        if side not in self.by_hand or not math.isfinite(t):
            raise ValueError("Pose needs a hand L/R and finite time")
        events = self.by_hand[side]
        if not events:
            rest = {"time": t, "velocity": .3, "drum": "snare" if "snare" in self.kit else next(iter(self.kit))}
            grip, normal, tangent = self._anchor(rest, side)
            lift, phase, near = .35, "rest", math.inf
        else:
            times = [e["time"] for e in events]
            index = int(np.searchsorted(times, t, side="right"))
            if index == 0:
                after = events[0]
                before = {**after, "time": after["time"]-1.}
                phase = "approach" if after["time"]-t < .20 else "rest"
                t_eval = max(before["time"]+.13, t)
            elif index == len(events):
                before = events[-1]
                after = {**before, "time": before["time"]+1.}
                phase = "rebound" if t-before["time"] < .13 else "rest"
                t_eval = min(after["time"]-.20, t)
            else:
                before, after = events[index-1:index+1]
                phase = "transfer"
                t_eval = t
            duration = after["time"]-before["time"]
            u = _smooth((t_eval-before["time"])/duration)
            g0, n0, d0 = self._anchor(before, side)
            g1, n1, d1 = self._anchor(after, side)
            grip = g0+(g1-g0)*u
            normal = unit(n0+(n1-n0)*u)
            tangent = d0+(d1-d0)*u
            tangent = unit(tangent-normal*np.dot(tangent, normal))
            lift = self._lift(t_eval, before, after)
            near = min(abs(t-e["time"]) for e in events)
            if near < 1e-8:
                phase = "contact"
        angle = self.rig.strike_pitch-lift
        direction = tangent*math.cos(angle)-normal*math.sin(angle)
        desired_wrist = grip-direction*self.rig.wrist_to_grip
        shoulder, elbow, wrist, flex, reach_error = solve_arm(desired_wrist, side, self.rig)
        # Move the entire hand/stick together when reach bounds are active.
        grip = grip+(wrist-desired_wrist)
        tip = grip+direction*self.rig.grip_to_tip
        butt = grip-direction*(self.rig.stick_length-self.rig.grip_to_tip)
        # Smooth, bounded activation: soft notes still retain the grip, accents
        # firm it more. Multiplicative overlap avoids a nearest-event jump.
        relaxed = 1.
        for event in events:
            activation = (.22+.50*event["velocity"])/.72*math.exp(-.5*((t-event["time"])/.025)**2)
            relaxed *= 1.-activation
        pressure = .28+.72*(1.-relaxed)
        # Five retained grip proxies. Each capsule centerline is tangent to the
        # stick, with 6 mm or more separation along its axis. Retargeting these
        # contact points into the anatomical hand rig is deliberately separate.
        radial = unit(np.cross(direction, normal))
        other = unit(np.cross(direction, radial))
        fingers = []
        for i, offset in enumerate((-.034, -.017, 0., .017, .034)):
            angles = np.radians([35, 110, 185, 255] if i else [210, 265, 320, 375])
            # Circumscribe the shaft: putting vertices on its radius would make
            # the phalange chords cut through it. The longest segment touches.
            radius = (self.rig.stick_radius+self.rig.finger_radius)/math.cos(max(np.diff(angles))/2)
            fingers.append([grip+offset*direction+radius*(math.cos(a)*radial+math.sin(a)*other) for a in angles])
        return {"shoulder": shoulder, "elbow": elbow, "wrist": wrist, "grip": grip,
                "tip": tip, "butt": butt, "fingers": np.asarray(fingers), "phase": phase,
                "pressure": pressure, "elbow_flex_deg": flex, "reach_error": reach_error,
                "stroke_lift_rad": lift}

    def frame(self, t):
        return {"time": float(t), "hands": {h: self.pose(h, t) for h in ("L", "R")}}

    def audit(self, times):
        """Continuous event samples supplement frame samples; off-grid hits count."""
        rig = self.rig
        times = sorted(set(map(float, times)) | {e["time"] for e in self.events})
        if not times or not all(math.isfinite(t) for t in times):
            raise ValueError("Audit needs at least one finite sample time")
        stats = {"samples": len(times), "events": len(self.events), "hits": 0,
                 "max_contact_error_mm": 0., "max_reach_error_mm": 0.,
                 "max_arm_length_error_mm": 0., "max_stick_length_error_mm": 0.,
                 "max_surface_penetration_mm": 0., "min_stick_gap_mm": math.inf,
                 "max_finger_surface_penetration_mm": 0.,
                 "min_finger_gap_mm": math.inf, "min_grip_gap_mm": math.inf,
                 "max_grip_detachment_mm": 0., "elbow_min_deg": 180., "elbow_max_deg": 0.}
        stats.update(max_wrist_speed_m_s=0., max_tip_speed_m_s=0.)
        previous = None
        for t in times:
            poses = [self.pose(h, t) for h in ("L", "R")]
            if previous is not None:
                dt = t-previous[0]
                # The exact event may differ from a frame time by one floating-
                # point ulp. Audit both poses, without dividing roundoff by an ulp.
                if dt > 1e-9:
                    for p, old in zip(poses, previous[1]):
                        for joint in ("wrist", "tip"):
                            field = f"max_{joint}_speed_m_s"
                            stats[field] = max(stats[field], float(np.linalg.norm(p[joint]-old[joint])/dt))
            previous = t, poses
            stats["min_stick_gap_mm"] = min(stats["min_stick_gap_mm"], 1000*(segment_distance(
                poses[0]["butt"], poses[0]["tip"], poses[1]["butt"], poses[1]["tip"])-2*rig.stick_radius))
            for p in poses:
                stats["max_reach_error_mm"] = max(stats["max_reach_error_mm"], 1000*p["reach_error"])
                stats["elbow_min_deg"] = min(stats["elbow_min_deg"], p["elbow_flex_deg"])
                stats["elbow_max_deg"] = max(stats["elbow_max_deg"], p["elbow_flex_deg"])
                stats["max_arm_length_error_mm"] = max(stats["max_arm_length_error_mm"],
                    1000*abs(np.linalg.norm(p["shoulder"]-p["elbow"])-rig.upper_arm),
                    1000*abs(np.linalg.norm(p["elbow"]-p["wrist"])-rig.forearm))
                stats["max_stick_length_error_mm"] = max(stats["max_stick_length_error_mm"],
                    1000*abs(np.linalg.norm(p["tip"]-p["butt"])-rig.stick_length))
                for i in range(5):
                    grip_gap = min(segment_distance(a, b, p["butt"], p["tip"])
                                   for a, b in zip(p["fingers"][i][:-1], p["fingers"][i][1:]))
                    grip_gap = 1000*(grip_gap-rig.finger_radius-rig.stick_radius)
                    stats["min_grip_gap_mm"] = min(stats["min_grip_gap_mm"], grip_gap)
                    stats["max_grip_detachment_mm"] = max(stats["max_grip_detachment_mm"], grip_gap)
                    for j in range(i+1, 5):
                        for a, b in zip(p["fingers"][i][:-1], p["fingers"][i][1:]):
                            for c, d in zip(p["fingers"][j][:-1], p["fingers"][j][1:]):
                                stats["min_finger_gap_mm"] = min(stats["min_finger_gap_mm"],
                                    1000*(segment_distance(a, b, c, d)-2*rig.finger_radius))
                # Shaft capsules against finite drum disks. Sampling includes the
                # tip; this is a sampled preflight, not a collision solver.
                finger_samples = (p["fingers"][:, :-1, None, :]
                    + (p["fingers"][:, 1:, None, :]-p["fingers"][:, :-1, None, :])
                    * np.linspace(0, 1, 5)[None, None, :, None]).reshape(-1, 3)
                for drum in self.kit.values():
                    samples = p["butt"]+(p["tip"]-p["butt"])*np.linspace(0, 1, 65)[:, None]
                    relative = samples-np.asarray(drum.center)
                    heights = relative@np.asarray(drum.normal)
                    radial = np.linalg.norm(relative-heights[:, None]*np.asarray(drum.normal), axis=1)
                    mask = radial <= drum.radius+rig.stick_radius
                    radii = np.full(65, rig.stick_radius)
                    radii[-1] = rig.tip_radius
                    if np.any(mask):
                        stats["max_surface_penetration_mm"] = max(stats["max_surface_penetration_mm"],
                            1000*float(np.max(radii[mask]-heights[mask])))
                    relative = finger_samples-np.asarray(drum.center)
                    heights = relative@np.asarray(drum.normal)
                    radial = np.linalg.norm(relative-heights[:, None]*np.asarray(drum.normal), axis=1)
                    mask = radial <= drum.radius+rig.finger_radius
                    if np.any(mask):
                        stats["max_finger_surface_penetration_mm"] = max(stats["max_finger_surface_penetration_mm"],
                            1000*float(np.max(rig.finger_radius-heights[mask])))
        missed = []
        for event in self.events:
            p = self.pose(event["hand"], event["time"])
            drum = self.kit[event["drum"]]
            relative = p["tip"]-np.asarray(drum.center)
            height = np.dot(relative, drum.normal)
            radial = np.linalg.norm(relative-height*np.asarray(drum.normal))
            error = max(abs(height-rig.tip_radius), max(0., radial-(drum.radius-rig.tip_radius)))
            stats["max_contact_error_mm"] = max(stats["max_contact_error_mm"], 1000*error)
            stats["hits"] += int(error <= .002)
            if error > .002:
                missed.append({**event, "contact_error_mm": 1000*error})
        gates = {
            "missed_contact": stats["hits"] == stats["events"],
            "reach": stats["max_reach_error_mm"] < .1,
            "drum_penetration": stats["max_surface_penetration_mm"] < .5,
            "finger_drum_penetration": stats["max_finger_surface_penetration_mm"] < .5,
            "stick_collision": stats["min_stick_gap_mm"] >= 0,
            "finger_collision": stats["min_finger_gap_mm"] >= 2.,
            "grip_penetration": stats["min_grip_gap_mm"] > -.01,
            "lost_grip": stats["max_grip_detachment_mm"] < .5,
            "arm_stretch": stats["max_arm_length_error_mm"] < .001,
            "stick_stretch": stats["max_stick_length_error_mm"] < .001,
            "elbow_bounds": (stats["elbow_min_deg"] >= rig.elbow_min_deg-1e-6
                             and stats["elbow_max_deg"] <= rig.elbow_max_deg+1e-6),
            "wrist_speed": stats["max_wrist_speed_m_s"] <= rig.max_wrist_speed,
            "tip_speed": stats["max_tip_speed_m_s"] <= rig.max_tip_speed,
        }
        stats["violations"] = [name for name, passed in gates.items() if not passed]
        stats["missed_events"] = missed
        stats["passed"] = not stats["violations"]
        return stats


def demo_events():
    """Short explicit test pattern: accents, ghost notes, rolls and kit transfers."""
    pattern = [("hihat", "L", .45), ("snare", "R", .85), ("hihat", "L", .35),
               ("snare", "R", .25), ("tom", "L", .65), ("ride", "R", .95),
               ("snare", "L", .6), ("snare", "R", .7), ("snare", "L", .5),
               ("snare", "R", .8)]
    times = [.30, .55, .80, 1.05, 1.30, 1.55, 1.80, 1.90, 2.00, 2.10]
    return [dict(time=t, drum=d, hand=h, velocity=v) for t, (d, h, v) in zip(times, pattern)]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", help="JSON list or {events, kit?, rig?}; velocities are normalized [0, 1]")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=60.)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if bool(args.events) == args.demo:
        parser.error("choose exactly one of --events or --demo")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    data = {"events": demo_events()} if args.demo else json.load(open(args.events))
    if isinstance(data, list):
        data = {"events": data}
    model = Drummer(data["events"], tuple(Drum(**d) for d in data["kit"]) if "kit" in data else DEFAULT_KIT,
                    Rig(**data.get("rig", {})))
    duration = args.duration if args.duration is not None else max((e["time"] for e in model.events), default=0.)+.4
    if not math.isfinite(duration) or duration <= 0 or any(e["time"] > duration for e in model.events):
        parser.error("duration must be finite, positive, and include every hit")
    times = np.arange(math.ceil(duration*args.fps)+1)/args.fps
    # Audit at 240 Hz minimum; output stays on the requested uniform frame clock.
    audit_times = np.arange(math.ceil(duration*max(240, args.fps))+1)/max(240, args.fps)
    output = {"schema": "drummer-blockout-v1", "fps": args.fps, "duration": duration,
              "kit": [asdict(d) for d in model.kit.values()], "rig": asdict(model.rig),
              "events": model.events, "frames": [model.frame(t) for t in times], "qa": model.audit(audit_times)}
    with open(args.out, "w") as stream:
        json.dump(output, stream, default=_json_value, allow_nan=False)
    print(json.dumps(output["qa"], indent=2))
    return 0 if output["qa"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

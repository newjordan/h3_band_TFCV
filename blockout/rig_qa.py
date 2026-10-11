"""Independent, frame-resolved piano rig audit (metres/radians in, mm/degrees out).

The historical key-sync score only asks whether a key went down once near onset.
This audit also measures sustained pad contact, release, geometry and motion.
Capsules are approximate skin proxies, not a claim of exact mesh collision.
No score is silently made perfect by dropping unassigned or unreachable notes.
"""
import itertools
import math

import numpy as np


def _stats(values, scale=1.0):
    a = np.asarray(values, float).ravel() * scale
    a = a[np.isfinite(a)]
    return {"samples": int(a.size), "min": float(a.min()) if a.size else None, "median": float(np.median(a)) if a.size else None,
            "p95": float(np.percentile(a, 95)) if a.size else None,
            "max": float(a.max()) if a.size else None}


def segment_closest(a, b, c, d):
    """Closest points on finite segments, including parallel/zero-length segments."""
    a, b, c, d = (np.asarray(v, float) for v in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
    if aa < 1e-16:
        s, t = 0., float(np.clip(ee / cc, 0, 1)) if cc > 1e-16 else 0.
    elif cc < 1e-16:
        s, t = float(np.clip(-dd / aa, 0, 1)), 0.
    else:
        den = aa * cc - bb * bb
        s = float(np.clip((bb * ee - cc * dd) / den, 0, 1)) if den > 1e-16 else 0.
        t = (bb * s + ee) / cc
        if t < 0:
            t, s = 0., float(np.clip(-dd / aa, 0, 1))
        elif t > 1:
            t, s = 1., float(np.clip((bb - dd) / aa, 0, 1))
    return a + s * u, c + t * v


def capsule_gap(a, b):
    """Signed surface clearance: negative means overlap of the capsule proxies."""
    p, q = segment_closest(a[0], a[1], b[0], b[1])
    return float(np.linalg.norm(p - q) - a[2] - b[2])


def _capsules(hand, radii):
    caps = []
    for f, points in enumerate(hand["fingers"]):
        p = np.asarray(points, float)
        for s in range(3):
            # The first 20% of MCP segments live in the webbing of the palm.
            a = p[s] + (0.2 if s == 0 else 0.) * (p[s + 1] - p[s])
            caps.append((a, p[s + 1], float((radii[f][s] + radii[f][s + 1]) / 2), f, s))
    return caps


def key_surface(key, y, depth, travel=0.010):
    """Hinged key top at the actual contact y, with travel measured at its front."""
    lever = float(np.clip((key["y0"] + key["l"] - y) / key["l"], 0, 1))
    return float(key["top"] - depth * travel * lever)


def pad_on_key(pad, key, keyboard, edge_tolerance=0.00001):
    """Geometric ownership, including black keys occluding rear white-key areas."""
    x, y, _ = pad
    gap = key.get("gap", 0.0 if key["black"] else 0.0006)
    if not (abs(x - key["x"]) <= key["w"] / 2 - gap + edge_tolerance
            and key["y0"] - edge_tolerance <= y <= key["y0"] + key["l"] + edge_tolerance):
        return False
    if not key["black"]:
        for k in keyboard.values():
            if k["black"] and k["y0"] <= y <= k["y0"] + k["l"] and abs(x - k["x"]) < k["w"] / 2:
                return False
    return True


def rig_joint_limits():
    """Actual projected solver limits, including neutral-centred MCP abduction."""
    from blockout.rig.hand import HandRig
    result = {}
    for side in ("L", "R"):
        rig = HandRig(side)
        result[side] = {(bone, dof): (float(lo), float(hi))
                        for ch in rig.chains
                        for (bone, dof, *_), lo, hi in zip(ch.dofs, ch.lo, ch.hi)}
    return result


def _note(n):
    if isinstance(n, (list, tuple)):
        return float(n[0]), float(n[1]), int(n[2])
    if isinstance(n, dict):
        return float(n["start"]), float(n["end"]), int(n["pitch"])
    return float(n.start), float(n.end), int(n.pitch)


def analyze(anim, notes=(), *, joint_limits=None, start=None, duration=None,
            contact_tolerance=0.002, collision_tolerance=0.003, expected_arm_lengths=None):
    """Audit one unstitched animation. start/duration delimit note onset scoring.

    Frame geometry and motion use that same interval. Hold/release use all supplied
    frames, so callers should generate at least 0.6 s after the scored interval.
    Notes and start are in performed absolute seconds; frame t is clip-relative.
    Missing observations are reported rather than counted as successful releases.
    """
    frames = anim["frames"]
    fps = float(anim["fps"])
    times = np.array([anim.get("start", 0.) + f.get("t", j / fps) for j, f in enumerate(frames)])
    if not len(frames):
        raise ValueError("Cannot audit an animation with no frames")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must increase strictly")
    start = float(times[0] if start is None else start)
    end = float(times[-1] + 0.5 / fps if duration is None else start + duration)
    indices = np.flatnonzero((times >= start - 1e-7) & (times < end - 1e-7))
    if not len(indices):
        raise ValueError("Requested interval contains no frames")
    keyboard = {int(k["pitch"]): k for k in anim["keyboard"]}
    radii = np.asarray(anim["finger_radii"], float)
    limits = joint_limits if joint_limits is not None else rig_joint_limits()
    ns = [_note(n) for n in notes]
    assignment = {}
    for side, slices in anim.get("fingering", {}).items():
        for sl in slices:
            for t, pitch, finger in sl:
                assignment.setdefault((round(t, 3), int(pitch)), []).append((side, int(finger)))
    self_gap, inter_gap, tip_gap = [], [], []
    bounds_bad, bounds_seen, bounds_worst, bounds_unchecked = 0, 0, 0., 0
    bounds_by_joint = {}
    bone_lengths, pad_paths, angle_paths = {}, {}, {}
    wrist_paths, elbow_paths, arm_lengths, wrist_deviation = {}, {}, {}, []
    skeletons, fk_error = {}, []
    unsupported_keys = depressed_keys = depth_bounds = 0
    surface_gaps, examples = [], []
    planned_total = planned_contact = planned_penetrating = planned_off = 0
    activation_values, activation_paths = [], {}
    worst_self = {"gap_mm": 0.}
    worst_inter = {"gap_mm": 0.}
    for j in indices:
        frame = frames[j]
        hands = frame["hands"]
        caps = {}
        for side, hand in hands.items():
            for field in ("fingers", "pads", "wrist"):
                if not np.all(np.isfinite(np.asarray(hand[field], float))):
                    raise ValueError(f"Non-finite {side} {field} in frame {j}")
            if "root" in hand and "angles" in hand:
                if side not in skeletons:
                    from blockout.rig.skeleton import Skeleton
                    skeletons[side] = Skeleton(side)
                sk = skeletons[side]
                _, heads, tails = sk.fk(np.asarray(hand["root"]["rot"]), np.asarray(hand["root"]["pos"]), hand["angles"])
                actual = np.asarray([sk.chain_points(heads, tails, f) for f in range(1, 6)])
                fk_error.extend(np.linalg.norm(actual - hand["fingers"], axis=2).ravel())
            caps[side] = _capsules(hand, radii)
            pairs = [(a, b) for a, b in itertools.combinations(caps[side], 2)
                     if a[3] != b[3] and not (a[4] == b[4] == 0)]
            gaps = [capsule_gap(a, b) for a, b in pairs]
            self_gap.append(min(gaps) if gaps else math.nan)
            if gaps and min(gaps) * 1000 < worst_self["gap_mm"]:
                a, b = pairs[int(np.argmin(gaps))]
                worst_self = {"gap_mm": min(gaps) * 1000, "time": float(times[j]), "side": side,
                              "finger_segments": [[a[3] + 1, a[4] + 1], [b[3] + 1, b[4] + 1]]}
            tip_gap.extend(float(np.linalg.norm(np.asarray(hand["pads"][a]) - hand["pads"][b]))
                           for a, b in itertools.combinations(range(5), 2))
            bone_lengths.setdefault(side, []).append(np.linalg.norm(np.diff(hand["fingers"], axis=1), axis=2))
            pad_paths.setdefault(side, []).append(hand["pads"])
            wrist_paths.setdefault(side, []).append(hand["wrist"])
            if "elbow" in hand:
                elbow_paths.setdefault(side, []).append(hand["elbow"])
                wrist, elbow = np.asarray(hand["wrist"]), np.asarray(hand["elbow"])
                forearm = wrist - elbow
                forward = np.mean(np.asarray(hand["fingers"])[1:, 0], axis=0) - wrist
                norm = np.linalg.norm(forearm) * np.linalg.norm(forward)
                if norm > 1e-12:
                    wrist_deviation.append(math.degrees(math.acos(float(np.clip(forearm @ forward / norm, -1, 1)))))
                shoulder = frame.get("body", {}).get("shoulders", {}).get(side)
                if shoulder is not None:
                    arm_lengths.setdefault(side, []).append([np.linalg.norm(elbow - shoulder), np.linalg.norm(forearm)])
            names = sorted((bone, dof) for bone, ds in hand.get("angles", {}).items() for dof in ds)
            angle_paths.setdefault(side, []).append([hand["angles"][b][d] for b, d in names])
            if "activation" in hand:
                activation_values.extend(hand["activation"])
                activation_paths.setdefault(side, []).append(hand["activation"])
                for finger, (active, pitch) in enumerate(zip(hand["activation"], hand.get("assigned_keys", []))):
                    # Controller holds at .8 after its firm attack; pre-onset
                    # approach is intentionally above the surface, not a miss.
                    if pitch is None or active < .79 or int(pitch) not in keyboard:
                        continue
                    if not any(p == int(pitch) and on <= times[j] <= off for on, off, p in ns):
                        continue
                    pad, key = hand["pads"][finger], keyboard[int(pitch)]
                    depth = float(frame["keys"].get(str(pitch), 0.))
                    on = pad_on_key(pad, key, keyboard)
                    gap = pad[2] - key_surface(key, pad[1], depth)
                    planned_total += 1
                    planned_contact += on and abs(gap) <= contact_tolerance and depth >= .5
                    planned_penetrating += on and gap < -contact_tolerance
                    planned_off += not on
            for bone, dof in names:
                if (bone, dof) not in limits.get(side, {}):
                    bounds_unchecked += 1
                    continue
                lo, hi = limits[side][bone, dof]
                q = hand["angles"][bone][dof]
                if not math.isfinite(q):
                    raise ValueError(f"Non-finite {side} {bone} {dof} in frame {j}")
                # JSON angles are rounded to 5 decimal places.
                excess = max(lo - q, q - hi, 0.)
                bounds_seen += 1
                bounds_worst = max(bounds_worst, excess)
                if excess > 1e-5:
                    bounds_bad += 1
                    name = f"{side}:{bone}:{dof}"
                    bounds_by_joint[name] = bounds_by_joint.get(name, 0) + 1
        for left, right in itertools.combinations(caps, 2):
            # Also check the opposite palm/forearm, without same-hand webbing.
            def full(side):
                h = hands[side]
                extra = [(np.asarray(h["wrist"]), np.asarray(p[0]), 0.015, 5, f)
                         for f, p in enumerate(h["fingers"])]
                if "elbow" in h:
                    extra.append((np.asarray(h["wrist"]), np.asarray(h["elbow"]), 0.030, 6, 0))
                return caps[side] + extra
            A, B = full(left), full(right)
            # The bounding-sphere test avoids work for hands far apart.
            gaps = []
            for a in A:
                for b in B:
                    lower = np.linalg.norm((a[0] + a[1] - b[0] - b[1]) / 2) - (np.linalg.norm(a[1] - a[0]) + np.linalg.norm(b[1] - b[0])) / 2 - a[2] - b[2]
                    if lower <= 0.04:
                        gap = capsule_gap(a, b)
                        gaps.append(gap)
                        if gap * 1000 < worst_inter["gap_mm"]:
                            def label(cap):
                                return f"finger{cap[3] + 1}:segment{cap[4] + 1}" if cap[3] < 5 else "palm" if cap[3] == 5 else "forearm"
                            worst_inter = {"gap_mm": gap * 1000, "time": float(times[j]),
                                           "parts": [f"{left}:{label(a)}", f"{right}:{label(b)}"]}
            inter_gap.append(min(min(gaps), 0.04) if gaps else 0.04)
        for pitch, value in frame["keys"].items():
            p, depth = int(pitch), float(value)
            if not math.isfinite(depth):
                raise ValueError(f"Non-finite key {p} depth in frame {j}")
            depth_bounds += int(not (0 <= depth <= 1))
            if depth < 0.5 or p not in keyboard:
                continue
            depressed_keys += 1
            gaps = [pad[2] - key_surface(keyboard[p], pad[1], depth)
                    for h in hands.values() for pad in h["pads"] if pad_on_key(pad, keyboard[p], keyboard)]
            surface_gaps.extend(gaps)
            unsupported_keys += int(not any(abs(g) <= contact_tolerance for g in gaps))

    note_results = []
    hold_total = hold_valid = hold_down = hold_off = hold_deep = hold_float = 0
    hold_slip = []
    hold_censored = 0
    releases, release_failed, release_missing, release_restrike = [], 0, 0, 0
    for onset, offset, pitch in ns:
        if not start <= onset < end or pitch not in keyboard:
            continue
        assigned = assignment.get((round(onset, 3), pitch), [])
        onset_frames = np.flatnonzero((times >= onset - 1e-7) & (times <= min(offset, onset + 0.25) + 1e-7))
        hold_start = onset + min(0.080, max(offset - onset, 0) * 0.25)
        hold_end = offset - min(0.040, max(offset - onset, 0) * 0.10)
        held = np.flatnonzero((times >= hold_start) & (times <= hold_end))
        hold_censored += hold_end > times[-1] + .5 / fps
        def contacts(j):
            out = []
            for side, finger in assigned:
                hand = frames[j]["hands"].get(side)
                if hand is None:
                    continue
                pad = hand["pads"][finger]
                d = float(frames[j]["keys"].get(str(pitch), 0.))
                on = pad_on_key(pad, keyboard[pitch], keyboard)
                gap = pad[2] - key_surface(keyboard[pitch], pad[1], d)
                out.append((on and abs(gap) <= contact_tolerance and d >= 0.5, on, gap, pad))
            return out
        onset_depth = max((frames[j]["keys"].get(str(pitch), 0.) for j in onset_frames), default=0.)
        onset_contact = any(any(c[0] for c in contacts(j)) for j in onset_frames)
        valid = 0
        previous_pad = previous_t = None
        for j in held:
            cs = contacts(j)
            best = min(cs, key=lambda c: (not c[0], not c[1], abs(c[2]))) if cs else None
            good = bool(best and best[0])
            hold_total += 1
            hold_valid += good
            valid += good
            hold_down += frames[j]["keys"].get(str(pitch), 0.) >= 0.5
            hold_off += not bool(best and best[1])
            hold_deep += bool(best and best[1] and best[2] < -contact_tolerance)
            hold_float += bool(best and best[1] and best[2] > contact_tolerance)
            if good and previous_pad is not None:
                hold_slip.append(float(np.linalg.norm(np.asarray(best[3])[:2] - previous_pad[:2]) / (times[j] - previous_t)))
            previous_pad = np.asarray(best[3]) if good else None
            previous_t = times[j]
        repeated = any(p == pitch and offset - 1 / fps <= s <= offset + .25 and s > onset + 1e-6 for s, _, p in ns)
        if repeated:
            release_restrike += 1
        elif offset + .25 > times[-1] + 1e-7:
            release_missing += 1
        else:
            rr = np.flatnonzero((times >= offset) & (times <= offset + .25))
            released = [j for j in rr if frames[j]["keys"].get(str(pitch), 0.) <= .1]
            if released:
                releases.append(float(times[released[0]] - offset))
            else:
                release_failed += 1
        result = {"onset": onset, "pitch": pitch, "assigned": bool(assigned), "key_down_onset": onset_depth >= .5,
                  "physical_onset": onset_contact, "hold_frames": len(held), "hold_contact_fraction": valid / len(held) if len(held) else None}
        note_results.append(result)
        if (not onset_contact or (len(held) and valid < len(held) * .8)) and len(examples) < 30:
            examples.append(result)

    speed, acceleration, jerk, qspeed, activation_speed = [], [], [], [], []
    wrist_speed, elbow_speed, elbow_step, arm_drift = [], [], [], []
    reversals = qsteps = 0
    length_deviation = []
    dt = np.diff(times[indices])
    for side, path in pad_paths.items():
        p = np.asarray(path)
        v = np.diff(p, axis=0) / dt[:, None, None]
        speed.extend(np.linalg.norm(v, axis=2).ravel())
        if len(v) > 1:
            a = np.diff(v, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None, None]
            acceleration.extend(np.linalg.norm(a, axis=2).ravel())
            if len(a) > 1:
                jerk.extend(np.linalg.norm(np.diff(a, axis=0) / dt[1:-1, None, None], axis=2).ravel())
        q = np.asarray(angle_paths[side])
        if q.size and len(q) > 1:
            dq = np.degrees(np.diff(q, axis=0))
            qspeed.extend((np.abs(dq) / dt[:, None]).ravel())
            if len(dq) > 1:
                reversals += int(((dq[1:] * dq[:-1] < 0) & (np.abs(dq[1:]) > 2) & (np.abs(dq[:-1]) > 2)).sum())
                qsteps += int(dq[1:].size)
        lengths = np.asarray(bone_lengths[side])
        length_deviation.extend(np.abs(lengths - np.median(lengths, axis=0)).ravel())
        if side in activation_paths and len(activation_paths[side]) == len(dt) + 1:
            activation_speed.extend((np.abs(np.diff(activation_paths[side], axis=0)) / dt[:, None]).ravel())
        if len(wrist_paths[side]) == len(dt) + 1:
            wrist_speed.extend(np.linalg.norm(np.diff(wrist_paths[side], axis=0), axis=1) / dt)
        if len(elbow_paths.get(side, [])) == len(dt) + 1:
            displacement = np.linalg.norm(np.diff(elbow_paths[side], axis=0), axis=1)
            elbow_step.extend(displacement)
            elbow_speed.extend(displacement / dt)
        if side in arm_lengths:
            length = np.asarray(arm_lengths[side])
            arm_drift.extend(np.abs(length - np.median(length, axis=0)).ravel())

    arm_error, arm_error_all = [], []
    if arm_lengths:
        if expected_arm_lengths is None:
            from blockout.performer import UPPER_ARM, FOREARM
            expected_arm_lengths = (UPPER_ARM, FOREARM)
        expected_arm_lengths = np.asarray(expected_arm_lengths, float)
        for length in arm_lengths.values():
            arm_error.extend(np.abs(np.asarray(length) - expected_arm_lengths).ravel())
        # Warmup and release-tail frames are also exported/renderable. A fixed
        # stretch there must not disappear just because onset scoring is cropped.
        for frame in frames:
            for side, hand in frame["hands"].items():
                shoulder = frame.get("body", {}).get("shoulders", {}).get(side)
                if shoulder is None or "elbow" not in hand:
                    continue
                elbow, wrist = np.asarray(hand["elbow"]), np.asarray(hand["wrist"])
                measured = np.array([np.linalg.norm(elbow - shoulder), np.linalg.norm(wrist - elbow)])
                arm_error_all.extend(np.abs(measured - expected_arm_lengths))

    def clearance(vals):
        arr = np.asarray(vals)
        arr = arr[np.isfinite(arr)]
        return {"observations": len(arr), "overlap_over_3mm": int((arr < -collision_tolerance).sum()),
                "min_clearance_mm": float(arr.min() * 1000) if len(arr) else None,
                "p05_clearance_mm": float(np.percentile(arr, 5) * 1000) if len(arr) else None}
    count = len(note_results)
    return {
        "schema": 1, "start": start, "duration": end - start, "frames": len(indices), "fps": fps,
        "animation_interval": [float(times[0]), float(times[-1])],
        "assumptions": {"contact_tolerance_mm": contact_tolerance * 1000, "collision_tolerance_mm": collision_tolerance * 1000,
                        "footprint_numerical_tolerance_mm": .01, "legacy_white_key_gap_per_side_mm": .6,
                        "expected_arm_lengths_m": None if expected_arm_lengths is None else np.asarray(expected_arm_lengths).tolist(),
                        "collision_geometry": "mean-radius bone capsule approximation, not exact mesh collision; excludes same-hand proximal/proximal webbing",
                        "hold_scope": "score MIDI duration fidelity, including failures caused by planned early release; controller_phase separately measures explicit assigned firm phases",
                        "interhand_clearance_upper_bound_mm": 40},
        "notes": {"count": count, "assigned": sum(n["assigned"] for n in note_results),
                  "onset_key_down": sum(n["key_down_onset"] for n in note_results),
                  "onset_physical_contact": sum(n["physical_onset"] for n in note_results),
                  "hold_note_frames": hold_total, "hold_contact_frames": hold_valid,
                  "hold_contact_fraction": hold_valid / hold_total if hold_total else None,
                  "hold_key_down_fraction": hold_down / hold_total if hold_total else None,
                  "hold_off_key_frames": hold_off, "hold_penetrating_frames": hold_deep, "hold_floating_frames": hold_float,
                  "hold_censored_clip_end": int(hold_censored),
                  "hold_lateral_speed_mm_s": _stats(hold_slip, 1000),
                  "release_latency_ms": _stats(releases, 1000), "release_failed_after_250ms": release_failed,
                  "release_censored_clip_end": release_missing, "release_censored_restrike": release_restrike},
        "contact": {"depressed_key_frames": depressed_keys, "unsupported_depressed_key_frames": unsupported_keys,
                    "key_depth_bound_violations": depth_bounds, "pad_surface_signed_gap_mm": _stats(surface_gaps, 1000)},
        "controller_phase": {"available": bool(activation_values), "firm_assigned_finger_frames": planned_total,
                             "definition": "assigned key, activation >= 0.79, within a matching score note; includes .8 hold gain, excludes pre-onset approach",
                             "firm_contact_frames": int(planned_contact),
                             "firm_contact_fraction": planned_contact / planned_total if planned_total else None,
                             "firm_penetrating_frames": int(planned_penetrating), "firm_off_key_frames": int(planned_off),
                             "activation_bound_violations": sum(not 0 <= v <= 1 for v in activation_values),
                             "activation_change_per_s": _stats(activation_speed)},
        "separation": {"self_hand_frames": clearance(self_gap), "interhand_frames": clearance(inter_gap),
                       "pad_pair_distance_mm": _stats(tip_gap, 1000), "worst_self": worst_self, "worst_inter": worst_inter},
        "bounds": {"joint_dof_frames": bounds_seen, "unchecked_joint_dof_frames": bounds_unchecked,
                   "joint_limit_violations": bounds_bad,
                   "worst_excess_deg": math.degrees(bounds_worst), "violations_by_joint": bounds_by_joint,
                   "bone_length_deviation_mm": _stats(length_deviation, 1000),
                   "fk_vs_exported_chain_mm": _stats(fk_error, 1000),
                   "arm_length_deviation_mm": _stats(arm_drift, 1000),
                   "arm_length_error_mm": _stats(arm_error, 1000),
                   "arm_length_error_all_clip_mm": _stats(arm_error_all, 1000),
                   "arm_length_violations_all_clip_over_0_1mm": int((np.asarray(arm_error_all) > .0001).sum())},
        "motion": {"pad_speed_m_s": _stats(speed), "pad_acceleration_m_s2": _stats(acceleration),
                   "wrist_speed_m_s": _stats(wrist_speed), "elbow_speed_m_s": _stats(elbow_speed),
                   "elbow_step_mm": _stats(elbow_step, 1000), "forearm_hand_angle_deg": _stats(wrist_deviation),
                   "pad_jerk_m_s3": _stats(jerk), "joint_speed_deg_s": _stats(qspeed),
                   "joint_steps_above_12rad_s": int((np.radians(qspeed) > 12.02).sum()),
                   "joint_reversals_over_2deg": reversals,
                   "reversal_rate_per_joint_s": reversals * fps / qsteps if qsteps else 0.},
        "problem_notes": examples,
    }

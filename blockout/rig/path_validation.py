"""Independent acceptance of sampled path observations, never a CCD claim."""
import math


MOTION_LIMITS = dict(joint_speed_rad_s=12., wrist_speed_m_s=.8,
                     wrist_angular_speed_rad_s=4., elbow_plane_speed_rad_s=4.)


def sampled_path_violations(observations):
    """Keep every required gate in sweep termination and candidate acceptance."""
    failures = []
    if not observations:
        return [dict(node=None, side=None, reasons=['empty path'])]
    for index, frame in enumerate(observations):
        reasons = []
        if frame.get('unsupported_pitches'):
            reasons.append('held support')
        travel = frame.get('key_travel_excess', math.inf)
        if not math.isfinite(travel) or travel > 1e-9:
            reasons.append('key travel')
        if frame.get('mechanical_contacts'):
            reasons.append('key mechanical interference')
        if not frame.get('hands'):
            reasons.append('missing hands')
        if reasons:
            failures.append(dict(node=frame.get('node', index), side=None, reasons=reasons))
        for side, hand in frame.get('hands', {}).items():
            reasons = []
            if not hand.get('instrument_clear', False):
                reasons.append('instrument skin')
            for name, low, high in (
                    ('proxy_self_clearance_mm', -1e-3, math.inf),
                    ('joint_bounds_excess_rad', 0., 1e-9),
                    ('arm_length_error_mm', 0., .001),
                    ('wrist_root_error_mm', 0., .001)):
                value = hand.get(name, math.nan)
                if not math.isfinite(value) or not low <= value <= high:
                    reasons.append(name)
            if index:
                for name, maximum in MOTION_LIMITS.items():
                    value = hand.get('motion', {}).get(name, math.nan)
                    if not math.isfinite(value) or not 0 <= value <= maximum + 1e-6:
                        reasons.append(name)
            if reasons:
                failures.append(dict(node=frame.get('node', index), side=side, reasons=reasons))
    return failures

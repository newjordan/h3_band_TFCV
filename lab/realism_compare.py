"""Compare independent QA reports, retaining every measured tradeoff.

python lab/realism_compare.py baseline.qa.json candidate.qa.json [--out comparison.json]
This prints JSON; it never invents a combined realism score or blesses a build.
"""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    for field in ("start", "duration", "fps"):
        if abs(baseline[field] - candidate[field]) > 1e-6:
            raise ValueError(f"Incomparable {field}: {baseline[field]} != {candidate[field]}")
    if baseline["notes"]["count"] != candidate["notes"]["count"]:
        raise ValueError("Note denominators differ")
    def metrics(r):
        n, s, m = r["notes"], r["separation"], r["motion"]
        pct = lambda numerator, denominator: 100 * numerator / denominator if denominator else None
        return {
            "onset_key_down_pct": pct(n["onset_key_down"], n["count"]),
            "onset_physical_contact_pct": pct(n["onset_physical_contact"], n["count"]),
            "hold_contact_pct": pct(n["hold_contact_frames"], n["hold_note_frames"]),
            "hold_penetration_pct": pct(n["hold_penetrating_frames"], n["hold_note_frames"]),
            "self_overlap_handframe_pct": pct(s["self_hand_frames"]["overlap_over_3mm"], s["self_hand_frames"]["observations"]),
            "interhand_overlap_frame_pct": pct(s["interhand_frames"]["overlap_over_3mm"], s["interhand_frames"]["observations"]),
            "self_max_overlap_mm": max(0., -(s["self_hand_frames"]["min_clearance_mm"] or 0.)),
            "interhand_max_overlap_mm": max(0., -(s["interhand_frames"]["min_clearance_mm"] or 0.)),
            "joint_bound_violations": r["bounds"]["joint_limit_violations"],
            "joint_speed_steps_above_12rad_s": m.get("joint_steps_above_12rad_s"),
            "joint_speed_max_deg_s": m["joint_speed_deg_s"]["max"],
            "joint_reversal_rate": m["reversal_rate_per_joint_s"],
            "pad_jerk_p95_m_s3": m["pad_jerk_m_s3"]["p95"],
            "elbow_step_max_mm": m.get("elbow_step_mm", {}).get("max"),
            "release_failed_after_250ms": n["release_failed_after_250ms"],
        }
    a, b = metrics(baseline), metrics(candidate)
    return {"start": baseline["start"], "duration": baseline["duration"], "notes": baseline["notes"]["count"],
            "equal_animation_context": baseline.get("animation_interval") == candidate.get("animation_interval")
                if "animation_interval" in baseline and "animation_interval" in candidate else None,
            "hold_frames": [baseline["notes"]["hold_note_frames"], candidate["notes"]["hold_note_frames"]],
            "metrics": {k: {"baseline": a[k], "candidate": b[k],
                            "delta": b[k] - a[k] if a[k] is not None and b[k] is not None else None} for k in a},
            "interpretation": "Positive deltas improve only contact rates; positive errors, overlap, speed and jitter are worse. Capsule overlaps are proxies. Score holds include planned early releases; clip-end holds/releases may be censored."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    print(result)

"""Regression tests for independently measured physical correctness, not scores."""
import copy
import unittest

import numpy as np

from blockout.rig_qa import analyze, capsule_gap, key_surface, pad_on_key, segment_closest


def fixture():
    key = {"pitch": 60, "black": False, "x": 0., "w": .0235, "l": .15, "y0": 0., "top": 0.}
    fingers = [[[f * .035, 0., .06], [f * .035, .02, .04], [f * .035, .04, .02], [f * .035, .06, .01]] for f in range(5)]
    hand = {"wrist": [0., -.10, .06], "fingers": fingers,
            "pads": [[0., .03, -.008]] + [[f * .035, .06, .02] for f in range(1, 5)],
            "angles": {"finger1-1": {"flex": .1}}}
    frames = []
    for j in range(21):
        frame = {"t": j / 20, "keys": {"60": 1.} if j < 12 else {}, "hands": {"R": copy.deepcopy(hand)}}
        if j >= 12:
            frame["hands"]["R"]["pads"][0][2] = .005
        frames.append(frame)
    return {"fps": 20, "start": 0., "keyboard": [key], "finger_radii": [[.004] * 4] * 5,
            "fingering": {"R": [[[0., 60, 0]]]}, "frames": frames}


class GeometryTests(unittest.TestCase):
    def test_segment_crossing_parallel_and_degenerate(self):
        p, q = segment_closest([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0])
        np.testing.assert_allclose(p, q)
        self.assertAlmostEqual(capsule_gap(([0, 0, 0], [1, 0, 0], .1), ([0, .15, 0], [1, .15, 0], .1)), -.05)
        p, q = segment_closest([0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(p - q), 1.)

    def test_hinge_uses_actual_contact_depth(self):
        k = fixture()["keyboard"][0]
        self.assertAlmostEqual(key_surface(k, .03, 1.), -.008)
        self.assertAlmostEqual(key_surface(k, .12, 1.), -.002)
        self.assertAlmostEqual(key_surface(k, .15, 1.), 0.)

    def test_black_key_owns_white_key_cutout(self):
        white = fixture()["keyboard"][0]
        black = dict(white, pitch=61, black=True, x=.010, w=.0137, y0=.055, l=.095, top=.012)
        keyboard = {60: white, 61: black}
        self.assertFalse(pad_on_key([.006, .08, 0], white, keyboard))
        self.assertTrue(pad_on_key([.006, .03, 0], white, keyboard))

    def test_white_key_physical_gap_cannot_press(self):
        white = fixture()["keyboard"][0]
        self.assertFalse(pad_on_key([white["w"] / 2 - .0002, .03, 0], white, {60: white}))


class AuditTests(unittest.TestCase):
    def audit(self, anim):
        return analyze(anim, [(0., .6, 60)], joint_limits={"R": {("finger1-1", "flex"): (0., .2)}})

    def test_valid_press_hold_and_release(self):
        r = self.audit(fixture())
        self.assertEqual(r["notes"]["onset_physical_contact"], 1)
        self.assertEqual(r["notes"]["hold_contact_fraction"], 1.)
        self.assertEqual(r["notes"]["release_failed_after_250ms"], 0)
        self.assertEqual(r["bounds"]["joint_limit_violations"], 0)

    def test_onset_only_success_cannot_hide_lost_hold(self):
        a = fixture()
        for f in a["frames"][3:12]:
            f["hands"]["R"]["pads"][0][2] = .03
        r = self.audit(a)
        self.assertEqual(r["notes"]["onset_key_down"], 1)
        self.assertEqual(r["notes"]["onset_physical_contact"], 1)
        self.assertLess(r["notes"]["hold_contact_fraction"], .3)
        self.assertGreater(r["contact"]["unsupported_depressed_key_frames"], 0)

    def test_penetration_is_not_successful_contact(self):
        a = fixture()
        for f in a["frames"][:12]:
            f["hands"]["R"]["pads"][0][2] -= .01
        r = self.audit(a)
        self.assertEqual(r["notes"]["onset_physical_contact"], 0)
        self.assertGreater(r["notes"]["hold_penetrating_frames"], 0)

    def test_missing_fingering_stays_in_denominator(self):
        a = fixture()
        a["fingering"] = {}
        r = self.audit(a)
        self.assertEqual(r["notes"]["count"], 1)
        self.assertEqual(r["notes"]["assigned"], 0)
        self.assertEqual(r["notes"]["hold_contact_fraction"], 0.)

    def test_self_collision_and_joint_bound_violation(self):
        a = fixture()
        for f in a["frames"]:
            h = f["hands"]["R"]
            h["fingers"][2] = copy.deepcopy(h["fingers"][1])
            h["angles"]["finger1-1"]["flex"] = .5
        r = self.audit(a)
        self.assertEqual(r["bounds"]["joint_limit_violations"], 21)
        self.assertEqual(r["separation"]["self_hand_frames"]["overlap_over_3mm"], 21)

    def test_release_at_clip_end_is_censored(self):
        a = fixture()
        a["frames"] = a["frames"][:14]
        r = self.audit(a)
        self.assertEqual(r["notes"]["release_censored_clip_end"], 1)
        self.assertEqual(r["notes"]["release_latency_ms"]["samples"], 0)

    def test_interhand_collision(self):
        a = fixture()
        for f in a["frames"]:
            f["hands"]["L"] = copy.deepcopy(f["hands"]["R"])
        r = self.audit(a)
        self.assertEqual(r["separation"]["interhand_frames"]["overlap_over_3mm"], 21)

    def test_stuck_key_release_is_failure(self):
        a = fixture()
        for f in a["frames"]:
            f["keys"]["60"] = 1.
        self.assertEqual(self.audit(a)["notes"]["release_failed_after_250ms"], 1)

    def test_planned_early_release_is_separate_from_score_hold(self):
        a = fixture()
        for j, f in enumerate(a["frames"]):
            h = f["hands"]["R"]
            h["activation"] = [1. if j < 5 else 0.] + [0.] * 4
            h["assigned_keys"] = [60 if j < 5 else None] + [None] * 4
            if j >= 5:
                f["keys"] = {}
                h["pads"][0][2] = .02
        r = self.audit(a)
        self.assertLess(r["notes"]["hold_contact_fraction"], .5)
        self.assertEqual(r["controller_phase"]["firm_contact_fraction"], 1.)

    def test_nonfinite_geometry_is_not_silently_ignored(self):
        a = fixture()
        a["frames"][0]["hands"]["R"]["pads"][0][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite R pads"):
            self.audit(a)

    def test_elbow_snap_is_reported_independently_of_finger_motion(self):
        a = fixture()
        for j, f in enumerate(a["frames"]):
            f["hands"]["R"]["elbow"] = [0., -.4, .05 if j < 10 else .25]
        r = self.audit(a)
        self.assertAlmostEqual(r["motion"]["elbow_step_mm"]["max"], 200.)

    def test_render_skeleton_disagreement_catches_interpolated_chain(self):
        from blockout.rig.skeleton import Skeleton
        sk = Skeleton("R")
        rot, pos = sk.rest_root()
        _, heads, tails = sk.fk(rot, pos, {})
        points = np.asarray([sk.chain_points(heads, tails, f) for f in range(1, 6)])
        a = fixture()
        for frame in a["frames"]:
            hand = frame["hands"]["R"]
            hand["root"] = {"rot": rot.tolist(), "pos": pos.tolist()}
            hand["angles"] = {}
            hand["fingers"] = points.tolist()
            hand["fingers"][1][3][2] += .01
        r = self.audit(a)
        self.assertAlmostEqual(r["bounds"]["fk_vs_exported_chain_mm"]["max"], 10.)

    def test_constant_arm_stretch_and_warmup_are_not_hidden_by_median(self):
        a = fixture()
        for j, frame in enumerate(a["frames"]):
            frame["body"] = {"shoulders": {"R": [0., 0., 0.]}}
            frame["hands"]["R"]["elbow"] = [0., -.3 if j == 0 else -.2, 0.]
            frame["hands"]["R"]["wrist"] = [0., -.6 if j == 0 else -.4, 0.]
        r = analyze(a, [], joint_limits={}, start=.1, duration=.5, expected_arm_lengths=(.2, .2))
        self.assertAlmostEqual(r["bounds"]["arm_length_deviation_mm"]["max"], 0.)
        self.assertAlmostEqual(r["bounds"]["arm_length_error_mm"]["max"], 0.)
        self.assertAlmostEqual(r["bounds"]["arm_length_error_all_clip_mm"]["max"], 100.)
        self.assertEqual(r["bounds"]["arm_length_violations_all_clip_over_0_1mm"], 2)


if __name__ == "__main__":
    unittest.main()

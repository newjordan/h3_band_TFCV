"""Contact, continuity and failure-mode tests for the new drummer controller."""
import unittest

import numpy as np

from blockout.drums import DEFAULT_KIT, Drum, Drummer, Rig, demo_events, segment_distance, segment_distances, solve_arm, stick_clearance


class DrummerTests(unittest.TestCase):
    def test_broadcast_segment_distances_match_scalar_and_degenerate_cases(self):
        rng = np.random.default_rng(6194)
        a, b, c, d = rng.normal(size=(4, 32, 3))
        b[0] = a[0]
        d[1] = c[1]
        expected = [segment_distance(*points) for points in zip(a, b, c, d)]
        np.testing.assert_allclose(segment_distances(a, b, c, d), expected, atol=1e-12)

    def test_opposite_stick_cannot_pass_through_a_finger_with_clear_sticks(self):
        class BrokenGrip(Drummer):
            def pose(self, side, t):
                pose = super().pose(side, t)
                if side == 'L':
                    other = super().pose('R', t)
                    pose['fingers'][0] = np.linspace(other['butt'], other['tip'], 4)
                return pose
        model = BrokenGrip([dict(time=.5, drum='snare', hand='R')])
        report = model.audit([.5])
        self.assertGreater(report['min_stick_gap_mm'], 0.)
        self.assertFalse(report['passed'])
        self.assertIn('finger_opposite_stick_collision', report['violations'])
        self.assertLess(report['min_finger_opposite_stick_gap_mm'], -11.)

    def test_arm_keeps_lengths_and_bounds_even_when_target_is_unreachable(self):
        rig = Rig()
        rng = np.random.default_rng(812)
        for side in ("L", "R"):
            for target in rng.uniform(-2, 2, (100, 3)):
                shoulder, elbow, wrist, angle, _ = solve_arm(target, side, rig)
                self.assertAlmostEqual(np.linalg.norm(shoulder-elbow), rig.upper_arm, places=12)
                self.assertAlmostEqual(np.linalg.norm(wrist-elbow), rig.forearm, places=12)
                self.assertGreaterEqual(angle+1e-8, rig.elbow_min_deg)
                self.assertLessEqual(angle-1e-8, rig.elbow_max_deg)

    def test_all_default_surfaces_have_exact_contact_and_rigid_sticks(self):
        for drum in DEFAULT_KIT:
            for hand in ("L", "R"):
                model = Drummer([dict(time=.4137, drum=drum.name, hand=hand)])
                p = model.pose(hand, .4137)
                self.assertLess(p["reach_error"], 1e-10)
                self.assertAlmostEqual(np.dot(p["tip"]-drum.center, drum.normal), model.rig.tip_radius, places=12)
                self.assertAlmostEqual(np.linalg.norm(p["tip"]-p["butt"]), model.rig.stick_length, places=12)

    def test_rebound_has_the_correct_direction_and_restitution(self):
        model = Drummer([dict(time=.5, drum="snare", hand="R", velocity=.8)])
        epsilon = 1e-6
        center = model.pose("R", .5)
        before, after = model.pose("R", .5-epsilon), model.pose("R", .5+epsilon)
        incoming = (center["tip"][2]-before["tip"][2])/epsilon
        outgoing = (after["tip"][2]-center["tip"][2])/epsilon
        self.assertLess(incoming, 0)
        self.assertGreater(outgoing, 0)
        self.assertAlmostEqual(outgoing/-incoming, model.rig.restitution, places=4)
        self.assertLess(np.linalg.norm(after["grip"]-before["grip"]), 1e-9)

    def test_accent_controls_height_speed_and_firmness_without_releasing_grip(self):
        soft = Drummer([dict(time=.5, drum="snare", hand="R", velocity=.15)])
        hard = Drummer([dict(time=.5, drum="snare", hand="R", velocity=.95)])
        self.assertGreater(hard.pose("R", .4)["tip"][2], soft.pose("R", .4)["tip"][2])
        self.assertGreater(hard.pose("R", .5)["pressure"], soft.pose("R", .5)["pressure"])
        for model in (soft, hard):
            impact, release = model.pose("R", .5), model.pose("R", .65)
            self.assertGreater(impact["pressure"], release["pressure"])
            for t in (.1, .4, .5, .55, 1.5):
                p = model.pose("R", t)
                for finger in p["fingers"]:
                    gap = min(segment_distance(a, b, p["butt"], p["tip"])
                              for a, b in zip(finger[:-1], finger[1:]))
                    self.assertAlmostEqual(gap, model.rig.stick_radius+model.rig.finger_radius, places=12)

    def test_fingers_remain_continuous_through_kit_transfer(self):
        model = Drummer([dict(time=.3, drum="hihat", hand="L"), dict(time=.8, drum="tom", hand="L")])
        samples = [model.pose("L", t)["fingers"] for t in np.linspace(.3, .8, 1001)]
        # In particular, no reference-axis switch can rotate the grip abruptly.
        self.assertLess(np.max(np.linalg.norm(np.diff(samples, axis=0), axis=-1)), .002)

    def test_off_grid_contacts_are_included_in_audit(self):
        model = Drummer([dict(time=.4137, drum="snare", hand="R")])
        report = model.audit([0., .2, .4, .6, .8])
        self.assertEqual(report["samples"], 6)
        self.assertEqual(report["hits"], 1)
        self.assertLess(report["max_contact_error_mm"], 1e-8)

    def test_impossible_events_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot make both"):
            Drummer([dict(time=.5, drum="snare", hand="R"), dict(time=.51, drum="ride", hand="R")])
        with self.assertRaisesRegex(ValueError, "No hand available"):
            Drummer([dict(time=.5, drum=d) for d in ("hihat", "snare", "tom")])
        with self.assertRaises(ValueError):
            Drummer([dict(time=.5, drum="kick")])
        with self.assertRaises(ValueError):
            Drummer([dict(time=.5, drum="snare", velocity=128)])

    def test_unreachable_surface_is_a_miss_and_fails_preflight(self):
        far = Drum("far", (0., 2., 0.), .2)
        model = Drummer([dict(time=.5, drum="far", hand="R")], (far,))
        report = model.audit([.4, .5, .6])
        self.assertFalse(report["passed"])
        self.assertEqual(report["hits"], 0)
        self.assertIn("missed_contact", report["violations"])
        self.assertEqual(len(report["missed_events"]), 1)
        self.assertGreater(report["max_reach_error_mm"], 500)
        self.assertLess(report["max_arm_length_error_mm"], 1e-8)

    def test_fast_cross_kit_teleport_fails_speed_gate(self):
        model = Drummer([dict(time=.3, drum="hihat", hand="R"), dict(time=.38, drum="ride", hand="R")])
        report = model.audit(np.linspace(.3, .38, 25))
        self.assertFalse(report["passed"])
        self.assertIn("wrist_speed", report["violations"])
        self.assertGreater(report["max_wrist_speed_m_s"], model.rig.max_wrist_speed)

    def test_unplayed_surface_can_obstruct_a_valid_hit(self):
        blocker = Drum("obstruction", (0., .18, .20), .2)
        model = Drummer([dict(time=.5, drum="snare", hand="R")], DEFAULT_KIT+(blocker,))
        report = model.audit(np.linspace(.3, .7, 21))
        self.assertEqual(report["hits"], 1)
        self.assertFalse(report["passed"])
        self.assertIn("drum_penetration", report["violations"])
        self.assertGreater(report["max_surface_penetration_mm"], 50)

    def test_flat_stick_cannot_hide_fingers_inside_a_surface(self):
        broad = Drum("snare", (0., .18, .10), .45)
        model = Drummer([dict(time=.5, drum="snare", hand="R")], (broad,), Rig(strike_pitch=.001))
        report = model.audit([.49, .5, .51])
        self.assertEqual(report["hits"], 1)
        self.assertIn("finger_drum_penetration", report["violations"])

    def test_demo_contacts_clearances_and_bounds(self):
        model = Drummer(demo_events())
        report = model.audit(np.linspace(0, 2.5, 151))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["hits"], 10)
        self.assertGreaterEqual(report["min_finger_gap_mm"], 5.9)
        self.assertLess(report["max_surface_penetration_mm"], 1e-8)

    def test_dense_roll_and_nearly_equal_frame_event_times(self):
        events = [dict(time=.3071+i*.05, drum="snare", velocity=.6) for i in range(16)]
        model = Drummer(events)
        report = model.audit(np.r_[np.arange(0, 1.5, 1/240), np.nextafter(.3071, 1.)])
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["hits"], 16)
        self.assertEqual(report["violations"], [])

    def test_segment_distance_parallel_crossed_and_degenerate(self):
        self.assertAlmostEqual(segment_distance([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]), 1)
        self.assertAlmostEqual(segment_distance([0, 0, 0], [1, 0, 0], [.5, -1, 0], [.5, 1, 0]), 0)
        self.assertAlmostEqual(segment_distance([0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]), 1)

    def test_tip_collision_cannot_pass_a_shaft_only_clearance_check(self):
        kit = (Drum("a", (-.1, .18, .1), .16), Drum("b", (-.155, .18, .1), .16))
        model = Drummer([dict(time=.5, drum="a", hand="L"),
                         dict(time=.5, drum="b", hand="R")], kit)
        report = model.audit([.5])
        self.assertAlmostEqual(report["min_stick_gap_mm"], -3., places=9)
        self.assertFalse(report["passed"])
        self.assertIn("stick_collision", report["violations"])

    def test_tip_to_opposite_shaft_uses_both_correct_radii(self):
        first = dict(butt=np.array([0., 0., 0.]), tip=np.array([1., 0., 0.]))
        second = dict(butt=np.array([.5, .014, 0.]), tip=np.array([1.5, .014, 0.]))
        # Shafts clear by 2 mm; the first tip enters the second shaft by 1 mm.
        self.assertAlmostEqual(stick_clearance(first, second, Rig()), -.001, places=12)
        self.assertAlmostEqual(stick_clearance(second, first, Rig()), -.001, places=12)

    def test_wider_tip_is_checked_at_an_unplayed_drum_edge(self):
        drum = Drum("snare", (0., .18, .1), .16)
        events = [dict(time=.5, drum="snare", hand="R")]
        pose = Drummer(events, (drum,)).pose("R", .5)
        lateral = np.cross(pose["tip"]-pose["butt"], [0., 0., 1.])
        lateral /= np.linalg.norm(lateral)
        # The tip is 8 mm outside the disk: its 9 mm sphere overlaps, whereas
        # the 6 mm shaft cannot. No unplayed rim may be excluded by shaft size.
        blocker = Drum("rim", tuple(pose["tip"]+.038*lateral), .03)
        report = Drummer(events, (drum, blocker)).audit([.5])
        self.assertIn("drum_penetration", report["violations"])

    def test_drum_directly_below_shoulder_has_a_finite_contact_pose(self):
        drum = Drum("under", (.145, -.2, .1), .16)
        model = Drummer([dict(time=.5, drum="under", hand="R")], (drum,))
        p = model.pose("R", .5)
        self.assertTrue(np.all(np.isfinite(p["fingers"])))
        self.assertAlmostEqual(np.dot(p["tip"]-drum.center, drum.normal), model.rig.tip_radius, places=12)

    def test_opposite_strike_directions_have_a_continuous_midpoint(self):
        kit = (Drum("front", (.145, -.1, .1), .16), Drum("back", (.145, -.3, .1), .16))
        model = Drummer([dict(time=.3, drum="front", hand="R"),
                         dict(time=.8, drum="back", hand="R")], kit)
        poses = [model.pose("R", t) for t in (.55-1e-6, .55, .55+1e-6)]
        fingers = np.array([p["fingers"] for p in poses])
        self.assertTrue(np.all(np.isfinite(fingers)))
        self.assertLess(np.max(np.linalg.norm(np.diff(fingers, axis=0), axis=-1)), .0001)
        for p in poses:
            self.assertAlmostEqual(np.linalg.norm(p["tip"]-p["butt"]), model.rig.stick_length, places=12)

    def test_invalid_geometry_is_rejected(self):
        for normal in ((0, 0), (0, 0, 0), (0, 0, -1), (float("nan"), 0, 1)):
            with self.assertRaises(ValueError):
                Drum("bad", (0, 0, 0), .2, normal)
        for values in (dict(stick_length=float("nan")), dict(restitution=2), dict(elbow_min_deg=-5)):
            with self.assertRaises(ValueError):
                Rig(**values)
        with self.assertRaises(ValueError):
            Drummer([]).audit([])


if __name__ == "__main__":
    unittest.main()

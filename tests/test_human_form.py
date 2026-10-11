import unittest

import numpy as np

from blockout.human_score import measures
from blockout.rig.calibrate import pose_points
from blockout.rig.hand import HandRig, rotvec
from blockout.rig.human_form import PianistFormPrior


class HumanFormPriorTests(unittest.TestCase):
    def test_landmark_measurement_is_invariant_to_world_frame_and_units(self):
        prior, rig = PianistFormPrior(), HandRig('R')
        points = pose_points(rig, rig.q_rest)
        activity = [.8, 0., .3, .8, 1.]
        first = prior.evaluate(points, 'R', activity)
        rotated = points @ rotvec(np.array([.2, -.4, .5])).T
        second = prior.evaluate(3. * rotated + [2., -.7, 4.], 'R', activity)
        self.assertAlmostEqual(first['cost'], second['cost'], places=10)
        np.testing.assert_allclose(first['angles_deg'], second['angles_deg'], atol=1e-10)
        expected = measures(points, 'R')
        np.testing.assert_allclose(first['angles_deg'], np.array([expected[k] for k in ('mcp_flex', 'pip_flex', 'dip_flex')]).T)

    def test_phase_guidance_is_continuous_and_does_not_change_anatomy(self):
        prior, rig = PianistFormPrior(), HandRig('L')
        before = rig.lo.copy(), rig.hi.copy()
        points = pose_points(rig, rig.q_rest)
        costs = [prior.evaluate(points, 'L', [a] * 5)['cost'] for a in (.5-1e-6, .5, .5+1e-6)]
        self.assertLess(max(np.abs(np.diff(costs))), 1e-4)
        np.testing.assert_array_equal(rig.lo, before[0])
        np.testing.assert_array_equal(rig.hi, before[1])
        # The FürElise-calibrated comfort pose is already inside its free
        # reference bands. Overcurl must receive guidance without widening
        # the same anatomical limits.
        self.assertEqual(prior.evaluate(points, 'L', [0.] * 5)['cost'], 0.)
        curled = pose_points(rig, np.clip(rig.q_rest + .6, rig.lo, rig.hi))
        self.assertGreater(prior.evaluate(curled, 'L', [0.] * 5)['cost'], .1)

    def test_invalid_or_degenerate_landmarks_are_rejected(self):
        prior = PianistFormPrior()
        for points, activity in ((np.zeros((21, 3)), [0.] * 5),
                                 (np.full((21, 3), float('nan')), [0.] * 5),
                                 (np.ones((20, 3)), [0.] * 5),
                                 (np.ones((21, 3)), [1.1] * 5)):
            with self.assertRaises(ValueError):
                prior.evaluate(points, 'R', activity)

import copy
import unittest

from blockout.rig.path_validation import MOTION_LIMITS, sampled_path_violations


class SampledPathGateTests(unittest.TestCase):
    def setUp(self):
        hand = dict(instrument_clear=True, proxy_self_clearance_mm=.2,
                    joint_bounds_excess_rad=0., arm_length_error_mm=0., wrist_root_error_mm=0.)
        self.observations = [dict(node=0, hands={'R': copy.deepcopy(hand)}, unsupported_pitches=[],
                                  key_travel_excess=0., mechanical_contacts=[]),
                             dict(node=1, hands={'R': copy.deepcopy(hand)}, unsupported_pitches=[],
                                  key_travel_excess=0., mechanical_contacts=[])]
        self.observations[1]['hands']['R']['motion'] = {name: bound * .5 for name, bound in MOTION_LIMITS.items()}

    def test_clear_skin_cannot_hide_lost_support_spacing_or_stretched_arm(self):
        self.assertEqual(sampled_path_violations(self.observations), [])
        cases = [('unsupported_pitches', [60]), ('mechanical_contacts', [{'first': 'key60', 'second': 'key61'}]),
                 ('key_travel_excess', .01)]
        for name, value in cases:
            frames = copy.deepcopy(self.observations)
            frames[1][name] = value
            self.assertTrue(sampled_path_violations(frames), name)
        for name, value in [('proxy_self_clearance_mm', -.1), ('arm_length_error_mm', 2.),
                            ('joint_bounds_excess_rad', .01), ('wrist_root_error_mm', 1.)]:
            frames = copy.deepcopy(self.observations)
            frames[1]['hands']['R'][name] = value
            self.assertTrue(sampled_path_violations(frames), name)

    def test_nonfinite_missing_or_fast_motion_is_rejected(self):
        for name, bound in MOTION_LIMITS.items():
            for value in (bound + .1, float('nan'), float('inf'), -1.):
                frames = copy.deepcopy(self.observations)
                frames[1]['hands']['R']['motion'][name] = value
                self.assertTrue(sampled_path_violations(frames))
        del self.observations[1]['hands']['R']['motion']
        self.assertTrue(sampled_path_violations(self.observations))
        self.assertTrue(sampled_path_violations([]))

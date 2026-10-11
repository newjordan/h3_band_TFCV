"""Regression for confusing a base cage with its projected limit surface."""
import unittest

import numpy as np

from blockout.skin import ASSET
from blockout.subdivision import catmull_clark
from lab.skin_deformation_audit import expected_geometry


class DeformationAuditTests(unittest.TestCase):
    def test_base_cage_control_preserves_the_actual_unsubdivided_asset(self):
        record = dict(context='bind_control', subdivision_levels=0)
        points, faces, sides, _regions, _fields = expected_geometry(record, 1., None)
        with np.load(ASSET, allow_pickle=False) as data:
            np.testing.assert_array_equal(points, data['vertices'])
            np.testing.assert_array_equal(faces, data['face_vertices'].reshape(-1, 4))
            np.testing.assert_array_equal(sides, data['sides'])
        # A projected limit cage is measurably different at the real knuckles.
        # It cannot validate or label the sharp base cage by nearest positions.
        operator, _ = catmull_clark(len(points), faces, 0, limit=True)
        self.assertGreater(np.linalg.norm(operator @ points - points, axis=1).max(), .001)


if __name__ == '__main__':
    unittest.main()

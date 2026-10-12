import json
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from blockout.mesh_self import self_intersections, quad_self_intersections
from blockout.rig.skin_pair_constraints import SkinPairConstraints
from blockout.triangle_contact import triangle_separation


class SkinPairWitnessTests(unittest.TestCase):
    def test_complete_optimizer_also_receives_mixed_face_crossings(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/skin_fold_mixed_diagonal.json').read_text())
        vertices, faces = np.asarray(fixture['vertices']), np.asarray(fixture['faces'])
        old = SkinPairConstraints(sparse.eye(len(vertices), format='csr'), faces, 1e-5, 'both_interior')
        new = SkinPairConstraints(sparse.eye(len(vertices), format='csr'), faces, 1e-5, 'complete_interior')
        self.assertFalse(old.update(vertices)['surface_intersection_clear'])
        self.assertFalse(new.update(vertices)['surface_intersection_clear'])
        self.assertEqual(len(old.pairs), 0)
        self.assertGreater(len(new.pairs), 0)
        self.assertTrue(all(key[0] == 2 for key in new.pairs))
        new.build()
        self.assertLess(new.evaluate(vertices).min(), 0.)

    def test_actual_shared_vertex_fold_gets_a_descent_witness(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/skin_fold_shared_vertex.json').read_text())
        vertices = np.asarray(fixture['vertices'])
        # Complete the actual crossing triangles into two diagnostic quads.
        vertices = np.vstack([vertices, vertices[0] + [0., 0., .001], vertices[4] + [0., .001, 0.]])
        faces = np.array([[0, 1, 2, 5], [1, 3, 4, 6]])
        witnesses = SkinPairConstraints(sparse.eye(len(vertices), format='csr'), faces, 1e-5, 'both_interior')
        audit = witnesses.update(vertices)
        witnesses.build()
        key = (0, 0, 1)
        self.assertIn(key, witnesses.incident)
        self.assertLess(witnesses.evaluate(vertices)[sorted(witnesses.pairs).index(key)], 0.)
        self.assertFalse(audit['surface_intersection_clear'])
        self.assertFalse(witnesses.summary()['clearance_certificate'])

    def test_expected_shared_topology_is_not_forced_apart(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                             [2., 0., 0.], [2., 1., 0.]])
        faces = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
        witnesses = SkinPairConstraints(sparse.eye(6, format='csr'), faces, 1e-5, 'both_interior')
        self.assertTrue(witnesses.update(vertices)['surface_intersection_clear'])
        witnesses.build()
        self.assertEqual(len(witnesses.evaluate(vertices)), 0)
        self.assertEqual(witnesses.summary()['discovered_pairs'], 0)

    def test_alternative_diagonals_are_never_mixed_into_fake_self_contacts(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2, 3]])
        witnesses = SkinPairConstraints(sparse.eye(4, format='csr'), faces, 1e-5, 'both_interior')
        self.assertTrue(witnesses.update(vertices)['surface_intersection_clear'])
        self.assertEqual(len(witnesses.pairs), 0)
        # Mixing the alternatives would introduce positive-area overlaps.
        mixed = np.concatenate(witnesses.triangles)
        self.assertGreater(self_intersections(vertices, mixed)['unexpected_surface_intersections'], 0)

    def test_legacy_mode_keeps_incident_fold_out_of_the_optimizer_only(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
                             [1., 0., -1.], [0., 1., 1.], [-1., 1., 1.]])
        faces = np.array([[0, 1, 2, 3], [0, 4, 5, 6]])
        witnesses = SkinPairConstraints(sparse.eye(7, format='csr'), faces, 1e-5)
        self.assertFalse(witnesses.update(vertices)['surface_intersection_clear'])
        self.assertNotIn((0, 0, 1), witnesses.pairs)
        self.assertFalse(quad_self_intersections(vertices, faces)['surface_intersection_clear'])

    def test_interior_clearance_cannot_certify_a_fold_near_the_shared_vertex(self):
        # A short real crossing near a shared vertex disappears after inset.
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[0., 0., 0.], [.002, .002, -.002], [.002, .002, .002]])
        full = np.vstack([a, b[1:]])
        self.assertFalse(self_intersections(full, np.array([[0, 1, 2], [0, 3, 4]]))['surface_intersection_clear'])
        inset = np.eye(3) * .97 + .01
        inner = np.array([inset @ a, inset @ b])
        self.assertGreater(triangle_separation(inner[:1], inner[1:])[0], 0.)

    def test_invalid_witness_modes_or_insets_are_rejected(self):
        operator, faces = sparse.eye(4, format='csr'), np.array([[0, 1, 2, 3]])
        for mode, inset in [('unknown', .01), ('both_interior', 0.), ('both_interior', 1. / 3.),
                            ('both_interior', float('nan'))]:
            with self.assertRaises(ValueError):
                SkinPairConstraints(operator, faces, 1e-5, mode, inset)


if __name__ == '__main__':
    unittest.main()

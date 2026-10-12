import unittest
import json
from pathlib import Path

import numpy as np

from blockout.mesh_self import candidate_pairs, self_intersections, quad_self_intersections
from blockout.triangle_contact import triangle_overlap


class MeshSelfTests(unittest.TestCase):
    def test_mixed_choices_on_different_quads_catch_a_real_blender_fold(self):
        data = json.loads((Path(__file__).parent / 'fixtures/skin_fold_mixed_diagonal.json').read_text())
        report = quad_self_intersections(np.asarray(data['vertices']), np.asarray(data['faces']))
        self.assertTrue(all(r['surface_intersection_clear'] for r in report['triangulations']))
        self.assertGreater(report['cross_diagonal']['unexpected_surface_intersections'], 0)
        self.assertFalse(report['surface_intersection_clear'])

    def test_complete_quad_veto_does_not_depend_on_cyclic_corner_order(self):
        data = json.loads((Path(__file__).parent / 'fixtures/skin_fold_mixed_diagonal.json').read_text())
        vertices, faces = np.asarray(data['vertices']), np.asarray(data['faces'])
        first = quad_self_intersections(vertices, faces)
        faces[1] = np.roll(faces[1], 1)
        second = quad_self_intersections(vertices, faces)
        self.assertFalse(first['surface_intersection_clear'])
        self.assertFalse(second['surface_intersection_clear'])
        self.assertEqual(first['possible_triangulation_crossing_pairs'], second['possible_triangulation_crossing_pairs'])

    def test_complete_quad_check_excludes_only_incompatible_same_face_alternatives(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                             [2., 0., 0.], [2., 1., 0.]])
        faces = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
        report = quad_self_intersections(vertices, faces)
        self.assertTrue(report['surface_intersection_clear'])
        self.assertEqual(report['cross_diagonal']['unexpected_surface_intersections'], 0)

    def test_incompatible_alternative_exclusion_does_not_hide_a_bow_tie_quad(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        report = quad_self_intersections(vertices, np.array([[0, 2, 1, 3]]))
        self.assertFalse(report['surface_intersection_clear'])
        self.assertTrue(any(not r['surface_intersection_clear'] for r in report['triangulations']))

    def test_variable_triangle_sizes_keep_every_exact_crossing_candidate(self):
        rng = np.random.default_rng(4)
        vertices = np.concatenate([rng.normal(size=(3, 3)) * size for size in np.geomspace(.001, 1., 18)])
        triangles = np.arange(len(vertices)).reshape(-1, 3)
        candidates = {tuple(pair) for pair in candidate_pairs(vertices, triangles)}
        geometry = vertices[triangles]
        for a in range(len(triangles)):
            for b in range(a+1, len(triangles)):
                if triangle_overlap(geometry[a:a+1], geometry[b:b+1])[0]:
                    self.assertIn((a, b), candidates)

    def test_shared_vertex_crossing_is_not_discarded_by_the_mesh_gate(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 0., -1.], [0., 1., 1.]])
        report = self_intersections(vertices, np.array([[0, 1, 2], [0, 3, 4]]))
        self.assertFalse(report['surface_intersection_clear'])
        self.assertEqual(report['unexpected_surface_intersections'], 1)

    def test_expected_edge_and_vertex_contacts_remain_clear(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.3, -.4, .1], [-1., 0., 1.], [0., -1., 1.]])
        report = self_intersections(vertices, np.array([[0, 1, 2], [0, 1, 3], [0, 4, 5]]))
        self.assertTrue(report['surface_intersection_clear'])
        self.assertEqual(report['unexpected_surface_intersections'], 0)

    def test_degenerate_faces_are_rejected_even_without_a_crossing(self):
        report = self_intersections(np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), np.array([[0, 1, 2]]))
        self.assertFalse(report['surface_intersection_clear'])
        self.assertEqual(report['degenerate_triangles'], 1)

    def test_actual_shared_vertex_fold_witness_is_inside_both_faces(self):
        data = json.loads((Path(__file__).parent / 'fixtures/skin_fold_shared_vertex.json').read_text())
        vertices, triangles = np.asarray(data['vertices']), np.asarray(data['triangles'])
        report = self_intersections(vertices, triangles)
        self.assertEqual(report['unexpected_surface_intersections'], 1)
        witness = report['witnesses'][0]
        self.assertAlmostEqual(witness['intersection_length_mm'], data['expected_length_mm'], places=8)
        for triangle in vertices[triangles]:
            basis = np.column_stack([triangle[1]-triangle[0], triangle[2]-triangle[0]])
            point = np.asarray(witness['witness']) - triangle[0]
            barycentric, *_ = np.linalg.lstsq(basis, point, rcond=None)
            self.assertLess(np.linalg.norm(basis @ barycentric - point), 1e-12)
            self.assertGreater(min(*barycentric, 1.-sum(barycentric)), .01)

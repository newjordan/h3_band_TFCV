"""Optimization witnesses for unexpected skin crossings, not a clearance gate.

Connected faces must keep their shared topology. Positive separation of their
full triangles is therefore impossible. Small interior patches provide a
descent witness without separating that topology. They can miss folds near a
shared feature; acceptance must still audit the complete untrimmed surface.
Mixed diagonal choices on different faces can coexist. Alternative triangles
of the same face cannot coexist and must not generate artificial overlaps.
"""
import numpy as np
from scipy import sparse

from ..mesh_self import self_intersections, quad_self_intersections
from ..triangle_contact import triangle_separation


class SkinPairConstraints:
    def __init__(self, operator, faces, gap, mode='legacy', inset=.01):
        if mode not in ('legacy', 'both_interior', 'complete_interior'):
            raise ValueError('Unknown skin pair constraint mode')
        if not np.isfinite(inset) or not 0 < inset < 1. / 3.:
            raise ValueError('Interior barycentric inset must lie between zero and one third')
        if not np.isfinite(gap) or gap < 0:
            raise ValueError('Skin witness gap must be finite and nonnegative')
        self.operator, self.mode, self.inset, self.gap = operator, mode, inset, gap
        self.faces = np.asarray(faces)
        self.triangles = [np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])]
        if mode != 'legacy':
            self.triangles.append(np.concatenate([faces[:, [1, 2, 3]], faces[:, [1, 3, 0]]]))
        if mode == 'complete_interior':
            self.triangles.append(np.concatenate(self.triangles))
        self.pairs, self.incident, self.margins = set(), set(), {}
        self.stencil, self.slacks = None, np.empty(0)
        self.constrained_pairs = 0

    def update(self, vertices):
        if self.mode == 'legacy':
            audit = self_intersections(vertices, self.triangles[0])
            audits = [audit]
        else:
            audit = quad_self_intersections(vertices, self.faces)
            audits = audit['triangulations']
            if self.mode == 'complete_interior':
                audits = audits + [audit['cross_diagonal']]
        for diagonal, result in enumerate(audits):
            triangles = self.triangles[diagonal]
            for a, b in result['pairs']:
                key = (diagonal, int(a), int(b))
                incident = bool(set(triangles[a]) & set(triangles[b]))
                if incident and self.mode == 'legacy':
                    continue
                if key in self.pairs:
                    continue
                self.pairs.add(key)
                self.margins[key] = self.gap
                if incident:
                    self.incident.add(key)
                    geometry = vertices[triangles[[a, b]]]
                    edges = np.roll(geometry, -1, axis=1) - geometry
                    twice_area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
                    altitude = twice_area / np.maximum(np.linalg.norm(edges, axis=2).max(1), 1e-30)
                    # A conditioning margin frozen at witness discovery, not
                    # a smaller threshold for accepting the original faces.
                    self.margins[key] = min(self.gap, .25 * self.inset * float(altitude.min()))
        return audit

    def build(self):
        stencils, margins = [], []
        interior = sparse.csr_matrix(np.eye(3) * (1. - 3. * self.inset) + self.inset)
        for key in sorted(self.pairs):
            diagonal, a, b = key
            triangles = self.triangles[diagonal][[a, b]]
            stencil = self.operator[triangles.ravel()]
            if key in self.incident:
                stencil = sparse.block_diag([interior, interior], format='csr') @ stencil
            stencils.append(stencil)
            margins.append(self.margins[key])
        self.stencil = sparse.vstack(stencils, format='csr') if stencils else None
        self.slacks = np.asarray(margins)
        self.constrained_pairs = len(stencils)

    def evaluate(self, base):
        if self.stencil is None:
            return np.empty(0)
        paired = (self.stencil @ base).reshape(-1, 2, 3, 3)
        return triangle_separation(paired[:, 0], paired[:, 1]) - self.slacks

    def summary(self):
        margins = [self.margins[key] * 1000. for key in self.incident]
        return dict(mode=self.mode, triangulations=1 if self.mode == 'legacy' else 2,
            mixed_choices_on_different_faces=self.mode == 'complete_interior', discovered_pairs=len(self.pairs),
            constrained_pairs=self.constrained_pairs, incident_interior_pairs=len(self.incident),
            nonincident_full_pairs=len(self.pairs) - len(self.incident),
            barycentric_inset=self.inset if self.mode != 'legacy' else None,
            incident_margin_minimum_mm=min(margins) if margins else None,
            incident_margin_maximum_mm=max(margins) if margins else None,
            clearance_certificate=False,
            scope='Optimization witnesses only; full untrimmed both-diagonal audit remains mandatory')

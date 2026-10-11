"""Soft pianist form prior measured on reconstructed joint landmarks.

The reference is pooled markerless MANO motion, not a musculoskeletal model.
Its phase-conditioned IQRs guide fingers 2–5; they never replace anatomy,
instrument clearance, held-note support or temporal bounds. Thumb shape and
expert visual acceptance remain separate requirements.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..human_score import measures


REFERENCE = Path(__file__).with_name('pianist_reference.json')
KINDS = ('mcp_flex', 'pip_flex', 'dip_flex')


class PianistFormPrior:
    def __init__(self, reference=REFERENCE, minimum_sigma_deg=8., free_weight=1., press_weight=.35):
        if not all(np.isfinite(v) and v > 0 for v in (minimum_sigma_deg, free_weight, press_weight)):
            raise ValueError('Finite positive form scales are required')
        self.reference = Path(reference)
        self.source_sha256 = hashlib.sha256(self.reference.read_bytes()).hexdigest()
        data = json.loads(self.reference.read_text())
        self.phase_bands = {}
        for phase in ('free', 'pressing'):
            bands = np.array([[data['phase_bands'][phase][kind][p] for kind in KINDS]
                              for p in ('p25', 'p50', 'p75')])
            if not np.isfinite(bands).all() or np.any(np.diff(bands, axis=0) < 0):
                raise ValueError('Invalid human reference bands')
            self.phase_bands[phase] = bands
        self.minimum_sigma_deg = minimum_sigma_deg
        self.free_weight, self.press_weight = free_weight, press_weight

    def evaluate(self, landmarks, side, activation):
        points, active = np.asarray(landmarks, float), np.asarray(activation, float)
        if (points.shape != (21, 3) or active.shape != (5,) or side not in ('L', 'R')
                or not np.isfinite(points).all() or not np.isfinite(active).all()
                or np.any(active < 0) or np.any(active > 1)):
            raise ValueError('Finite 21-point hand and five bounded activations are required')
        for finger in range(5):
            if np.any(np.linalg.norm(np.diff(points[1 + 4*finger:5 + 4*finger], axis=0), axis=1) < 1e-10):
                raise ValueError('Degenerate finger landmarks')
        across, forward = points[17] - points[5], points[[5, 9, 13, 17]].mean(0) - points[0]
        if np.linalg.norm(np.cross(across, forward)) < 1e-12:
            raise ValueError('Degenerate palm landmarks')
        measured = measures(points, side)
        angles = np.array([measured[kind] for kind in KINDS]).T
        if not np.isfinite(angles).all():
            raise ValueError('Undefined anatomical angle measurement')
        # Activation changes the guidance continuously, without declaring
        # dimensionless contact stiffness to be measured muscle activation.
        phase = active[1:] ** 2 * (3. - 2. * active[1:])
        free, pressing = self.phase_bands['free'], self.phase_bands['pressing']
        bands = (1. - phase[:, None, None]) * free[None] + phase[:, None, None] * pressing[None]
        lower, median, upper = bands[:, 0], bands[:, 1], bands[:, 2]
        sigma = np.maximum(self.minimum_sigma_deg, (upper - lower) / 1.349)
        excess = angles - np.clip(angles, lower, upper)
        residual = excess / sigma
        weight = (1. - phase) * self.free_weight + phase * self.press_weight
        # Smooth robust loss limits the effect of uncertain/outlying capture
        # statistics. Hard geometric and mechanical gates remain unchanged.
        cost = float(np.sum(weight[:, None] * 2. * (np.sqrt(1. + residual ** 2) - 1.)))
        return dict(cost=cost, angles_deg=angles.tolist(), target_iqr_deg=np.stack([lower, upper], axis=1).tolist(),
                    target_median_deg=median.tolist(), outside_iqr=int(np.count_nonzero(excess)),
                    maximum_iqr_excess_deg=float(np.max(np.abs(excess))),
                    source_sha256=self.source_sha256, scope='soft pooled phase prior for fingers 2–5; no realism acceptance')

"""Absolute wrist history and rotation smoothing on SO(3).

These are kinematic tools, not anatomical or collision guarantees. Smoothing
uses spatial angular increments in one world frame, rather than offsets from
different per-sample reference rotations. Its weights are engineering choices.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def checked_rotations(value):
    matrices = np.asarray(value, float)
    if (matrices.ndim < 2 or matrices.shape[-2:] != (3, 3)
            or not np.isfinite(matrices).all()
            or not np.allclose(matrices @ np.swapaxes(matrices, -1, -2), np.eye(3), atol=1e-7, rtol=0)
            or not np.allclose(np.linalg.det(matrices), 1., atol=1e-7, rtol=0)):
        raise ValueError('Expected finite proper rotation matrices')
    return matrices


def split_reference_rotations(value):
    """Separate exported frame precision from rotation, retaining its metric.

    The source wrist rest matrices have about 1e-6 orthogonality error. Polar
    decomposition exposes this small fixed right stretch explicitly. Reject
    reflections or larger affine deformation; callers reconstruct Q @ stretch
    rather than silently projecting the actual FK/skin matrices onto SO(3).
    """
    matrices = np.asarray(value, float)
    if (matrices.ndim < 2 or matrices.shape[-2:] != (3, 3)
            or not np.isfinite(matrices).all()
            or np.any(np.linalg.det(matrices) <= 0)
            or np.max(np.abs(matrices @ np.swapaxes(matrices, -1, -2) - np.eye(3))) > 1e-5):
        raise ValueError('Expected a proper near-rotation within exported reference precision')
    u, singular, vt = np.linalg.svd(matrices)
    proper = u @ vt
    stretch = np.swapaxes(vt, -1, -2) @ (singular[..., :, None] * vt)
    checked_rotations(proper)
    return proper, stretch


def _fixed_reference(value, expected_stretch):
    proper, stretch = split_reference_rotations(value)
    if expected_stretch is not None and not np.allclose(stretch, expected_stretch, atol=1e-10, rtol=0):
        raise ValueError('Physical history and current reference must share the fixed wrist metric')
    return proper


def absolute_pose(vector, rotation, fixed_stretch=None):
    """Encode a physical predecessor independently of its former reference."""
    vector = np.asarray(vector, float)
    if vector.ndim != 1 or len(vector) < 6 or not np.isfinite(vector).all():
        raise ValueError('Expected a finite hand pose vector')
    result = vector.copy()
    result[3:6] = Rotation.from_matrix(_fixed_reference(rotation, fixed_stretch)).as_rotvec()
    return result


def relative_pose(physical, reference, fixed_stretch=None):
    """Transport absolute wrist history into the current solver coordinates."""
    physical = np.asarray(physical, float)
    if physical.ndim != 1 or len(physical) < 6 or not np.isfinite(physical).all():
        raise ValueError('Expected a finite absolute hand pose vector')
    reference = _fixed_reference(reference, fixed_stretch)
    result = physical.copy()
    result[3:6] = (Rotation.from_rotvec(physical[3:6])
                   * Rotation.from_matrix(reference).inv()).as_rotvec()
    return result


def forearm_angles(direction, across, forward, up):
    """Signed direction chart: backward-facing hands cannot alias to zero.

    Flexion is atan2(up, forward); deviation is the elevation towards across.
    This preserves the forward/backward distinction, but is not a clinical
    wrist model or a hard orientation bound.
    """
    vectors = np.asarray([direction, across, forward, up], float)
    if not np.isfinite(vectors).all() or np.linalg.norm(vectors[0]) < 1e-12:
        raise ValueError('A finite nonzero forearm direction is required')
    basis = vectors[1:]
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Hand directions must be orthonormal within exported reference precision')
    # A chart-only orthonormal basis. Actual FK/skin frames keep their small
    # fixed reference stretch; this does not change geometry or bone lengths.
    left, _, right = np.linalg.svd(basis)
    basis = left @ right
    a, f, u = basis @ (vectors[0] / np.linalg.norm(vectors[0]))
    return float(np.arctan2(u, f)), float(np.arctan2(a, np.hypot(u, f)))


def smooth_rotations(rotations, weights, penalty, max_evaluations=30):
    """Fit weighted absolute rotations with a spatial-increment roughness cost.

    Minimize sum w_i |Log(Q_i A_i^-1)|^2 plus sum lambda_i
    |Log(Q_(i+1) Q_i^-1) - Log(Q_i Q_(i-1)^-1)|^2. Both increments are
    spatial/world vectors. The second term is a discrete angular-acceleration
    descriptor at a fixed timestep, not a hard velocity/acceleration limit.
    No global optimum or continuous path feasibility is claimed.
    """
    matrices = np.asarray(rotations, float)
    proper, stretch = split_reference_rotations(matrices)
    if matrices.ndim != 3 or not len(matrices):
        raise ValueError('A nonempty rotation trajectory is required')
    n = len(matrices)
    if not np.allclose(stretch, stretch[0], atol=1e-10, rtol=0):
        raise ValueError('A rotation trajectory must retain one fixed exported reference metric')
    w = np.broadcast_to(np.asarray(weights, float), (n,)).copy()
    lam = np.broadcast_to(np.asarray(penalty, float), (max(0, n - 2),)).copy()
    if (not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0)
            or not np.isfinite(lam).all() or np.any(lam < 0)
            or not isinstance(max_evaluations, int) or max_evaluations < 1):
        raise ValueError('Finite nonnegative smoothing weights and a positive budget are required')
    reference = Rotation.from_matrix(proper)
    def residual(value):
        track = Rotation.from_rotvec(value.reshape(n, 3)) * reference
        anchor = (track * reference.inv()).as_rotvec() * np.sqrt(w)[:, None]
        increment = (track[1:] * track[:-1].inv()).as_rotvec()
        roughness = np.diff(increment, axis=0) * np.sqrt(lam)[:, None]
        return np.r_[anchor.ravel(), roughness.ravel()]
    seed = np.zeros(3 * n)
    before = float(residual(seed) @ residual(seed))
    if n < 3 or not np.any(lam):
        return matrices.copy(), dict(changed=False, optimizer_success=True,
            evaluations=0, initial_cost=before, final_cost=before, hard_motion_bounds=False)
    pattern = sparse.lil_matrix((3 * n + 3 * (n - 2), 3 * n), dtype=int)
    for i in range(n):
        pattern[3*i:3*i+3, 3*i:3*i+3] = 1
    for i in range(n - 2):
        pattern[3*n+3*i:3*n+3*i+3, 3*i:3*i+9] = 1
    result = least_squares(residual, seed, jac_sparsity=pattern.tocsr(),
        max_nfev=max_evaluations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    cost = float(result.fun @ result.fun)
    keep = bool(np.isfinite(result.x).all() and np.isfinite(cost) and cost <= before + 1e-10)
    output = ((Rotation.from_rotvec(result.x.reshape(n, 3)) * reference).as_matrix() @ stretch
              if keep else matrices.copy())
    _, output_metric = split_reference_rotations(output)
    np.testing.assert_allclose(output_metric, stretch, atol=1e-10, rtol=0)
    return output, dict(changed=bool(keep and not np.array_equal(output, matrices)),
        optimizer_success=bool(result.success), evaluations=int(result.nfev),
        initial_cost=before, final_cost=cost if keep else before,
        numerical_iterate_kept=keep, hard_motion_bounds=False,
        fixed_reference_stretch_retained=True,
        source_orthogonality_maximum_error=float(np.max(np.abs(matrices @ np.swapaxes(matrices,-1,-2)-np.eye(3)))),
        scope='absolute rotations and discrete spatial increments; no anatomical/geometry/force guarantee')

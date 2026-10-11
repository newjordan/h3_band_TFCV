"""Exact vertex penetration for finite rigid boxes, independent of Blender.

Positive values are the distance to the nearest escape face, in metres. This
tests a vertex against a solid box; it is not a mesh/mesh intersection proof.
"""
import numpy as np


def box_signed_depth(points, world_matrix, lower, upper):
    """Nearest-face depth in a box's local frame, preserving whole-box rotation.

    ``world_matrix`` must be a rigid local-to-world affine transform. Bounds are
    the actual mesh bounds, so physical key gaps are retained. A nonpositive
    result means outside or touching; negative magnitudes are not Euclidean
    distances outside corners.
    """
    points = np.asarray(points, float)
    transform = np.asarray(world_matrix, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4)
            or lower.shape != (3,) or upper.shape != (3,)):
        raise ValueError("Expected Nx3 points, a 4x4 transform, and 3D box bounds")
    if not all(np.isfinite(x).all() for x in (points, transform, lower, upper)):
        raise ValueError("Box geometry must be finite")
    if np.any(upper <= lower):
        raise ValueError("Box bounds must enclose positive volume")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("Box transform must be rigid; scaled distances need separate handling")
    local = (points-transform[:3, 3]) @ rotation
    return np.minimum(local-lower, upper-local).min(axis=1)

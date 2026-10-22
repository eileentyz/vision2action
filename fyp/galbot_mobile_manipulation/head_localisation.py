"""Coarse metric basket position from head stereo RGB detections; no ROS dependencies.

The head pair is rectified (parallel optical axes, right camera displaced along
left optical -X by the baseline, identical intrinsics). Triangulating the YOLO
mask centroids from both images gives a coarse 3D prior for choosing a wrist
viewpoint. It is never a grasp target: silhouette centroids from two
viewpoints are not exact correspondences. The wrist RGB-D pose remains the
only metric basket pose used downstream.

A single-camera ray/support-plane intersection is deliberately not used: the
head optical centre is only a few centimetres above the basket centre in the
workcell, so the ray grazes the plane and pixel noise becomes large range error.
"""
import math
import numpy as np


def mask_centroid(mask):
    """(u, v) centroid of a nonempty binary mask; None when empty.

    Raises when the mask touches the image border: a clipped silhouette has a
    biased centroid, so the detection is not usable for ranging.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError('Mask must be a single-channel image')
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    if rows.min() == 0 or cols.min() == 0 or rows.max() == mask.shape[0]-1 or cols.max() == mask.shape[1]-1:
        raise ValueError('Head detection is clipped by the image border')
    return float(cols.mean()), float(rows.mean())


def stereo_baseline(left_p, right_p):
    """Baseline (m) from rectified ROS projection matrices: P_right[3] = -fx*B."""
    left_p, right_p = np.asarray(left_p, float).reshape(3, 4), np.asarray(right_p, float).reshape(3, 4)
    if (not np.isfinite(np.r_[left_p.ravel(), right_p.ravel()]).all() or left_p[0, 0] <= 0
            or not np.allclose(left_p[:, :3], right_p[:, :3], atol=1e-6) or abs(left_p[0, 3]) > 1e-9):
        raise ValueError('Head cameras are not a rectified pair with shared intrinsics')
    baseline = -right_p[0, 3] / right_p[0, 0]
    if not 0 < baseline <= 0.5:
        raise ValueError('Invalid head stereo baseline')
    return baseline


def triangulate(left_px, right_px, k, baseline, min_disparity=2.0, max_row_error=6.0):
    """Point in the left optical frame from matched rectified pixels."""
    (ul, vl), (ur, vr) = left_px, right_px
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.isfinite([ul, vl, ur, vr, baseline]).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid stereo pixels, intrinsics or baseline')
    if abs(vl - vr) > max_row_error:
        raise ValueError(f'Left/right detections disagree vertically ({abs(vl - vr):.1f} px)')
    disparity = ul - ur
    if disparity < min_disparity:
        raise ValueError(f'Stereo disparity {disparity:.1f} px is too small for ranging')
    z = k[0, 0] * baseline / disparity
    v = (vl + vr) / 2
    return np.array([(ul - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])


def coarse_position(points, max_spread=0.03):
    """Median of consecutive base-frame triangulations, rejecting inconsistent sets."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Coarse localisation needs finite triangulated head points')
    median = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - median, axis=1)))
    if spread > max_spread:
        raise ValueError(f'Head stereo estimates are inconsistent ({1000 * spread:.0f} mm spread)')
    return median, dict(readings=len(points), spread_m=spread)


def check_workspace(point, low, high):
    """Raise unless the coarse estimate lies inside the arm observation region."""
    point, low, high = (np.asarray(x, dtype=float) for x in (point, low, high))
    if point.shape != (3,) or low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
        raise ValueError('Invalid workspace bounds')
    if np.any(point < low) or np.any(point > high):
        raise ValueError('Detected basket at ' + str(np.round(point, 3).tolist()) +
                         ' is outside the arm observation workspace; base repositioning required')
    return point

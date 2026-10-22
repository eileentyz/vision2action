"""Experimental head-stereo yaw of an upright box by silhouette fitting; no ROS dependencies.

The head views the block obliquely (about 40 degrees), so its top face cannot
be isolated reliably and the 6 cm stereo baseline is too short for 3-D shape.
Instead the known box (e.g. 30 x 30 x 60 mm, upright, centre at a known
height) is projected into both calibrated head cameras for candidate yaws and
XY offsets; the yaw whose outlines best overlap both detector masks wins.
A square top makes yaw observable only modulo 90 degrees: the canonical
representative in [-45, 45) is returned. Needs complete object masks (YOLO
segmentation); colour thresholds that drop shadowed faces bias the fit.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def canonical_deg(yaw_deg):
    return (yaw_deg + 45.0) % 90.0 - 45.0


@dataclass
class HeadYaw:
    yaw_deg: float     # canonical, [-45, 45)
    iou: float         # mean silhouette overlap of the two cameras at the fit
    xy: tuple          # fitted box centre XY in the extrinsics' frame (base_link)


def box_corners(half_size):
    hx, hy, hz = half_size
    return np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def projected_hull(K, rotation, origin, centre, yaw_deg, corners):
    """Image convex hull of the box. rotation/origin: optical frame -> base (columns are optical axes)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    points = corners @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + centre
    in_camera = (points - origin) @ rotation
    if np.any(in_camera[:, 2] <= 1e-6):
        raise ValueError('Box is behind the camera')
    pixels = (K @ (in_camera / in_camera[:, 2:3]).T).T[:, :2]
    return cv2.convexHull(pixels.astype(np.float32))


def silhouette_iou(mask, hull, roi):
    x0, y0, x1, y1 = roi
    polygon = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(polygon, np.round((hull.reshape(-1, 2) - [x0, y0]) * 4).astype(np.int32), 1, shift=2)
    observed = mask[y0:y1, x0:x1] > 0
    union = np.count_nonzero(observed | (polygon > 0))
    return np.count_nonzero(observed & (polygon > 0)) / union if union else 0.0


def mask_roi(mask, margin=15):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Empty head mask')
    if xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1] - 1 or ys.max() == mask.shape[0] - 1:
        raise ValueError('Head mask touches the image border')
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(mask.shape[1], xs.max() + margin + 1), min(mask.shape[0], ys.max() + margin + 1))


def fit_head_yaw(masks, K, extrinsics, xy0, centre_z, half_size, xy_search=0.012):
    """Fit yaw (and a small XY correction) of an upright box to head masks.

    masks: binary images (left, right); K: 3x3 intrinsics shared by both;
    extrinsics: [(rotation optical->base, origin in base)] per camera; xy0:
    initial XY (stereo triangulation). Coarse-to-fine search: XY on a 3 mm grid
    with 15-degree yaw samples, then 1-degree yaw, then 1 mm / 0.25 degree.
    """
    K = np.asarray(K, float).reshape(3, 3)
    corners = box_corners(half_size)
    rois = [mask_roi(m) for m in masks]

    def score(xy, yaw):
        centre = np.array([xy[0], xy[1], centre_z])
        return sum(silhouette_iou(m, projected_hull(K, R, o, centre, yaw, corners), roi)
                   for m, (R, o), roi in zip(masks, extrinsics, rois)) / len(masks)

    def joint(xy, yaw, yaws, steps):
        # Yaw and XY are coupled (a shifted outline mimics a turned one): search them together.
        return max(((score((xy[0] + dx, xy[1] + dy), y), (xy[0] + dx, xy[1] + dy), y)
                    for y in yaws for dx in steps for dy in steps), key=lambda t: t[0])

    grid = np.arange(-xy_search, xy_search + 1e-9, 0.003)
    _, xy, _ = joint(xy0, 0.0, np.arange(-45.0, 45.0, 15.0), grid)
    _, yaw = max((score(xy, y), y) for y in np.arange(-45.0, 45.0, 1.0))
    _, xy, yaw = joint(xy, yaw, np.arange(yaw - 4.0, yaw + 4.01, 1.0), np.arange(-0.004, 0.00401, 0.001))
    best, xy, yaw = joint(xy, yaw, np.arange(yaw - 1.0, yaw + 1.01, 0.25), np.arange(-0.001, 0.00101, 0.0005))
    return HeadYaw(canonical_deg(yaw), float(best), (float(xy[0]), float(xy[1])))


def combine_yaws(yaws_deg):
    """Median of canonical yaws, unwrapped about the first (handles the +/-45 seam); and spread."""
    if not yaws_deg:
        raise ValueError('No yaw estimates')
    reference = yaws_deg[0]
    unwrapped = [reference + canonical_deg(y - reference) for y in yaws_deg]
    median = float(np.median(unwrapped))
    return canonical_deg(median), float(max(abs(u - median) for u in unwrapped))


def draw_head_yaw(images, masks, K, extrinsics, centre_z, half_size, fit=None, note='', height=240, pad=40):
    """Display image (RGB): a zoomed crop of each head camera around the block, with the
    detector mask edge (blue) and the box projected at the fitted yaw (green). fit None or
    a red note marks a frame whose yaw was not accepted. Display only; nothing reads it."""
    K = np.asarray(K, float).reshape(3, 3)
    corners = box_corners(half_size)
    panels = []
    for image, mask, (rotation, origin) in zip(images, masks, extrinsics):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(mask.shape[1], xs.max() + pad + 1), min(mask.shape[0], ys.max() + pad + 1)
        zoom = height / (y1 - y0)
        crop = cv2.resize(np.ascontiguousarray(image[y0:y1, x0:x1]), None, fx=zoom, fy=zoom,
                          interpolation=cv2.INTER_LINEAR)

        def pixels(points):
            return np.round((np.asarray(points, float).reshape(-1, 2) - [x0, y0]) * zoom * 4).astype(np.int32)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            cv2.polylines(crop, [pixels(contour.reshape(-1, 2) + 0.5)], True, (0, 140, 255), 2, cv2.LINE_AA, shift=2)
        if fit is not None:
            try:
                hull = projected_hull(K, rotation, origin, np.array([fit.xy[0], fit.xy[1], centre_z]),
                                      fit.yaw_deg, corners)
                cv2.polylines(crop, [pixels(hull)], True, (0, 220, 0), 2, cv2.LINE_AA, shift=2)
            except ValueError:
                pass
        panels.append(crop)
    out = np.hstack([cv2.copyMakeBorder(p, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in panels])
    text = f'head yaw {fit.yaw_deg:+.1f} deg  IoU {fit.iou:.2f}' if fit is not None else ''
    colour = (0, 230, 0)
    if note:
        text, colour = (text + '  ' if text else '') + note, (255, 60, 60)
    for thickness, c in ((4, (0, 0, 0)), (1, colour)):
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, thickness + 1, cv2.LINE_AA)
    return out

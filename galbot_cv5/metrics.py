"""Mask comparison metrics shared by the benchmark and tests (NumPy + OpenCV only)."""

import cv2
import numpy as np


def mask_iou(a, b):
    """IoU of two binary masks; None when both are empty (undefined)."""
    a, b = np.asarray(a) != 0, np.asarray(b) != 0
    union = np.count_nonzero(a | b)
    return None if union == 0 else np.count_nonzero(a & b) / union


def mask_centroid(mask):
    """(u, v) pixel centroid from image moments; None for an empty mask."""
    moments = cv2.moments((np.asarray(mask) != 0).astype(np.uint8), binaryImage=True)
    if not moments['m00']:
        return None
    return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']


def centroid_error(a, b):
    """Euclidean centroid distance in pixels; None when either mask is empty."""
    ca, cb = mask_centroid(a), mask_centroid(b)
    if ca is None or cb is None:
        return None
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def polygon_mask(label_text, shape):
    """Rasterise a YOLO-seg label (class x1 y1 x2 y2 ... normalised) to a 0/255 mask.

    Identical to tools/evaluate_cube_yolo.py:truth_mask so ground truth matches
    the FYP's existing evaluation.
    """
    mask = np.zeros(shape[:2], np.uint8)
    label = label_text.strip()
    if label:
        values = np.asarray([float(v) for v in label.split()[1:]]).reshape(-1, 2)
        polygon = np.rint(values * [shape[1], shape[0]]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def summarise(values):
    """Robust summary of a list of floats (ignores None)."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(n=0)
    return dict(n=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.percentile(values, 5)), p95=float(np.percentile(values, 95)),
                min=float(values.min()), max=float(values.max()))

"""Synthetic tests for the OpenCV 5 YOLO-seg decoder; no model, dataset or torch needed."""

import cv2
import numpy as np
import pytest

from galbot_cv5.yolo_seg_dnn import (decode_detections, letterbox, reconstruct_masks,
                                     scale_boxes_to_original)


def test_opencv_major_version_is_5():
    assert int(cv2.__version__.split('.')[0]) >= 5


def test_letterbox_is_noop_for_fyp_camera_shape():
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)
    padded, info = letterbox(image, (480, 640))
    assert padded.shape == (480, 640, 3)
    assert np.array_equal(padded, image)
    assert (info.gain, info.left, info.top) == (1.0, 0, 0)


def test_letterbox_square_pads_centre_with_114():
    image = np.zeros((480, 640, 3), np.uint8)
    padded, info = letterbox(image, (640, 640))
    assert (info.top, info.left, info.gain) == (80, 0, 1.0)
    assert (padded[:80] == 114).all() and (padded[560:] == 114).all()
    assert (padded[80:560] == 0).all()


@pytest.mark.parametrize('h, w', [(333, 500), (500, 333), (101, 77)])
def test_letterbox_odd_sizes_follow_ultralytics_rounding(h, w):
    padded, info = letterbox(np.zeros((h, w, 3), np.uint8), (640, 640))
    gain = min(640 / h, 640 / w)
    new_w, new_h = round(w * gain), round(h * gain)
    assert padded.shape == (640, 640, 3)
    assert info.top == round((640 - new_h) / 2 - 0.1)
    assert info.left == round((640 - new_w) / 2 - 0.1)


def test_scale_boxes_inverts_letterbox_and_clips():
    boxes = np.array([[100, 180, 200, 280], [-10, 70, 700, 600]], np.float32)  # 640x640 letterbox space
    out = scale_boxes_to_original(boxes, (640, 640), (480, 640))
    assert np.allclose(out[0], [100, 100, 200, 200])
    assert np.allclose(out[1], [0, 0, 640, 480])  # clipped to the image


def raw_output(rows, num_classes=1, num_masks=32):
    """Build a (1, 4 + nc + nm, N) head output from (cx, cy, w, h, scores, coeffs) rows."""
    columns = []
    for cx, cy, w, h, scores, coeffs in rows:
        columns.append(np.r_[cx, cy, w, h, scores, coeffs])
    return np.asarray(columns, np.float32).T[None]


def test_decode_threshold_nms_and_order():
    c = np.zeros(32)
    rows = [(100, 100, 40, 40, [0.90], c),     # kept
            (102, 101, 40, 40, [0.80], c),     # suppressed by NMS (IoU ~0.9 with the first)
            (300, 300, 40, 40, [0.60], c),     # kept, separate object
            (400, 400, 40, 40, [0.35], c)]     # dropped: score must be strictly > threshold
    boxes, scores, classes, coefficients = decode_detections(raw_output(rows), conf_threshold=0.35)
    assert np.allclose(scores, [0.9, 0.6])
    assert np.allclose(boxes[0], [80, 80, 120, 120])
    assert classes.tolist() == [0, 0]
    assert coefficients.shape == (2, 32)


def test_decode_nms_is_class_aware_and_class_filter_applies():
    c = np.zeros(32)
    rows = [(100, 100, 40, 40, [0.9, 0.1], c), (101, 100, 40, 40, [0.1, 0.8], c)]
    boxes, scores, classes, _ = decode_detections(raw_output(rows, num_classes=2), 0.35)
    assert sorted(classes.tolist()) == [0, 1]     # same place, different classes: both survive
    _, scores, classes, _ = decode_detections(raw_output(rows, num_classes=2), 0.35, classes=[1])
    assert classes.tolist() == [1] and np.allclose(scores, [0.8])


def test_decode_empty_and_max_det():
    c = np.zeros(32)
    assert len(decode_detections(raw_output([(10, 10, 4, 4, [0.1], c)]), 0.35)[0]) == 0
    rows = [(20 + 50 * i, 50, 10, 10, [0.5 + 0.01 * i], c) for i in range(10)]
    boxes, scores, _, _ = decode_detections(raw_output(rows), 0.35, max_det=3)
    assert len(boxes) == 3 and np.all(np.diff(scores) <= 0)


def test_decode_rejects_bad_channel_count():
    with pytest.raises(ValueError):
        decode_detections(np.zeros((1, 30, 5), np.float32), 0.35)


def test_reconstruct_masks_removes_padding_upsamples_and_crops():
    # 640x640 letterbox of a 480x640 image -> 160x160 prototypes with 20 rows of padding top/bottom.
    protos = np.full((32, 160, 160), -1.0, np.float32)
    protos[0, 20 + 30:20 + 60, 40:80] = 1.0          # positive blob, rows 30..59 / cols 40..79 of content
    protos[0, :20] = 5.0                              # padding must be discarded
    coefficients = np.zeros((1, 32), np.float32)
    coefficients[0, 0] = 1.0
    box = scale_boxes_to_original(np.array([[0, 80, 640, 560]], np.float32), (640, 640), (480, 640))
    mask = reconstruct_masks(protos[None], coefficients, box, (640, 640), (480, 640))[0]
    assert mask.shape == (480, 640) and set(np.unique(mask)) <= {0, 1}
    ys, xs = np.nonzero(mask)
    # 4x upsampling of the blob; bilinear edges land within one prototype pixel (4 px).
    assert abs(ys.min() - 120) <= 4 and abs(ys.max() - 239) <= 4
    assert abs(xs.min() - 160) <= 4 and abs(xs.max() - 319) <= 4
    assert not mask[:100].any()                       # padding logits did not leak in

    cropped = reconstruct_masks(protos[None], coefficients, np.array([[0, 0, 200, 480]], np.float32),
                                (640, 640), (480, 640))[0]
    assert not cropped[:, 200:].any() and cropped[:, :200].any()


def test_reconstruct_masks_empty():
    masks = reconstruct_masks(np.zeros((1, 32, 120, 160), np.float32), np.zeros((0, 32), np.float32),
                              np.zeros((0, 4), np.float32), (480, 640), (480, 640))
    assert masks.shape == (0, 480, 640)

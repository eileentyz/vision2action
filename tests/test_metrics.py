import numpy as np
import pytest

from galbot_cv5.metrics import centroid_error, mask_centroid, mask_iou, polygon_mask, summarise


def square(x0, y0, size, shape=(100, 100)):
    mask = np.zeros(shape, np.uint8)
    mask[y0:y0 + size, x0:x0 + size] = 255
    return mask


def test_iou():
    assert mask_iou(square(10, 10, 10), square(10, 10, 10)) == 1.0
    assert mask_iou(square(10, 10, 10), square(15, 10, 10)) == pytest.approx(50 / 150)
    assert mask_iou(square(0, 0, 5), square(50, 50, 5)) == 0.0
    assert mask_iou(np.zeros((5, 5)), np.zeros((5, 5))) is None


def test_centroid_and_error():
    assert mask_centroid(square(10, 20, 10)) == pytest.approx((14.5, 24.5))
    assert mask_centroid(np.zeros((5, 5))) is None
    assert centroid_error(square(10, 10, 10), square(13, 14, 10)) == pytest.approx(5.0)
    assert centroid_error(square(10, 10, 10), np.zeros((100, 100))) is None


def test_polygon_mask_matches_fyp_label_format():
    mask = polygon_mask('0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5', (100, 200))
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 100, 10, 50)
    assert not polygon_mask('', (10, 10)).any()


def test_summarise_ignores_none():
    s = summarise([1.0, None, 3.0])
    assert s['n'] == 2 and s['median'] == 2.0
    assert summarise([None]) == dict(n=0)

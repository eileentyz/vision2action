"""galbot_cv5.geometry.mask_geometry must equal the FYP yolo_mask.mask_geometry.

The FYP function is compiled from its current source with ``ast`` (importing
the module would pull in rclpy), so any drift on either side fails here. yolo_mask.py
is not part of this repository: set GALBOT_FYP_PACKAGE to the FYP package directory
(packages/galbot_mobile_manipulation of the FYP checkout) to run these tests.
"""
import ast
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from galbot_cv5.geometry import FYP_PACKAGE, fyp_modules, mask_geometry

SOURCE = Path(os.environ.get('GALBOT_FYP_PACKAGE', FYP_PACKAGE)) / 'galbot_mobile_manipulation/yolo_mask.py'
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason='FYP yolo_mask.py not available (GALBOT_FYP_PACKAGE)')
DATA = Path(os.environ.get('GALBOT_CUBE_DATASET', Path(__file__).resolve().parents[1] / 'data/cube_seg'))
REFERENCE = Path(__file__).resolve().parents[1] / f'data/reference/{DATA.name}_val_ultralytics.npz'


def load_fyp_mask_geometry():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mask_geometry')
    namespace = {'cv2': cv2, 'np': np, 'math': math}
    exec(compile(ast.Module([node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['mask_geometry']


fyp_mask_geometry = load_fyp_mask_geometry() if SOURCE.exists() else None


def assert_same(mask):
    ours, theirs = mask_geometry(mask), fyp_mask_geometry(mask)
    assert len(ours['contours']) == len(theirs['contours'])
    for a, b in zip(ours['contours'], theirs['contours']):
        assert np.array_equal(a, b)
    assert ours['centroid'] == theirs['centroid']
    assert (ours['box'] is None) == (theirs['box'] is None)
    if ours['box'] is not None:
        assert np.array_equal(ours['box'], theirs['box'])
    assert ours['angle'] == theirs['angle']
    return ours


def blank():
    return np.zeros((120, 160), np.uint8)


def rotated_rect(angle, size=(60, 20), centre=(80, 60)):
    mask = blank()
    cv2.fillConvexPoly(mask, np.rint(cv2.boxPoints((centre, size, angle))).astype(np.int32), 255)
    return mask


def test_source_function_found():
    assert SOURCE.exists()
    assert callable(fyp_mask_geometry)


@pytest.mark.parametrize('angle', [-89, -60, -45, -30, -10, 0, 10, 30, 45, 60, 89, 90])
def test_rotated_rectangles(angle):
    geometry = assert_same(rotated_rect(angle))
    assert geometry['angle'] is not None


def test_square_has_no_angle():
    mask = blank()
    mask[40:80, 60:100] = 255
    geometry = assert_same(mask)
    assert geometry['box'] is not None and geometry['angle'] is None


def test_empty_mask():
    geometry = assert_same(blank())
    assert geometry['centroid'] is None and geometry['box'] is None and geometry['contours'] == ()


@pytest.mark.parametrize('case', ['single_pixel', 'one_pixel_line', 'two_pixel_line'])
def test_degenerate_thin_shapes(case):
    mask = blank()
    if case == 'single_pixel':
        mask[50, 50] = 255
    elif case == 'one_pixel_line':
        mask[50, 20:100] = 255
    else:
        mask[50:52, 20:100] = 255
    assert_same(mask)


def test_two_blobs_hole_and_border():
    mask = blank()
    cv2.circle(mask, (40, 60), 25, 255, -1)
    cv2.circle(mask, (40, 60), 10, 0, -1)           # hole -> inner contour with RETR_LIST
    mask[0:30, 120:160] = 255                         # second blob touching the border
    assert_same(mask)


def test_binary_values_other_than_255():
    mask = rotated_rect(25)
    mask[mask != 0] = 1
    assert_same(mask)


def test_random_blobs_regression():
    rng = np.random.default_rng(20261022)
    for _ in range(50):
        mask = blank()
        for _ in range(rng.integers(1, 4)):
            centre = tuple(int(v) for v in rng.integers((10, 10), (150, 110)))
            axes = tuple(int(v) for v in rng.integers(1, 30, 2))
            cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
        assert_same(mask)


@pytest.mark.skipif(not (REFERENCE.exists() and DATA.exists()), reason='local dataset/reference not available')
def test_real_segmentation_masks():
    reference = np.load(REFERENCE, allow_pickle=True)
    checked = 0
    for packed, (h, w), name in zip(reference['masks'], reference['shapes'], reference['names']):
        mask = np.unpackbits(np.asarray(packed, np.uint8), count=int(h * w)).reshape(h, w) * 255
        assert_same(mask.astype(np.uint8))
        checked += 1
    assert checked == len(reference['names'])


def test_fyp_geometry_modules_import_without_ros():
    head_localisation, head_yaw = fyp_modules()
    assert callable(head_localisation.triangulate) and callable(head_yaw.fit_head_yaw)

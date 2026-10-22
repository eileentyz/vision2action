"""ROS-free geometry used by the evaluation.

``mask_geometry`` is a verbatim port of
``galbot_mobile_manipulation.yolo_mask.mask_geometry`` (that module imports rclpy
through red_cube_detector, so it cannot be imported in the OpenCV 5 / NumPy 2
environment). tests/test_geometry_equivalence.py checks it against the FYP
source on every run, so a change on either side fails the tests.

The stereo and silhouette-fit functions are imported, not copied, from the FYP
package's ROS-free modules (head_localisation, head_yaw); see ``fyp_modules``.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

FYP_PACKAGE = Path(__file__).resolve().parents[1] / 'fyp'  # the two vendored FYP modules


def mask_geometry(mask):
    """Inspect the selected instance without eroding, filling holes or HSV gating."""
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(mask, binaryImage=True)
    geometry = {'contours': contours, 'centroid': None, 'box': None, 'angle': None}
    if not moments['m00']:
        return geometry
    geometry['centroid'] = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    if not contours:
        return geometry
    rect = cv2.minAreaRect(np.concatenate(contours))
    if min(rect[1]) < 2:
        return geometry
    box = cv2.boxPoints(rect)
    geometry['box'] = box
    # Nearly square masks have no stable long-axis orientation.
    if max(rect[1]) > 1.1 * min(rect[1]):
        edges = np.roll(box, -1, axis=0) - box
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        geometry['angle'] = (math.atan2(float(edge[1]), float(edge[0])) + math.pi/2) % math.pi - math.pi/2
    return geometry


def fyp_modules():
    """(head_localisation, head_yaw) from the installed FYP package, else the copies in fyp/.

    Both modules depend only on NumPy and OpenCV. The Lambda image copies these
    two files under galbot_mobile_manipulation/.
    """
    try:
        from galbot_mobile_manipulation import head_localisation, head_yaw
    except ImportError:
        if str(FYP_PACKAGE) not in sys.path:
            sys.path.insert(0, str(FYP_PACKAGE))
        from galbot_mobile_manipulation import head_localisation, head_yaw
    return head_localisation, head_yaw

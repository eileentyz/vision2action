"""Model-free OpenCV segmentation of a red object (for synthetic data and as a baseline).

HSV thresholds on both red hue bands, morphological open/close, then the largest
external contour filled. Independent of the FYP's red_cube_detector (which imports
rclpy); thresholds are arguments so the evaluation records what was used.
"""

import cv2
import numpy as np

DEFAULT_BANDS = (((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (180, 255, 255)))


def red_mask(rgb, bands=DEFAULT_BANDS, kernel=3, min_area=20.0):
    """0/255 mask of the largest red region in an RGB uint8 image (empty if none >= min_area)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('red_mask expects an HxWx3 uint8 RGB image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw = np.zeros(rgb.shape[:2], np.uint8)
    for low, high in bands:
        raw |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    cleaned = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, element), cv2.MORPH_CLOSE, element)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask

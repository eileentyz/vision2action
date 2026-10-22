# Perception evaluation: synthetic-smoke

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.15 (x86_64); run 2026-10-22T06:26:38.407382+00:00 to 2026-10-22T06:26:45.000696+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `inputs/synthetic_stereo.npz` (synthetic, 12 stereo samples)

Procedural OpenCV-drawn stereo views of a red 30x30x60 mm box (no project data)

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| hsv | 12/12 | 0.984 / 0.991 / 0.992 | 0.12 / 0.47 / 0.73 | 0.43 / 0.75 / 0.94 | 0.44 / 0.82 / 1.03 | 0.81 / 2.05 / 2.12 | 3.4 / 569.0 / 1260.1 |
| gt | 12/12 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.82 / 1.36 / 1.40 | 0.25 / 0.51 / 0.51 | 0.31 / 0.79 / 0.84 | 0.0 / 0.0 / 0.0 |

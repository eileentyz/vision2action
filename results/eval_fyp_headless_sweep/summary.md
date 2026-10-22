# Perception evaluation: fyp-headless-sweep

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.3 (x86_64); run 2026-10-22T04:52:13.054872+00:00 to 2026-10-22T04:52:19.370963+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `build/bundles/headless_sweep.npz` (fyp_replay, 21 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 21/21 | 0.923 / 0.934 / 0.936 | 0.50 / 0.65 / 0.68 | 1.53 / 2.97 / 4.87 | 1.53 / 2.57 / 2.74 | 1.02 / 2.98 / 6.02 | 49.4 / 55.4 / 65.2 |
| gt | 21/21 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.66 / 1.33 / 1.36 | 0.65 / 1.50 / 2.05 | 3.51 / 7.27 / 7.52 | 0.0 / 0.0 / 0.0 |
| hsv | 21/21 | 0.893 / 0.969 / 0.971 | 0.61 / 4.30 / 4.38 | 6.81 / 10.46 / 10.90 | 1.39 / 4.16 / 4.19 | 9.02 / 36.27 / 41.27 | 1.4 / 2.0 / 3.5 |

## Live FYP perception in recorded missions

11/11 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 10 missions succeeded.

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 1.62 / 2.49 / 2.99 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 2.01 / 3.89 / 4.02 |

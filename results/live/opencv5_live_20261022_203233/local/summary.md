# Perception evaluation: live-20261022_203233

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.3 (x86_64); run 2026-10-22T12:34:56.705874+00:00 to 2026-10-22T12:34:57.531716+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `opencv5_live_20261022_203233.npz` (fyp_replay, 3 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 3/3 | 0.900 / 0.903 / 0.904 | 0.63 / 0.74 / 0.75 | 2.94 / 3.04 / 3.05 | 2.21 / 2.24 / 2.25 | 0.27 / 0.46 / 0.48 | 42.3 / 48.0 / 48.6 |
| gt | 3/3 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.19 / 0.58 / 0.62 | 0.19 / 0.19 / 0.19 | 3.77 / 5.12 / 5.27 | 0.0 / 0.0 / 0.0 |
| hsv | 3/3 | 0.915 / 0.943 / 0.945 | 0.26 / 0.50 / 0.52 | 6.76 / 7.36 / 7.43 | 0.49 / 2.15 / 2.33 | 3.27 / 5.07 / 5.27 | 1.2 / 1.3 / 1.3 |

## Live missions: the robot's own detector during the mission

1/1 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 1 missions succeeded.

Input: `live_missions.json` (sanitised pose records; no images).

Detector on the robot: opencv5_dnn; median per-frame inference 33.5 / 33.5 / 33.5 ms (median / p95 / max over trials).

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 2.86 / 2.86 / 2.86 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 0.52 / 0.52 / 0.52 |

| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |
|---|---|---|---|---|---|---|
| opencv5_live_20261022_203233/yaw+30-1 | ok | opencv5_dnn 5.0.0 | 242 (242) | 33.5 | 2.86 | -0.52 |

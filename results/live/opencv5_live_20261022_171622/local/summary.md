# Perception evaluation: live-20261022_171622

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.3 (x86_64); run 2026-10-22T09:19:02.470869+00:00 to 2026-10-22T09:19:03.321856+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `opencv5_live_20261022_171622.npz` (fyp_replay, 3 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 3/3 | 0.913 / 0.924 / 0.925 | 0.55 / 0.69 / 0.71 | 1.12 / 1.60 / 1.66 | 1.30 / 2.35 / 2.47 | 0.24 / 1.14 / 1.24 | 50.4 / 64.3 / 65.8 |
| gt | 3/3 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.24 / 0.35 / 0.36 | 0.24 / 0.35 / 0.36 | 2.51 / 2.73 / 2.76 | 0.0 / 0.0 / 0.0 |
| hsv | 3/3 | 0.921 / 0.950 / 0.951 | 0.26 / 0.50 / 0.50 | 6.25 / 6.74 / 6.79 | 0.99 / 1.24 / 1.27 | 4.01 / 4.23 / 4.26 | 1.3 / 1.6 / 1.6 |

## Live missions: the robot's own detector during the mission

1/1 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 1 missions succeeded.

Input: `live_missions.json` (sanitised pose records; no images).

Detector on the robot: opencv5_dnn; median per-frame inference 34.1 / 34.1 / 34.1 ms (median / p95 / max over trials).

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 1.60 / 1.60 / 1.60 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 0.26 / 0.26 / 0.26 |

| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |
|---|---|---|---|---|---|---|
| opencv5_live_20261022_171622/yaw+30-1 | ok | opencv5_dnn 5.0.0 | 268 (268) | 34.1 | 1.60 | -0.26 |

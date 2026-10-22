# Perception evaluation: live-20261022_180808

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.15 (x86_64); run 2026-10-22T10:11:16.936655+00:00 to 2026-10-22T10:11:19.849598+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `inputs/opencv5_live_20261022_180808.npz` (fyp_replay, 3 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 3/3 | 0.907 / 0.912 / 0.912 | 0.64 / 0.79 / 0.79 | 2.20 / 2.28 / 2.28 | 1.64 / 2.33 / 2.41 | 1.76 / 2.21 / 2.26 | 250.3 / 307.2 / 313.5 |
| gt | 3/3 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.21 / 0.26 / 0.27 | 0.25 / 0.30 / 0.30 | 3.26 / 3.26 / 3.26 | 0.0 / 0.0 / 0.0 |
| hsv | 3/3 | 0.923 / 0.950 / 0.950 | 0.27 / 0.47 / 0.47 | 6.68 / 6.95 / 6.98 | 1.83 / 2.06 / 2.09 | 5.26 / 5.26 / 5.26 | 3.5 / 6.8 / 7.1 |

## Live missions: the robot's own detector during the mission

1/1 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 1 missions succeeded.

Input: `inputs/live_missions.json` (sanitised pose records; no images).

Detector on the robot: opencv5_dnn; median per-frame inference 46.1 / 46.1 / 46.1 ms (median / p95 / max over trials).

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 2.28 / 2.28 / 2.28 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 1.51 / 1.51 / 1.51 |

| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |
|---|---|---|---|---|---|---|
| opencv5_live_20261022_180808/yaw+30-1 | ok | opencv5_dnn 5.0.0 | 170 (162) | 46.1 | 2.28 | -1.51 |

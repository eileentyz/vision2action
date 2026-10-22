# Perception evaluation: live-20261022_173928

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.15 (x86_64); run 2026-10-22T09:56:09.859951+00:00 to 2026-10-22T09:56:17.541331+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `inputs/opencv5_live_20261022_173928.npz` (fyp_replay, 3 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 3/3 | 0.896 / 0.903 / 0.904 | 0.60 / 0.73 / 0.74 | 2.44 / 2.60 / 2.62 | 2.23 / 2.72 / 2.77 | 1.27 / 2.36 / 2.48 | 245.6 / 1546.7 / 1691.3 |
| gt | 3/3 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.41 / 0.54 / 0.56 | 0.12 / 0.13 / 0.13 | 2.77 / 2.77 / 2.77 | 0.0 / 0.0 / 0.0 |
| hsv | 3/3 | 0.911 / 0.943 / 0.943 | 0.30 / 0.53 / 0.53 | 7.05 / 7.12 / 7.13 | 0.88 / 0.88 / 0.88 | 5.27 / 5.27 / 5.27 | 3.4 / 190.2 / 210.9 |

## Live missions: the robot's own detector during the mission

1/1 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 1 missions succeeded.

Input: `inputs/live_missions.json` (sanitised pose records; no images).

Detector on the robot: opencv5_dnn; median per-frame inference 42.3 / 42.3 / 42.3 ms (median / p95 / max over trials).

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 1.71 / 1.71 / 1.71 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 1.02 / 1.02 / 1.02 |

| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |
|---|---|---|---|---|---|---|
| opencv5_live_20261022_173928/yaw+30-1 | ok | opencv5_dnn 5.0.0 | 154 (154) | 42.3 | 1.71 | -1.02 |

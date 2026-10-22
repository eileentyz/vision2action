# Perception evaluation: live-20261022_203608

OpenCV 5.0.0, NumPy 2.5.4, Python 3.12.3 (x86_64); run 2026-10-22T12:38:48.891112+00:00 to 2026-10-22T12:38:49.814139+00:00.

Statistics are median / p95 / max. Yaw errors are modulo 90 deg (square footprint).

## Bundle `opencv5_live_20261022_203608.npz` (fyp_replay, 3 stereo samples)

Re-rendered head stereo views of recorded FYP trials

| Mask source | Pose estimated | Mask IoU | Centroid error (px) | Stereo XY error (mm) | Fit XY error (mm) | Yaw error (deg) | Segmentation ms |
|---|---|---|---|---|---|---|---|
| dnn | 3/3 | 0.906 / 0.909 / 0.910 | 0.47 / 0.53 / 0.53 | 2.27 / 2.44 / 2.46 | 1.16 / 1.33 / 1.35 | 0.24 / 1.37 / 1.49 | 56.1 / 59.0 / 59.3 |
| gt | 3/3 | 1.000 / 1.000 / 1.000 | 0.00 / 0.00 / 0.00 | 0.56 / 0.60 / 0.60 | 0.53 / 0.58 / 0.58 | 3.01 / 3.01 / 3.01 | 0.0 / 0.0 / 0.0 |
| hsv | 3/3 | 0.923 / 0.951 / 0.951 | 0.25 / 0.45 / 0.45 | 6.18 / 6.71 / 6.77 | 0.56 / 0.57 / 0.57 | 3.76 / 5.11 / 5.26 | 1.6 / 1.8 / 1.8 |

## Live missions: the robot's own detector during the mission

1/1 trials evaluated (first pose consumed by the pick vs simulator truth at the same time); 1 missions succeeded.

Input: `live_missions.json` (sanitised pose records; no images).

Detector on the robot: opencv5_dnn; median per-frame inference 35.0 / 35.0 / 35.0 ms (median / p95 / max over trials).

| Metric | median / p95 / max |
|---|---|
| XY position error (mm) | 2.55 / 2.55 / 2.55 |
| abs Z error (mm) | 0.00 / 0.00 / 0.00 |
| abs yaw error (deg) | 0.01 / 0.01 / 0.01 |

| Trial | Mission | Detector | Frames (cube) | Inference ms | XY error (mm) | Yaw error (deg) |
|---|---|---|---|---|---|---|
| opencv5_live_20261022_203608/yaw+30-1 | ok | opencv5_dnn 5.0.0 | 252 (252) | 35.0 | 2.55 | -0.01 |

# Live demo

`run_live_demo.sh` runs the whole competition flow in one command. **It needs the FYP ROS 2
workspace, which is private and not part of this repository** (see
[`../integration/ros2/`](../integration/ros2/)). Without it, see the recorded evidence in
[`../results/live/`](../results/live/) and [`../docs/media/`](../docs/media/).
1. It starts the OpenCV 5 mask server.
2. It runs the robot mission with `yolo_backend:=opencv5`.
3. It exports the run and writes the local report.
4. With `--cloud`, it adds the AWS report.

What you'll see, and how to tell OpenCV 5 is the detector: [`../README.md`](../README.md#6-live-robot-demonstration-needs-the-private-fyp-workspace).

```bash
live/run_live_demo.sh --yaw 30 --viewer [--record] [--cloud] [--run-dir DIR]
```

| Option | Effect |
|---|---|
| `--yaw DEG` | cube yaw for one mission; repeat for several (default 30) |
| `--viewer` | MuJoCo viewer and camera panel; leave out for headless |
| `--record` | save every frame OpenCV 5 served, with its mask, to `build/live/<run>/frames/` (local only; `tools/make_demo_media.py` makes the video) |
| `--cloud` | upload the sanitised job and fetch the Lambda report (needs `AWS_PROFILE` and `aws login`) |
| `--run-dir DIR` | skip the mission and reuse a finished run |

Environment overrides:
- `GALBOT_FYP_WS` (required): the FYP colcon workspace
- `GALBOT_FYP_REPO`: the FYP source checkout (default `$GALBOT_FYP_WS/src/galbot_mobile_manipulation`)
- `GALBOT_CV5_WS`: an overlay workspace, if the OpenCV 5 backend is built separately
- `GALBOT_CV5_SOCKET`: the socket path (default `$XDG_RUNTIME_DIR/galbot_cv5_mask.sock`)

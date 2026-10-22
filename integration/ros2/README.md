# ROS 2 integration: the OpenCV 5 backend on the robot

These files add an opt-in OpenCV 5 detector to the robot's perception node. The robot
software itself (the Galbot FYP ROS 2 workspace: mission, Nav2 and MoveIt 2
configuration, worlds) is the author's final-year project. It is **not public**, so the
live robot demo can't be rerun from this repository alone. The recorded evidence of the
live missions is in [`../../results/live/`](../../results/live/) and
[`../../docs/media/`](../../docs/media/).

| File | Role |
|---|---|
| `galbot_mobile_manipulation/opencv5_protocol.py` | Wire protocol between the ROS 2 node (NumPy 1) and the OpenCV 5 server (NumPy 2): fixed `struct` headers, raw `rgb8` frames in, bit-packed masks out. Identical to `galbot_cv5/opencv5_protocol.py`, and a test checks this. |
| `galbot_mobile_manipulation/opencv5_mask.py` | `OpenCV5MaskProvider`: the mask provider the node uses when `yolo_backend:=opencv5`. It reconnects on failure and never falls back to Ultralytics. It imports two helpers from the FYP package (`red_cube_detector.rgb_from_message`, `yolo_mask.mask_geometry` and `debug_overlay`). |
| `fyp_opencv5_backend.patch` | The changes to 4 FYP files: the `yolo_backend` / `opencv5_socket` launch arguments and node parameters, and the factory branch in `object_mask.py`. The default backend is unchanged (`ultralytics`). |
| `test_opencv5_mask.py` | The FYP-side tests, run against a fake server. They need the FYP package and ROS 2. |

How a frame flows:

```text
camera (ROS 2 Image) -> object_pointcloud node -> OpenCV5MaskProvider
    --Unix socket, opencv5_protocol--> tools/opencv5_mask_server.py (OpenCV 5 cv2.dnn, models/*.onnx)
    <-- mask + score + forward/total ms --
-> FYP mask geometry -> stereo point + yaw fit -> pick pose -> MoveIt 2 / Nav2 mission
```

With the FYP workspace available, the steps are:
1. Apply the patch to the FYP checkout and copy in the two modules.
2. Build the workspace.
3. Run [`../../live/run_live_demo.sh`](../../live/run_live_demo.sh).

The server side runs without ROS 2. See "OpenCV 5 mask server" in the top-level README.

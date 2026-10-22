# Notices and attribution

Vision2Action is distributed under the **GNU Affero General Public License v3.0**
(`LICENSE`).

## Third-party material in this repository

| Material | Where | Origin and licence |
|---|---|---|
| Cube segmentation model (ONNX) | `models/` | Fine-tuned from Ultralytics YOLO11n-seg weights, AGPL-3.0 (https://www.ultralytics.com/license). See `models/README.md`. |
| `head_localisation.py`, `head_yaw.py` | `fyp/galbot_mobile_manipulation/` | From the author's Galbot mobile manipulation project, included unchanged and distributed here under AGPL-3.0. |
| OpenCV 5 backend for the ROS 2 node | `integration/ros2/` | From the author's Galbot mobile manipulation project (added for this competition). AGPL-3.0. |
| Validation images and labels | `data/cube_seg/` | Rendered by the author in MuJoCo. The rendered robot is the Galbot One Golf description, Apache-2.0, © Galaxy General Robotics (https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description). |
| Demo video, GIFs and figures | `docs/media/`, `docs/figures/` | Made by the author from MuJoCo renders of the same robot description. |

## Dependencies (installed separately, not redistributed)

| Software | Licence |
|---|---|
| OpenCV 5 (`opencv-python`) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Ultralytics (export and reference only, `requirements-export.txt`) | AGPL-3.0 |
| PyTorch (export and reference only) | BSD-3-Clause |
| ONNX | Apache-2.0 |
| boto3 (Lambda handler; provided by the Lambda runtime) | Apache-2.0 |
| AWS SAM CLI (deployment only) | Apache-2.0 |
| ROS 2 Jazzy, Nav2, MuJoCo (robot side, not in this repository) | Apache-2.0 |
| MoveIt 2 (robot side, not in this repository) | BSD-3-Clause |

## Acknowledgements

Thanks to the OpenCV team for OpenCV 5 and the OpenCV AI Competition 2026, to Ultralytics
for YOLO11, to Galaxy General Robotics for the Galbot One Golf description, and to the
ROS 2, Nav2, MoveIt 2 and MuJoCo communities.

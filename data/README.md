# Data

## `cube_seg/`: 150 validation images

The validation split of the cube segmentation dataset, used for the OpenCV 5 against
Ultralytics parity benchmark (`tools/benchmark_dnn.py`, `tests/test_parity.py`).

- **Images:** 60 head-camera images (`val_head_*.png`) and 90 left-wrist-camera images
  (`val_wrist_*.png`), each 640×480, rendered in MuJoCo from the Galbot One Golf model
  and a simple pick-and-place world. 140 contain the cube and 10 are empty.
- **Labels:** YOLO segmentation polygons, generated from MuJoCo geom segmentation, so
  they are exact rather than hand-drawn.
- **Not included:** the 750-image training split.

The images are synthetic and show no people or real places. The rendered robot is the
Galbot One Golf description (Apache-2.0, see `../NOTICE.md`).

## `reference/`: Ultralytics outputs on the same 150 images

`cube_seg_val_ultralytics.npz` and `.json` hold the masks, confidences and latencies the
original Ultralytics/PyTorch detector produced for these images, made with
`tools/reference_ultralytics.py`. That run needed the FYP's ROS 2 `YoloMaskProvider`.
The OpenCV 5 benchmark compares against these, so you can check parity without
installing PyTorch, Ultralytics or ROS. The `.npz` also stores each image's SHA-256, and
the benchmark checks it.

Both are distributed under this repository's licence (`../LICENSE`).

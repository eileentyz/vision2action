# Model: `cube_yolo11n_seg.onnx`

One-class (`cube`) YOLO11n-seg instance segmentation model, exported to ONNX for
OpenCV 5 `cv2.dnn`.

| | |
|---|---|
| File | `cube_yolo11n_seg.onnx` (SHA-256 `72236da98ad910a225586c4c12f9e5130e7755d5120f22bcf93f2de8ea10b419`) |
| Sidecar | `cube_yolo11n_seg.json`: class names, input size, export versions, source checkpoint hash |
| Input | 1×3×480×640 (static), RGB, 0–1 |
| Opset | 17 |
| Exported with | Ultralytics 8.4.170, PyTorch 2.14.1 (CPU), `tools/export_onnx.py` |
| Training | fine-tuned from Ultralytics `yolo11n-seg.pt` on synthetic MuJoCo images of a red cube rendered from the Galbot head and wrist cameras (750 training, 150 validation images; the validation split is in `../data/cube_seg/`) |

The training checkpoint (`.pt`) and the training images are part of the author's
final-year project and are not included. `tools/export_onnx.py` documents how the
ONNX file was produced from the checkpoint. Only the ONNX `description` metadata
entry was changed afterwards, to remove a local path (`tools/sanitize_for_upload.py`).
The weights and graph are unchanged.

**Scope:** trained and evaluated on simulated images only. Don't assume it detects
other objects or works on real camera images.

## Licence

The weights derive from Ultralytics YOLO11 pretrained weights, which Ultralytics
distributes under **AGPL-3.0** (unless covered by an Ultralytics Enterprise licence).
This model is therefore distributed under AGPL-3.0, the licence of this repository
(`../LICENSE`). The ONNX metadata keeps Ultralytics' licence entry.

- Ultralytics licensing: https://www.ultralytics.com/license
- Ultralytics source and licence text: https://github.com/ultralytics/ultralytics

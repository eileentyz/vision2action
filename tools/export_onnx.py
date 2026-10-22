"""Export the FYP cube YOLO11n-seg checkpoint to a static-shape ONNX graph (export env only).

The default input is 480x640 (h, w): the FYP head and wrist cameras are 640x480
and Ultralytics predicts .pt checkpoints with rect=True, i.e. a stride-aligned
minimum rectangle with no padding. A static 640x640 graph would see 80 px of
letterbox padding the mission never sees and give different scores.

The checkpoint is copied into the build directory first so Ultralytics never
writes next to the FYP package's models/. The export is plain (no embedded NMS)
so OpenCV 5 performs decoding, NMS and mask reconstruction.

    .venv-export/bin/python tools/export_onnx.py
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('YOLO_OFFLINE', 'True')

HERE = Path(__file__).resolve().parents[1]
# Needs the private FYP checkout (GALBOT_FYP_REPO) for the .pt checkpoint; the ONNX it made ships in models/.
FYP_REPO = Path(os.environ['GALBOT_FYP_REPO']) if 'GALBOT_FYP_REPO' in os.environ else HERE.parent / 'galbot_mobile_manipulation'
DEFAULT_WEIGHTS = FYP_REPO / 'packages/galbot_mobile_manipulation/models/cube_yolo11n_seg.pt'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--out-dir', type=Path, default=HERE / 'build')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[480, 640], metavar=('H', 'W'))
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    from ultralytics import YOLO
    import ultralytics
    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = args.out_dir / args.weights.name
    shutil.copyfile(args.weights, weights)
    model = YOLO(str(weights))
    if model.task != 'segment':
        raise SystemExit(f'{args.weights} is not a segmentation model')
    onnx_path = Path(model.export(format='onnx', imgsz=list(args.imgsz), opset=args.opset,
                                  dynamic=False, simplify=True, nms=False, half=False, device='cpu'))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    meta = dict(source_weights=str(args.weights.relative_to(FYP_REPO)) if args.weights.is_relative_to(FYP_REPO)
                else str(args.weights),
                source_sha256=sha256(args.weights), onnx=onnx_path.name, onnx_sha256=sha256(onnx_path),
                imgsz=list(args.imgsz), opset=args.opset, names={int(k): v for k, v in names.items()},
                ultralytics=ultralytics.__version__, torch=torch.__version__)
    meta_path = onnx_path.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

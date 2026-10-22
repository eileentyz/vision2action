"""Record reference masks from the FYP's unmodified YoloMaskProvider (export env only).

Runs galbot_mobile_manipulation.yolo_mask.YoloMaskProvider exactly as the mission
does (Ultralytics, retina masks, CPU) on every image of a YOLO-seg split and
stores the selected mask, best confidence, latency and a SHA-256 of the RGB
input so the OpenCV 5 benchmark can prove it saw identical pixels.

Requires a sourced ROS 2 Jazzy shell (the adapter module imports rclpy):

    source /opt/ros/jazzy/setup.bash
    .venv-export/bin/python tools/reference_ultralytics.py --data data/cube_seg
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('YOLO_OFFLINE', 'True')

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
# Needs the private FYP checkout (GALBOT_FYP_REPO): ROS 2 package, recordings, checkpoint.
REPO = Path(os.environ['GALBOT_FYP_REPO']) if 'GALBOT_FYP_REPO' in os.environ else HERE.parent / 'galbot_mobile_manipulation'
sys.path.insert(0, str(REPO / 'packages/galbot_mobile_manipulation'))


def rgb_message(rgb):
    """Duck-typed sensor_msgs/Image (rgb8) accepted by rgb_from_message."""
    return SimpleNamespace(encoding='rgb8', width=rgb.shape[1], height=rgb.shape[0],
                           step=rgb.shape[1] * 3, data=rgb.tobytes(), header=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, required=True, help='YOLO-seg dataset root (images/<split>)')
    parser.add_argument('--split', default='val')
    parser.add_argument('--weights', type=Path,
                        default=REPO / 'packages/galbot_mobile_manipulation/models/cube_yolo11n_seg.pt')
    parser.add_argument('--target-class', default='cube')
    parser.add_argument('--confidence', type=float, default=0.35, help='mission launch default')
    parser.add_argument('--threads', type=int, default=None, help='torch intra-op threads (default: torch)')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, default=HERE / 'data/reference')
    args = parser.parse_args()

    import torch
    import ultralytics
    from galbot_mobile_manipulation.yolo_mask import YoloMaskProvider

    if args.threads:
        torch.set_num_threads(args.threads)
    provider = YoloMaskProvider(str(args.weights), args.target_class, args.confidence, 'cpu')
    paths = sorted((args.data / 'images' / args.split).glob('*.png'))
    if not paths:
        raise SystemExit(f'No PNG images in {args.data / "images" / args.split}')
    images = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f'Cannot read {path}')
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    for rgb in images[:args.warmup]:
        provider(rgb_message(rgb))

    masks, confidences, latency, digests = [], [], [], []
    for rgb in images:
        started = time.perf_counter()
        mask = provider(rgb_message(rgb))
        latency.append(1000 * (time.perf_counter() - started))
        masks.append(np.packbits(mask != 0, axis=None))
        best = provider.last_confidence.get('best_confidence')
        confidences.append(np.nan if best is None else best)
        digests.append(hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'{args.data.name}_{args.split}_ultralytics'
    np.savez_compressed(args.output / f'{stem}.npz', names=np.array([p.name for p in paths]),
                        shapes=np.array([im.shape[:2] for im in images]),
                        masks=np.array(masks, dtype=object), confidences=np.array(confidences),
                        latency_ms=np.array(latency), rgb_sha256=np.array(digests))
    meta = dict(dataset=args.data.name, split=args.split, images=len(paths),
                weights=args.weights.name, target_class=args.target_class, confidence=args.confidence,
                device='cpu', torch_threads=torch.get_num_threads(), warmup=args.warmup,
                latency_scope='YoloMaskProvider.__call__ (rgb8 decode, RGB->BGR, Ultralytics predict, '
                              'selection, mask geometry)',
                versions=dict(python=platform.python_version(), ultralytics=ultralytics.__version__,
                              torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__),
                cpu=platform.processor() or platform.machine())
    (args.output / f'{stem}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))
    print(f'median latency {np.median(latency):.1f} ms; detections {int(np.isfinite(confidences).sum())}/{len(paths)}')


if __name__ == '__main__':
    main()

"""Compare OpenCV 5 cv2.dnn YOLO11-seg masks with the Ultralytics reference and ground truth.

Inputs are the same PNG files the reference used; each decoded RGB array is
hashed and checked against the reference digest before comparison.

    .venv-cv5/bin/python tools/benchmark_dnn.py --data data/cube_seg
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.metrics import centroid_error, mask_iou, polygon_mask, summarise  # noqa: E402
from galbot_cv5.yolo_seg_dnn import YoloSegDnn  # noqa: E402


def load_reference(path):
    data = np.load(path, allow_pickle=True)
    shapes = data['shapes']
    masks = [np.unpackbits(np.asarray(packed, dtype=np.uint8), count=int(h * w)).reshape(h, w)
             for packed, (h, w) in zip(data['masks'], shapes)]
    return dict(names=list(data['names']), masks=masks, confidences=data['confidences'],
                latency_ms=data['latency_ms'], rgb_sha256=list(data['rgb_sha256']))


def group_of(name):
    return 'wrist' if '_wrist_' in name else 'head' if '_head_' in name else 'other'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--onnx', type=Path, default=HERE / 'models/cube_yolo11n_seg.onnx')
    parser.add_argument('--reference', type=Path, default=None)
    parser.add_argument('--confidence', type=float, default=0.35)
    parser.add_argument('--engines', default='new,classic')
    parser.add_argument('--threads', type=int, default=None, help='cv2.setNumThreads (default: OpenCV)')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()

    reference_path = args.reference or HERE / f'data/reference/{args.data.name}_{args.split}_ultralytics.npz'
    reference = load_reference(reference_path)
    reference_meta = json.loads(reference_path.with_suffix('.json').read_text())
    label_dir = args.data / 'labels' / args.split
    images = []
    for name, digest in zip(reference['names'], reference['rgb_sha256']):
        bgr = cv2.imread(str(args.data / 'images' / args.split / name), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() != digest:
            raise SystemExit(f'{name}: decoded RGB differs from the reference input')
        truth = polygon_mask((label_dir / (Path(name).stem + '.txt')).read_text(), rgb.shape)
        images.append((name, rgb, truth))

    ultra_rows = []
    for (name, _, truth), ref_mask, ref_conf, ref_ms in zip(
            images, reference['masks'], reference['confidences'], reference['latency_ms']):
        ultra_rows.append(dict(image=name, truth_iou=mask_iou(ref_mask, truth),
                               truth_centroid_px=centroid_error(ref_mask, truth),
                               detected=bool(ref_mask.any()), truth_present=bool(truth.any()),
                               confidence=None if np.isnan(ref_conf) else float(ref_conf),
                               latency_ms=float(ref_ms)))

    report = dict(
        dataset=args.data.name, split=args.split, images=len(images), confidence=args.confidence,
        input_check='sha256 of every decoded RGB input matches the Ultralytics reference',
        onnx=json.loads(args.onnx.with_suffix('.json').read_text()) if args.onnx.with_suffix('.json').exists()
        else str(args.onnx),
        reference=reference_meta,
        environment=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                         cv2_threads=cv2.getNumThreads() if args.threads is None else args.threads,
                         cpu=platform.processor() or platform.machine()),
        ultralytics=dict(ground_truth=accuracy(ultra_rows),
                         latency_ms=summarise([r['latency_ms'] for r in ultra_rows])),
        opencv5={})

    args.output.mkdir(parents=True, exist_ok=True)
    for engine in [e.strip() for e in args.engines.split(',') if e.strip()]:
        runner = YoloSegDnn(args.onnx, conf_threshold=args.confidence, engine=engine, threads=args.threads)
        for _, rgb, _ in images[:args.warmup]:
            runner.best_mask(rgb)
        rows = []
        for (name, rgb, truth), ref_mask, ref_conf, ultra in zip(
                images, reference['masks'], reference['confidences'], ultra_rows):
            started = time.perf_counter()
            mask, score, prediction = runner.best_mask(rgb)
            total = 1000 * (time.perf_counter() - started)
            rows.append(dict(
                image=name, group=group_of(name), detected=bool(mask.any()), truth_present=bool(truth.any()),
                reference_detected=bool(ref_mask.any()),
                confidence=score, reference_confidence=ultra['confidence'],
                confidence_abs_diff=None if score is None or ultra['confidence'] is None
                else abs(score - ultra['confidence']),
                reference_iou=mask_iou(mask, ref_mask),
                reference_centroid_px=centroid_error(mask, ref_mask),
                reference_pixel_disagreement=int(np.count_nonzero((mask != 0) != (ref_mask != 0))),
                truth_iou=mask_iou(mask, truth), truth_centroid_px=centroid_error(mask, truth),
                forward_ms=prediction.timings_ms['forward_ms'],
                postprocess_ms=prediction.timings_ms['postprocess_ms'], latency_ms=total))
        both = [r for r in rows if r['detected'] and r['reference_detected']]
        report['opencv5'][engine] = dict(
            agreement=dict(
                detection_decision_agreement=sum(r['detected'] == r['reference_detected'] for r in rows),
                images=len(rows), both_detected=len(both),
                only_opencv5=[r['image'] for r in rows if r['detected'] and not r['reference_detected']],
                only_ultralytics=[r['image'] for r in rows if r['reference_detected'] and not r['detected']],
                mask_iou_vs_ultralytics=summarise([r['reference_iou'] for r in both]),
                centroid_px_vs_ultralytics=summarise([r['reference_centroid_px'] for r in both]),
                pixel_disagreement=summarise([r['reference_pixel_disagreement'] for r in both]),
                confidence_abs_diff=summarise([r['confidence_abs_diff'] for r in both])),
            ground_truth=accuracy(rows),
            latency_ms=dict(total=summarise([r['latency_ms'] for r in rows]),
                            forward=summarise([r['forward_ms'] for r in rows]),
                            postprocess=summarise([r['postprocess_ms'] for r in rows])))
        stem = f'{args.data.name}_{args.split}_opencv5_{engine}'
        with open(args.output / f'{stem}.csv', 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    out = args.output / f'{args.data.name}_{args.split}_benchmark.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('ultralytics', 'opencv5')}, indent=2))
    print(f'wrote {out}')


def accuracy(rows):
    """Recall / false positives / IoU / centroid error against the polygon ground truth, per camera."""
    result = {}
    for group in ('head', 'wrist', 'all'):
        subset = [r for r in rows if group == 'all' or group_of(r['image']) == group]
        positives = [r for r in subset if r['truth_present']]
        found = [r for r in positives if r['detected']]
        result[group] = dict(
            positive_images=len(positives), found=len(found),
            recall=len(found) / len(positives) if positives else None,
            empty_images=len(subset) - len(positives),
            false_positives=sum(r['detected'] for r in subset if not r['truth_present']),
            mask_iou=summarise([r['truth_iou'] for r in found]),
            centroid_px=summarise([r['truth_centroid_px'] for r in found]))
    return result


if __name__ == '__main__':
    main()

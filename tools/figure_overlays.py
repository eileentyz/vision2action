"""Publication figures from the sanitised FYP bundle: OpenCV 5 cv2.dnn mask vs ground truth.

Runs the sanitised ONNX through YoloSegDnn on build/upload/headless_sweep.npz and draws
the dnn mask outline (blue) and the MuJoCo ground-truth outline (orange). Per-sample
numbers in the captions come from the AWS Lambda report, so the figure matches what the
cloud measured. No account identifiers or local paths are drawn.

    .venv-cv5/bin/python tools/figure_overlays.py
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.bundle import load_bundle  # noqa: E402
from galbot_cv5.yolo_seg_dnn import YoloSegDnn  # noqa: E402

DNN_BGR = (214, 120, 42)      # #2a78d6
GT_BGR = (52, 104, 235)       # #eb6834
INK = (11, 11, 11)
MUTED = (78, 81, 82)
SURFACE = (251, 252, 252)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PICKS = ('yaw-45-1#1', 'yaw+0-1#1', 'yaw+30-1#1', 'yaw+45-1#1')


def outline(image, mask, color, scale, thickness=2):
    big = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((big > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def text(image, label, org, scale=0.6, color=INK, thickness=1):
    cv2.putText(image, label, org, FONT, scale, color, thickness, cv2.LINE_AA)


def crop_panel(sample, dnn_mask, caption, half=(48, 36), scale=7):
    ys, xs = np.nonzero(sample['gt_left'])
    cx, cy = int(xs.mean()), int(ys.mean())
    h, w = sample['gt_left'].shape
    x0 = int(np.clip(cx - half[0], 0, w - 2 * half[0]))
    y0 = int(np.clip(cy - half[1], 0, h - 2 * half[1]))
    window = (slice(y0, y0 + 2 * half[1]), slice(x0, x0 + 2 * half[0]))
    bgr = cv2.cvtColor(sample['left'][window], cv2.COLOR_RGB2BGR)
    panel = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    outline(panel, sample['gt_left'][window], GT_BGR, scale, 3)
    outline(panel, dnn_mask[window], DNN_BGR, scale, 2)
    framed = np.full((panel.shape[0] + 44, panel.shape[1], 3), SURFACE, np.uint8)
    framed[44:] = panel
    text(framed, caption, (8, 30), 0.65)
    return framed


def legend(width):
    bar = np.full((64, width, 3), SURFACE, np.uint8)
    text(bar, 'OpenCV 5 cv2.dnn YOLO11n-seg on re-rendered FYP head-camera views (left camera, cropped)',
         (12, 24), 0.6)
    cv2.line(bar, (12, 47), (52, 47), DNN_BGR, 3, cv2.LINE_AA)
    text(bar, 'cv2.dnn mask', (60, 53), 0.55, MUTED)
    cv2.line(bar, (210, 47), (250, 47), GT_BGR, 3, cv2.LINE_AA)
    text(bar, 'ground truth (MuJoCo segmentation)', (258, 53), 0.55, MUTED)
    return bar


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, default=HERE / 'build/upload/headless_sweep.npz')
    parser.add_argument('--model', type=Path, default=HERE / 'models/cube_yolo11n_seg.onnx')
    parser.add_argument('--report', type=Path, default=HERE / 'results/cloud/fyp-headless-sweep-dnn/report.json')
    parser.add_argument('--output', type=Path, default=HERE / 'docs/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    samples, _ = load_bundle(args.bundle)
    by_id = {s['id'].split('/')[-1]: s for s in samples}
    cloud = {r['id'].split('/')[-1]: r for r in
             json.loads(args.report.read_text())['bundles'][0]['results']['dnn']['samples']}
    runner = YoloSegDnn(args.model, input_shape=(480, 640))

    panels = []
    for pick in PICKS:
        sample, row = by_id[pick], cloud[pick]
        mask = runner.best_mask(sample['left'])[0]
        caption = (f"true yaw {row['truth_yaw_deg']:+.0f} deg | IoU {row['iou_left']:.3f} | "
                   f"yaw error {abs(row['yaw_error_deg']):.2f} deg")
        panels.append(crop_panel(sample, mask, caption))
    gap = np.full((panels[0].shape[0], 12, 3), SURFACE, np.uint8)
    rows = [np.hstack([panels[0], gap, panels[1]]), np.hstack([panels[2], gap, panels[3]])]
    hgap = np.full((12, rows[0].shape[1], 3), SURFACE, np.uint8)
    grid = np.vstack([legend(rows[0].shape[1]), rows[0], hgap, rows[1]])
    cv2.imwrite(str(args.output / 'fig_dnn_masks_vs_ground_truth.png'), grid)

    sample = by_id['yaw+30-1#1']
    views = []
    for side in ('left', 'right'):
        bgr = cv2.cvtColor(sample[side], cv2.COLOR_RGB2BGR)
        outline(bgr, runner.best_mask(sample[side])[0], DNN_BGR, 1, 2)
        text(bgr, f'head {side} camera', (12, 28), 0.7, (255, 255, 255), 2)
        views.append(bgr)
    stereo = np.hstack([views[0], np.full((480, 8, 3), SURFACE, np.uint8), views[1]])
    cv2.imwrite(str(args.output / 'fig_stereo_pair_dnn.png'), stereo)
    print('\n'.join(str(p.relative_to(HERE)) for p in sorted(args.output.glob('fig_*.png'))))


if __name__ == '__main__':
    main()

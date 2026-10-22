"""Network forward-pass latency on one fixed input tensor, for one runtime and thread count.

Runs in either environment and prints one JSON line, so a driver can alternate
runtimes (they cannot share a process: OpenCV 5 needs NumPy 2, the FYP's
Ultralytics/torch stack is pinned to NumPy 1.26):

    .venv-cv5/bin/python    tools/latency_probe.py --runtime opencv-new --threads 8
    .venv-export/bin/python tools/latency_probe.py --runtime torch      --threads 8

Only the network is timed (no decode/NMS/masks); the input is the letterboxed
blob of one validation image saved by ``--save-blob`` (identical bytes for all).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runtime', required=True, choices=['opencv-new', 'opencv-classic', 'torch'])
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--blob', type=Path, default=HERE / 'build/latency_blob.npy')
    parser.add_argument('--onnx', type=Path, default=HERE / 'models/cube_yolo11n_seg.onnx')
    parser.add_argument('--weights', type=Path, default=HERE / 'build/cube_yolo11n_seg.pt')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    blob = np.load(args.blob)

    if args.runtime.startswith('opencv'):
        import cv2
        cv2.setNumThreads(args.threads)
        engine = cv2.dnn.ENGINE_NEW if args.runtime == 'opencv-new' else cv2.dnn.ENGINE_CLASSIC
        net = cv2.dnn.readNetFromONNX(str(args.onnx), engine=engine)
        names = list(net.getUnconnectedOutLayersNames())

        def run():
            net.setInput(blob)
            net.forward(names)
        version = cv2.__version__
    else:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(args.threads)
        model = YOLO(str(args.weights)).model.float().eval().fuse()
        tensor = torch.from_numpy(blob)

        def run():
            with torch.inference_mode():
                model(tensor)
        version = torch.__version__

    for _ in range(args.warmup):
        run()
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        run()
        samples.append(1000 * (time.perf_counter() - started))
    samples = np.asarray(samples)
    print(json.dumps(dict(runtime=args.runtime, version=version, threads=args.threads,
                          input_shape=list(blob.shape), iterations=args.iterations,
                          median_ms=float(np.median(samples)), p05_ms=float(np.percentile(samples, 5)),
                          p95_ms=float(np.percentile(samples, 95)), cpu=platform.processor() or platform.machine())))


if __name__ == '__main__':
    main()

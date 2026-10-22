"""Serve cube masks from OpenCV 5 cv2.dnn to the live FYP mission over a Unix socket (OpenCV 5 env).

The robot's object_pointcloud node runs with yolo_backend:=opencv5 and sends each
rgb8 frame here; the reply is the highest-scoring cube mask, selected exactly as
YoloMaskProvider does (stage 1 parity). Frames are not stored unless --record DIR
is given: then each served frame is written to DIR as a JPEG with the OpenCV 5 mask,
score and timing drawn on it (local only, for the demo video; tools/make_demo_media.py).

    env -u PYTHONPATH .venv-cv5/bin/python tools/opencv5_mask_server.py \\
        --model models/cube_yolo11n_seg.onnx --threads 4
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from galbot_cv5.yolo_seg_dnn import YoloSegDnn  # noqa: E402
from galbot_cv5 import opencv5_protocol as wire  # noqa: E402  (same file as integration/ros2/)


def respond(connection, status, payload=b'', score=math.nan, forward_ms=0.0, total_ms=0.0):
    connection.sendall(wire.RESPONSE.pack(wire.MAGIC, status, score, forward_ms, total_ms, len(payload)) + payload)


def save_frame(record, index, rgb, mask, score, forward_ms, total_ms):
    """The frame exactly as the robot sent it, with the mask OpenCV 5 returned for it."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if score is not None:
        tint = bgr.copy()
        tint[mask != 0] = (0, 200, 255)
        bgr = cv2.addWeighted(tint, 0.45, bgr, 0.55, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 200, 255), 2)
    height, width = bgr.shape[:2]
    lines = [f'OpenCV {cv2.__version__} cv2.dnn  frame {index}  {width}x{height}',
             (f'cube {score:.3f}' if score is not None else 'no cube') +
             f'   forward {forward_ms:.1f} ms   total {total_ms:.1f} ms']
    cv2.rectangle(bgr, (0, 0), (width, 62), (0, 0, 0), cv2.FILLED)
    for row, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 24 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(record / f'frame_{index:05d}_{width}x{height}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def serve_client(connection, runner, hello, lock, stats, log_every, record):
    with connection:
        while True:
            try:
                header = wire.receive_exactly(connection, wire.REQUEST.size)
            except ConnectionError:
                return
            magic, kind, height, width, confidence, class_id = wire.REQUEST.unpack(header)
            if magic != wire.MAGIC or kind not in (wire.HELLO, wire.PREDICT):
                return
            if kind == wire.HELLO:
                respond(connection, wire.OK, json.dumps(hello).encode())
                continue
            if not 0 < height * width <= wire.MAX_PIXELS:
                respond(connection, wire.ERROR, f'bad image size {height}x{width}'.encode())
                return
            rgb = np.frombuffer(wire.receive_exactly(connection, height * width * 3), np.uint8)
            rgb = rgb.reshape(height, width, 3)
            started = time.perf_counter()
            try:
                with lock:
                    runner.conf_threshold = float(confidence)
                    mask, score, prediction = runner.best_mask(rgb, class_id=int(class_id))
            except Exception as error:      # report to the client; keep serving
                respond(connection, wire.ERROR, f'{type(error).__name__}: {error}'.encode())
                continue
            total_ms = 1000 * (time.perf_counter() - started)
            stats['frames'] += 1
            stats['detections'] += score is not None
            if log_every and stats['frames'] % log_every == 1:
                # Visible proof during the mission that the robot's masks come from OpenCV 5.
                print(f"opencv5_mask_server: frame {stats['frames']} {width}x{height}: "
                      f"{'cube %.3f' % score if score is not None else 'no cube'}, "
                      f"cv2.dnn forward {prediction.timings_ms['forward_ms']:.1f} ms, total {total_ms:.1f} ms",
                      flush=True)
            respond(connection, wire.OK, wire.pack_mask(mask), math.nan if score is None else score,
                    prediction.timings_ms['forward_ms'], total_ms)
            if record is not None:      # after the reply, so the robot does not wait for the JPEG
                save_frame(record, stats['frames'], rgb, mask, score, prediction.timings_ms['forward_ms'], total_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', type=Path, default=HERE / 'models/cube_yolo11n_seg.onnx')
    parser.add_argument('--socket', default=wire.default_socket_path())
    parser.add_argument('--engine', choices=['new', 'classic'], default='new')
    parser.add_argument('--log-every', type=int, default=25, help='print one frame line every N frames (0: off)')
    parser.add_argument('--record', type=Path, help='save every served frame with its mask here (local only)')
    parser.add_argument('--threads', type=int, default=4,
                        help='cv2 threads; the simulator and mission share the CPU')
    args = parser.parse_args()

    if args.record is not None:
        args.record.mkdir(parents=True, exist_ok=True)
    runner = YoloSegDnn(args.model, engine=args.engine, threads=args.threads)
    sidecar = args.model.with_suffix('.json')
    names = json.loads(sidecar.read_text())['names'] if sidecar.exists() else {'0': 'cube'}
    hello = dict(names=names, opencv=cv2.__version__, numpy=np.__version__, engine=args.engine,
                 threads=cv2.getNumThreads(), input_shape=list(runner.input_shape), model=args.model.name,
                 model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    # Warm up so the first live frame does not pay for graph initialisation.
    runner.best_mask(np.zeros((*runner.input_shape, 3), np.uint8))

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    os.chmod(args.socket, 0o600)
    server.listen(4)
    lock, stats = threading.Lock(), dict(frames=0, detections=0)

    def stop(*_):
        print(f"opencv5_mask_server: {stats['frames']} frames, {stats['detections']} with a cube", flush=True)
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f'opencv5_mask_server: OpenCV {cv2.__version__} cv2.dnn ({args.engine}, {hello["threads"]} threads), '
          f'{args.model.name} {hello["input_shape"]}, listening on {args.socket}', flush=True)
    while True:
        connection, _ = server.accept()
        threading.Thread(target=serve_client, args=(connection, runner, hello, lock, stats, args.log_every, args.record), daemon=True).start()


if __name__ == '__main__':
    main()

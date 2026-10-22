"""YOLO segmentation through OpenCV 5 cv2.dnn, served by a separate local process.

Drop-in for YoloMaskProvider (same mask, geometry, debug overlay and confidence
metrics). The network runs in competition/tools/opencv5_mask_server.py under the
OpenCV 5 environment; this side only sends rgb8 frames over a Unix socket.
"""

import json
import math
import socket
import threading
import time

import numpy as np

from galbot_mobile_manipulation import opencv5_protocol as wire
from galbot_mobile_manipulation.red_cube_detector import rgb_from_message
from galbot_mobile_manipulation.yolo_mask import debug_overlay, mask_geometry


class OpenCV5MaskProvider:
    def __init__(self, socket_path='', target_class='cube', confidence_threshold=0.5,
                 debug_callback=None, confidence_callback=None, connect_timeout=20.0,
                 request_timeout=5.0):
        if not target_class:
            raise ValueError('OpenCV 5 target_class must be nonempty')
        if not np.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
            raise ValueError('confidence_threshold must be in (0, 1]')
        self.socket_path = socket_path or wire.default_socket_path()
        self.request_timeout = request_timeout
        self.confidence = confidence_threshold
        self.debug_callback = debug_callback
        self.confidence_callback = confidence_callback
        self.geometry = None
        self.last_confidence = None
        self.connection = None
        self.lock = threading.Lock()
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self.server = self._hello()
                break
            except OSError as error:
                if time.monotonic() > deadline:
                    raise ValueError(f'OpenCV 5 mask server not reachable at {self.socket_path} ({error}); '
                                     'start competition/tools/opencv5_mask_server.py first') from error
                time.sleep(0.5)
        names = {int(k): v for k, v in self.server['names'].items()}
        matches = [k for k, name in names.items() if name == target_class or str(k) == target_class]
        if not matches:
            raise ValueError(f'unknown target_class {target_class!r}; available names: {list(names.values())}')
        self.class_id = matches[0]
        self.class_name = names[self.class_id]

    def _connect(self):
        if self.connection is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.request_timeout)
            try:
                connection.connect(self.socket_path)
            except OSError:
                connection.close()
                raise
            self.connection = connection
        return self.connection

    def _exchange(self, kind, height=0, width=0, class_id=0, pixels=b''):
        connection = self._connect()
        try:
            connection.sendall(wire.REQUEST.pack(wire.MAGIC, kind, height, width, self.confidence, class_id)
                               + pixels)
            magic, status, score, forward_ms, total_ms, length = wire.RESPONSE.unpack(
                wire.receive_exactly(connection, wire.RESPONSE.size))
            if magic != wire.MAGIC:
                raise ConnectionError('OpenCV 5 mask server sent an invalid response')
            payload = wire.receive_exactly(connection, length)
        except (OSError, ConnectionError):
            # Reconnect on the next call; the server may have restarted.
            connection.close()
            self.connection = None
            raise
        if status != wire.OK:
            raise ValueError(payload.decode(errors='replace'))
        return score, forward_ms, total_ms, payload

    def _hello(self):
        with self.lock:
            return json.loads(self._exchange(wire.HELLO)[3])

    def __call__(self, rgb_message):
        started = time.monotonic()
        rgb = rgb_from_message(rgb_message)
        height, width = rgb.shape[:2]
        self.geometry = None
        label = f'OpenCV 5: no {self.class_name} detected'
        try:
            with self.lock:
                score, forward_ms, server_ms, payload = self._exchange(
                    wire.PREDICT, height, width, self.class_id, np.ascontiguousarray(rgb).tobytes())
            mask = wire.unpack_mask(payload, height, width)
        except Exception as error:
            self.last_confidence = dict(target_class=self.class_name, threshold=self.confidence,
                                        best_confidence=None, inference_calls=1,
                                        accepted_mask_produced=False, inference_error=str(error),
                                        backend='opencv5_dnn',
                                        inference_total_ms=1000 * (time.monotonic() - started))
            if self.confidence_callback is not None:
                self.confidence_callback(rgb_message.header, self.last_confidence)
            raise ValueError(f'OpenCV 5 inference failed: {error}') from error
        best = None if math.isnan(score) else float(score)
        if best is not None:
            label = f'OpenCV 5 {self.class_name}: {best:.3f}'
        metrics = dict(target_class=self.class_name, threshold=self.confidence,
                       best_confidence=best, confidence_scope='normal_thresholded_output',
                       inference_calls=1, accepted_mask_produced=bool(mask.any()),
                       accepted_mask_pixels=int(np.count_nonzero(mask)),
                       backend='opencv5_dnn', opencv=self.server.get('opencv'),
                       forward_ms=float(forward_ms), server_total_ms=float(server_ms),
                       inference_total_ms=1000 * (time.monotonic() - started))
        self.last_confidence = metrics
        if self.confidence_callback is not None:
            self.confidence_callback(rgb_message.header, metrics)
        self.geometry = mask_geometry(mask)
        if self.debug_callback is not None:
            self.debug_callback(rgb_message.header, debug_overlay(rgb, mask, self.geometry, label))
        return mask

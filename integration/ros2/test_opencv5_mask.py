import json
import math
import socket
import threading

import numpy as np
import pytest
from rclpy.time import Time

from galbot_mobile_manipulation import opencv5_protocol as wire
from galbot_mobile_manipulation.object_mask import create_mask_provider, validate_object_mask
from galbot_mobile_manipulation.opencv5_mask import OpenCV5MaskProvider
from test_rgbd_cube_estimator import bundle


class FakeServer:
    """Speaks the server side of the wire format; masks the red pixels it receives."""

    def __init__(self, path, fail=False):
        self.requests, self.fail = [], fail
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen(2)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            with connection:
                while True:
                    try:
                        header = wire.receive_exactly(connection, wire.REQUEST.size)
                    except ConnectionError:
                        break
                    magic, kind, height, width, confidence, class_id = wire.REQUEST.unpack(header)
                    self.requests.append((kind, height, width, round(confidence, 3), class_id))
                    if kind == wire.HELLO:
                        payload = json.dumps(dict(names={'0': 'cube'}, opencv='5.0.0')).encode()
                        connection.sendall(wire.RESPONSE.pack(wire.MAGIC, wire.OK, math.nan, 0, 0, len(payload))
                                           + payload)
                        continue
                    rgb = np.frombuffer(wire.receive_exactly(connection, height * width * 3), np.uint8)
                    if self.fail:
                        payload = b'cv2.error: boom'
                        connection.sendall(wire.RESPONSE.pack(wire.MAGIC, wire.ERROR, math.nan, 0, 0, len(payload))
                                           + payload)
                        continue
                    mask = rgb.reshape(height, width, 3)[..., 0] > 200
                    payload = wire.pack_mask(mask)
                    score = 0.97 if mask.any() else math.nan
                    connection.sendall(wire.RESPONSE.pack(wire.MAGIC, wire.OK, score, 12.5, 14.0, len(payload))
                                       + payload)

    def close(self):
        self.listener.close()


@pytest.fixture
def server(tmp_path):
    fake = FakeServer(tmp_path / 'cv5.sock')
    yield fake
    fake.close()


def test_mask_geometry_debug_and_metrics_match_yolo_provider(server, tmp_path):
    debug, metrics = [], []
    provider = create_mask_provider('yolo', backend='opencv5', opencv5_socket=str(tmp_path / 'cv5.sock'),
                                    model_path='ignored.pt', target_class='cube', confidence_threshold=0.35,
                                    device='auto', debug_callback=lambda h, p: debug.append(p),
                                    confidence_callback=lambda h, m: metrics.append(m))
    assert isinstance(provider, OpenCV5MaskProvider)
    rgb, _, _ = bundle(Time(seconds=1).to_msg())
    mask = provider(rgb)
    validate_object_mask(mask, (80, 100))
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 40 * 40 and mask[20:60, 30:70].all()
    assert provider.geometry['centroid'] == pytest.approx((49.5, 39.5))
    assert debug[0].shape == (80, 100, 3)
    assert metrics[0]['backend'] == 'opencv5_dnn' and metrics[0]['best_confidence'] == pytest.approx(0.97)
    assert metrics[0]['forward_ms'] == pytest.approx(12.5) and metrics[0]['accepted_mask_pixels'] == 1600
    assert server.requests[-1] == (wire.PREDICT, 80, 100, 0.35, 0)


def test_no_detection_and_server_errors(tmp_path):
    fake = FakeServer(tmp_path / 'cv5.sock')
    provider = OpenCV5MaskProvider(str(tmp_path / 'cv5.sock'), confidence_threshold=0.35)
    rgb, _, _ = bundle(Time(seconds=1).to_msg())
    rgb.data = bytes(len(rgb.data))
    assert not provider(rgb).any() and provider.geometry['centroid'] is None
    assert provider.last_confidence['best_confidence'] is None
    fake.fail = True
    with pytest.raises(ValueError, match='boom'):
        provider(rgb)
    assert provider.last_confidence['inference_error'] == 'cv2.error: boom'
    fake.close()


def test_unknown_class_missing_server_and_backend(server, tmp_path):
    with pytest.raises(ValueError, match='unknown target_class'):
        OpenCV5MaskProvider(str(tmp_path / 'cv5.sock'), target_class='tote')
    with pytest.raises(ValueError, match='not reachable'):
        OpenCV5MaskProvider(str(tmp_path / 'absent.sock'), connect_timeout=0)
    with pytest.raises(ValueError, match='yolo_backend'):
        create_mask_provider('yolo', backend='tensorrt')


def test_mask_packing_round_trip():
    mask = np.random.default_rng(0).random((37, 53)) > 0.5
    np.testing.assert_array_equal(wire.unpack_mask(wire.pack_mask(mask), 37, 53), mask.astype(np.uint8) * 255)
